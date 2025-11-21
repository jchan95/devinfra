from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from datetime import datetime
from typing import Optional, List, Dict
from supabase import create_client, Client
import time
import re
import json
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic


# Helper Functions - Chunking and Embeddings

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[dict]:
    """
    Break text into overlapping chunks.
    
    Analogy: Like cutting a long sandwich into bite-sized pieces,
    with each piece slightly overlapping so you don't lose context.
    """
    chunks = []
    
    # Split by paragraphs first (better semantic boundaries)
    paragraphs = text.split('\n\n')
    
    current_chunk = ""
    chunk_index = 0
    
    for para in paragraphs:
        # If adding this paragraph doesn't exceed chunk size, add it
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\n\n"
        else:
            # Save current chunk
            if current_chunk.strip():
                chunks.append({
                    "chunk_index": chunk_index,
                    "text": current_chunk.strip(),
                    "tokens": len(current_chunk.split())  # Rough estimate
                })
                chunk_index += 1
            
            # Start new chunk with overlap from previous
            words = current_chunk.split()
            overlap_text = " ".join(words[-overlap:]) if len(words) > overlap else ""
            current_chunk = overlap_text + " " + para + "\n\n"
    
    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append({
            "chunk_index": chunk_index,
            "text": current_chunk.strip(),
            "tokens": len(current_chunk.split())
        })
    
    return chunks

# Vector Search - Find Relevant Chunks

def search_similar_chunks(question: str, workspace_id: str, top_k: int = 3) -> List[dict]:
    """
    Find the most relevant chunks using vector similarity search.
    
    How it works:
    1. Convert question to embedding (vector)
    2. Use pgvector to find chunks with similar embeddings
    3. Return top_k most similar chunks
    
    Analogy: Like using a metal detector to find the exact ingredients
    you need from a huge refrigerator.
    """
    # Create embedding for the question
    question_embedding = create_embedding(question)
    
    # Search for similar chunks using pgvector
    # This uses cosine similarity to find chunks with similar meaning
    result = supabase.rpc(
        'match_chunks',
        {
            'query_embedding': question_embedding,
            'match_threshold': 0.5,  # Minimum similarity score (0-1)
            'match_count': top_k,
            'workspace_filter': workspace_id
        }
    ).execute()
    
    return result.data if result.data else []

def create_embedding(text: str) -> List[float]:
    """
    Convert text to a vector using OpenAI's embedding model.
    
    Analogy: Like creating a unique fingerprint for each piece of text
    that captures its meaning in numbers.
    """
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# WEEK 2: LLM JUDGE FUNCTION

def judge_answer_with_llm(
    question: str,
    answer: str,
    retrieved_contexts: List[str]
) -> Dict:
    """
    Use Gemini as a judge to score answer quality.
    
    This is like a food critic scoring a dish on:
    - Relevance: Did they answer what was asked?
    - Faithfulness: Is it supported by the context?
    - Completeness: Is it thorough?
    
    Returns dict with scores (0-5) and explanation
    """
    
    # Build the context string from retrieved chunks
    context_text = "\n\n---\n\n".join([
        f"Context {i+1}:\n{ctx}" 
        for i, ctx in enumerate(retrieved_contexts)
    ])
    
    # Create the judging prompt
    judge_prompt = f"""You are an expert evaluator assessing the quality of answers from a RAG system.

**Question Asked:**
{question}

**Retrieved Context:**
{context_text}

**Answer Provided:**
{answer}

**Your Task:**
Evaluate the answer on three dimensions (0-5 scale, where 5 is best):

1. **Relevance (0-5)**: Does the answer directly address the question?
   - 5: Perfectly addresses the question
   - 3: Partially addresses the question
   - 1: Barely related
   - 0: Completely irrelevant

2. **Faithfulness (0-5)**: Is the answer supported by the context?
   - 5: Fully supported, no hallucinations
   - 3: Mostly supported with minor unsupported claims
   - 1: Many unsupported claims
   - 0: Completely hallucinated

3. **Completeness (0-5)**: Is the answer thorough?
   - 5: Comprehensive and detailed
   - 3: Basic answer, missing some details
   - 1: Minimal information
   - 0: No useful information

**Response Format:**
Provide ONLY a valid JSON object (no markdown, no extra text):
{{
    "relevance": <score 0-5>,
    "faithfulness": <score 0-5>,
    "completeness": <score 0-5>,
    "overall": <average of above three>,
    "explanation": "<brief 1-2 sentence explanation>"
}}

DO NOT include any text outside the JSON. DO NOT use markdown code blocks."""

        
    try:
        start_time = time.time()
        
        # Call OpenAI GPT-4o-mini to judge
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert evaluator of RAG system answers. Respond only with valid JSON."},
                {"role": "user", "content": judge_prompt}
            ],
            temperature=0
        )
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Parse the response
        response_text = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        # Parse JSON
        scores = json.loads(response_text)
        
        # Add metadata
        scores["judge_model"] = "gpt-4o-mini"
        scores["judge_latency_ms"] = latency_ms
        
        return scores
        
    except Exception as e:
        print(f"Error in LLM judge: {str(e)}")
        # Return neutral scores on error
        return {
            "relevance": 2.5,
            "faithfulness": 2.5,
            "completeness": 2.5,
            "overall": 2.5,
            "explanation": f"Error during judging: {str(e)}",
            "judge_model": "gpt-4o-mini",
            "judge_latency_ms": 0
        }

# Query Models - RAG Question/Answer

class QueryRequest(BaseModel):
    """What the user sends when asking a question"""
    workspace_id: str
    question: str
    top_k: int = 3  # How many chunks to retrieve

class QueryResponse(BaseModel):
    """What we return after answering the question"""
    question: str
    answer: str
    contexts: List[dict]  # The chunks that were used
    latency_ms: int

# Settings
class Settings(BaseSettings):
    SUPABASE_URL: str
    SUPABASE_KEY: str
    OPENAI_API_KEY: str
    GEMINI_API_KEY: str
    ANTHROPIC_API_KEY: str
    class Config:
        env_file = ".env"

settings = Settings()

# Database connection
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

# Initialize OpenAI client
openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

# Initialize Anthropic client
anthropic_client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

# FastAPI app
app = FastAPI(title="DevInfra RAG API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Local development
        "https://devinfra.netlify.app"  # Production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class WorkspaceCreate(BaseModel):
    name: str
    description: Optional[str] = None

class Workspace(BaseModel):
    id: str
    name: str
    description: Optional[str]
    created_at: datetime

class DocumentCreate(BaseModel):
    workspace_id: str
    source_type: str  # 'api_doc', 'runbook', 'readme', etc.
    service_name: Optional[str] = None
    title: str
    content: str  # The actual document text
    uri: Optional[str] = None  # URL or file path

class Document(BaseModel):
    id: str
    workspace_id: str
    source_type: str
    service_name: Optional[str]
    title: str
    uri: Optional[str]
    created_at: datetime

# EVALUATION MODELS

class PipelineConfigCreate(BaseModel):
    workspace_id: str
    name: str
    description: Optional[str] = None
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 3
    similarity_threshold: float = 0.5

class PipelineConfig(BaseModel):
    id: str
    workspace_id: str
    name: str
    description: Optional[str]
    parameters: Dict
    is_active: bool
    created_at: str

class EvalRunCreate(BaseModel):
    eval_set_id: str
    pipeline_config_id: str

class EvalRunStatus(BaseModel):
    eval_run_id: str
    status: str
    total_examples: int
    completed: int
    avg_scores: Optional[Dict[str, float]] = None

# Routes
@app.get("/")
def read_root():
    return {"message": "DevInfra RAG API is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/workspaces", response_model=Workspace, tags=["workspaces"])
async def create_workspace(workspace: WorkspaceCreate):
    """Create a new workspace"""
    try:
        result = supabase.table("workspaces").insert({
            "name": workspace.name,
            "description": workspace.description
        }).execute()
        
        return result.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/workspaces", response_model=List[Workspace], tags=["workspaces"])
async def list_workspaces():
    """Get all workspaces"""
    try:
        result = supabase.table("workspaces").select("*").execute()
        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/workspaces/{workspace_id}", response_model=Workspace, tags=["workspaces"])
async def get_workspace(workspace_id: str):
    """Get a specific workspace by ID"""
    try:
        result = supabase.table("workspaces").select("*").eq("id", workspace_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Workspace not found")
        
        return result.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/workspaces/{workspace_id}", tags=["workspaces"])
async def delete_workspace(workspace_id: str):
    """Delete a workspace"""
    try:
        result = supabase.table("workspaces").delete().eq("id", workspace_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Workspace not found")
        
        return {"message": "Workspace deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents", response_model=Document, tags=["documents"])
async def create_document(doc: DocumentCreate):
    """Upload and ingest a document with automatic chunking and embedding"""
    try:
        # Step 1: Insert the document
        result = supabase.table("documents").insert({
            "workspace_id": doc.workspace_id,
            "source_type": doc.source_type,
            "service_name": doc.service_name,
            "title": doc.title,
            "uri": doc.uri,
            "raw_text": doc.content
        }).execute()
        
        document = result.data[0]
        document_id = document["id"]
        
        # Step 2: Chunk the document
        chunks = chunk_text(doc.content)
        print(f"Created {len(chunks)} chunks for document {document_id}")
        
        # Step 3: For each chunk, create embedding and store
        for chunk in chunks:
            # Insert chunk
            chunk_result = supabase.table("chunks").insert({
                "document_id": document_id,
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "tokens": chunk["tokens"],
                "metadata": {}
            }).execute()
            
            chunk_id = chunk_result.data[0]["id"]
            
            # Create embedding
            embedding = create_embedding(chunk["text"])
            
            # Store embedding
            supabase.table("embeddings").insert({
                "chunk_id": chunk_id,
                "embedding": embedding,
                "model_name": "text-embedding-3-small"
            }).execute()
        
        print(f"✅ Document {document_id} ingested with {len(chunks)} chunks and embeddings")
        
        return document
    except Exception as e:
        print(f"❌ Error ingesting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", response_model=List[Document], tags=["documents"])
async def list_documents(workspace_id: Optional[str] = None):
    """Get all documents, optionally filtered by workspace"""
    try:
        query = supabase.table("documents").select("id, workspace_id, source_type, service_name, title, uri, created_at")
        
        if workspace_id:
            query = query.eq("workspace_id", workspace_id)
        
        result = query.execute()
        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{document_id}", response_model=Document, tags=["documents"])
async def get_document(document_id: str):
    """Get a specific document"""
    try:
        result = supabase.table("documents").select("id, workspace_id, source_type, service_name, title, uri, created_at").eq("id", document_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return result.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# RAG Query Endpoint
# ============================================

@app.post("/query", response_model=QueryResponse, tags=["query"])
async def query_rag(query: QueryRequest):
    """
    Answer a question using RAG (Retrieval Augmented Generation).
    
    Process:
    1. Convert question to vector embedding
    2. Search for similar chunks in the workspace
    3. Send chunks + question to LLM
    4. Return AI-generated answer
    """
    start_time = time.time()
    
    try:
        # Step 1: Find relevant chunks using vector search
        print(f"🔍 Searching for chunks relevant to: {query.question}")
        chunks = search_similar_chunks(query.question, query.workspace_id, query.top_k)
        
        if not chunks:
            raise HTTPException(
                status_code=404, 
                detail="No relevant context found. Try uploading documents first."
            )
        
        print(f"✅ Found {len(chunks)} relevant chunks")
        
        # Step 2: Build context from chunks
        context_text = "\n\n---\n\n".join([
            f"Source {i+1}:\n{chunk['text']}" 
            for i, chunk in enumerate(chunks)
        ])
        
        # Step 3: Create prompt for LLM
        prompt = f"""You are a helpful assistant answering questions based on documentation.

Context from documentation:
{context_text}

Question: {query.question}

Instructions:
- Answer the question based ONLY on the context provided above
- If the context doesn't contain enough information, say so
- Be concise and specific
- Cite which source(s) you used if relevant

Answer:"""
        
        # Step 4: Call OpenAI to generate answer
        print(f"🤖 Calling OpenAI to generate answer...")

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant answering questions based on documentation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        answer = response.choices[0].message.content
        
        print(f"✅ Answer generated")
    
        # Step 5: Log the query to database
        query_result = supabase.table("queries").insert({
            "workspace_id": query.workspace_id,
            "question": query.question,
            "source": "api"
        }).execute()
        
        query_id = query_result.data[0]["id"]
        
        # Step 6: Log the answer
        latency_ms = int((time.time() - start_time) * 1000)
        
        supabase.table("answers").insert({
            "query_id": query_id,
            "answer_text": answer,
            "model_name": "gpt-4o-mini",
            "latency_ms": latency_ms,
            "metadata": {"chunks_used": len(chunks)}
        }).execute()
        
        # Step 7: Return response
        return QueryResponse(
            question=query.question,
            answer=answer,
            contexts=[{
                "text": chunk["text"],
                "similarity": chunk["similarity"],
                "chunk_id": str(chunk["id"])
            } for chunk in chunks],
            latency_ms=latency_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error during query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# WEEK 2: EVALUATION ENDPOINTS
# ============================================

@app.post("/eval/run", response_model=EvalRunStatus)
async def run_evaluation(eval_run: EvalRunCreate):
    
    # Step 1: Create the eval_run record (start the inspection session)
    eval_run_record = supabase.table("eval_runs").insert({
        "eval_set_id": eval_run.eval_set_id,
        "pipeline_config_id": eval_run.pipeline_config_id,
        "status": "running"
    }).execute()
    
    eval_run_id = eval_run_record.data[0]["id"]
    
    # Step 2: Get the workspace_id from eval_set
    eval_set = supabase.table("eval_sets").select("*").eq("id", eval_run.eval_set_id).execute()
    workspace_id = eval_set.data[0]["workspace_id"]
    
    # Step 3: Get the pipeline config parameters
    config_result = supabase.table("pipeline_configs").select("*").eq("id", eval_run.pipeline_config_id).execute()
    pipeline_config = config_result.data[0]
    
    # Extract parameters
    top_k = pipeline_config["parameters"]["top_k"]
    
    # Step 4: Get all 15 test questions
    examples = supabase.table("eval_examples").select("*").eq("eval_set_id", eval_run.eval_set_id).execute()
    
    total_examples = len(examples.data)
    completed = 0
    all_scores = []
    
    print(f"\n🎯 Starting eval run: {total_examples} questions with config '{pipeline_config['name']}'")
    
    # Step 5: Run each question through RAG pipeline
    for i, example in enumerate(examples.data, 1):
        question = example["question"]
        print(f"\n📝 Question {i}/{total_examples}: {question[:60]}...")
        
        try:
            # 5a. Search for relevant chunks (your existing function!)
            chunks = search_similar_chunks(question, workspace_id, top_k=top_k)
            
            if not chunks or len(chunks) == 0:
                print(f"   ⚠️  No chunks found")
                continue
            
            print(f"   ✓ Found {len(chunks)} chunks")
            
            # 5b. Build context from chunks
            context_text = "\n\n---\n\n".join([
                f"Source {i+1}:\n{chunk['text']}" 
                for i, chunk in enumerate(chunks)
            ])
            
            # 5c. Generate answer with Gemini
            prompt = f"""You are a helpful assistant answering questions based on documentation.

Context from documentation:
{context_text}

Question: {question}

Instructions:
- Answer the question based ONLY on the context provided above
- If the context doesn't contain enough information, say so
- Be concise and specific
- Cite which source(s) you used if relevant

Answer:"""
            
            start_time = time.time()
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant answering questions based on documentation."},
                    {"role": "user", "content": prompt}
                ],
                    temperature=0
                )
            answer_text = response.choices[0].message.content
            latency_ms = int((time.time() - start_time) * 1000)
            
            print(f"   ✓ Generated answer ({latency_ms}ms)")
            
            # 5d. Save query to database
            query_record = supabase.table("queries").insert({
                "workspace_id": workspace_id,
                "pipeline_config_id": eval_run.pipeline_config_id,
                "question": question,
                "source": "eval"
            }).execute()
            
            query_id = query_record.data[0]["id"]
            
            # 5e. Save answer to database
            answer_record = supabase.table("answers").insert({
                "query_id": query_id,
                "answer_text": answer_text,
                "model_name": "gpt-4o-mini",
                "latency_ms": latency_ms,
                "used_pipeline_config_id": eval_run.pipeline_config_id,
                "metadata": {"chunks_used": len(chunks)}
            }).execute()
            
            answer_id = answer_record.data[0]["id"]
            
            # 5f. Save retrieval logs
            for rank, chunk in enumerate(chunks, 1):
                supabase.table("retrieval_logs").insert({
                    "query_id": query_id,
                    "chunk_id": chunk["id"],
                    "rank": rank,
                    "score": chunk["similarity"],
                    "pipeline_config_id": eval_run.pipeline_config_id,
                    "workspace_id": workspace_id
                }).execute()
            
            # 5g. JUDGE THE ANSWER (This is the new Week 2 magic!)
            context_texts = [chunk["text"] for chunk in chunks]
            scores = judge_answer_with_llm(question, answer_text, context_texts)
            
            print(f"   ⭐ Scores: Overall={scores['overall']}, Relevance={scores['relevance']}, Faithfulness={scores['faithfulness']}, Completeness={scores['completeness']}")
            
            # 5h. Save evaluation result
            supabase.table("eval_results").insert({
                "eval_run_id": eval_run_id,
                "eval_example_id": example["id"],
                "query_id": query_id,
                "answer_id": answer_id,
                "score_overall": scores["overall"],
                "score_relevance": scores["relevance"],
                "score_faithfulness": scores["faithfulness"],
                "score_completeness": scores["completeness"],
                "judge_model": scores["judge_model"],
                "judge_explanation": scores["explanation"],
                "judge_latency_ms": scores["judge_latency_ms"]
            }).execute()
            
            all_scores.append(scores)
            completed += 1

        # Wait to avoid rate limits
           ## time.sleep(1)  # 1 second delay between questions

            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            continue
    
    # Step 6: Calculate summary metrics
    if all_scores:
        summary_metrics = {
            "avg_overall": round(sum(s["overall"] for s in all_scores) / len(all_scores), 2),
            "avg_relevance": round(sum(s["relevance"] for s in all_scores) / len(all_scores), 2),
            "avg_faithfulness": round(sum(s["faithfulness"] for s in all_scores) / len(all_scores), 2),
            "avg_completeness": round(sum(s["completeness"] for s in all_scores) / len(all_scores), 2),
            "total_examples": total_examples,
            "completed": completed
        }
    else:
        summary_metrics = {
            "avg_overall": 0.0,
            "avg_relevance": 0.0,
            "avg_faithfulness": 0.0,
            "avg_completeness": 0.0,
            "total_examples": total_examples,
            "completed": completed
        }
    
    # Step 7: Update eval_run with results
    supabase.table("eval_runs").update({
        "status": "completed",
        "summary_metrics": summary_metrics
    }).eq("id", eval_run_id).execute()
    
    print(f"\n✅ Eval run complete!")
    print(f"   Average scores: Overall={summary_metrics['avg_overall']}, Relevance={summary_metrics['avg_relevance']}, Faithfulness={summary_metrics['avg_faithfulness']}, Completeness={summary_metrics['avg_completeness']}")
    
    return {
        "eval_run_id": eval_run_id,
        "status": "completed",
        "total_examples": total_examples,
        "completed": completed,
        "avg_scores": summary_metrics
    }

@app.get("/eval/runs/{eval_run_id}/results")
async def get_eval_results(eval_run_id: str):
    """
    Get detailed results for a completed eval run.
    
    Shows each question, answer, and scores.
    Like reading the food critic's full review.
    """
    
    # Get the eval_run info
    eval_run = supabase.table("eval_runs").select("*, eval_sets(name), pipeline_configs(name)").eq("id", eval_run_id).execute()
    
    if not eval_run.data:
        raise HTTPException(status_code=404, detail="Eval run not found")
    
    run_info = eval_run.data[0]
    
    # Get all results for this run
    results = supabase.table("eval_results") \
        .select("*, eval_examples(question), answers(answer_text)") \
        .eq("eval_run_id", eval_run_id) \
        .execute()
    
    # Format the response
    detailed_results = []
    for row in results.data:
        detailed_results.append({
            "question": row["eval_examples"]["question"],
            "answer": row["answers"]["answer_text"],
            "score_overall": row["score_overall"],
            "score_relevance": row["score_relevance"],
            "score_faithfulness": row["score_faithfulness"],
            "score_completeness": row["score_completeness"],
            "judge_explanation": row["judge_explanation"]
        })
    
    return {
        "eval_run_id": eval_run_id,
        "eval_set_name": run_info["eval_sets"]["name"],
        "pipeline_config_name": run_info["pipeline_configs"]["name"],
        "status": run_info["status"],
        "summary_metrics": run_info["summary_metrics"],
        "results": detailed_results
    }


@app.get("/eval/runs/compare")
async def compare_eval_runs(eval_set_id: str):
    """
    Compare all eval runs for an eval set.
    
    Shows which pipeline config performed best.
    Like comparing different recipes side-by-side.
    """
    
    # Get all runs for this eval set
    runs = supabase.table("eval_runs") \
        .select("*, pipeline_configs(name, parameters)") \
        .eq("eval_set_id", eval_set_id) \
        .eq("status", "completed") \
        .order("created_at", desc=True) \
        .execute()
    
    if not runs.data:
        return {"message": "No completed runs found for this eval set", "runs": []}
    
    # Format comparison
    comparison = []
    for run in runs.data:
        comparison.append({
            "eval_run_id": run["id"],
            "config_name": run["pipeline_configs"]["name"],
            "config_params": run["pipeline_configs"]["parameters"],
            "avg_overall": run["summary_metrics"].get("avg_overall"),
            "avg_relevance": run["summary_metrics"].get("avg_relevance"),
            "avg_faithfulness": run["summary_metrics"].get("avg_faithfulness"),
            "avg_completeness": run["summary_metrics"].get("avg_completeness"),
            "completed": run["summary_metrics"].get("completed"),
            "total": run["summary_metrics"].get("total_examples"),
            "created_at": run["created_at"]
        })
    
    # Sort by avg_overall descending (best first)
    comparison.sort(key=lambda x: x["avg_overall"] if x["avg_overall"] else 0, reverse=True)
    
    return {
        "eval_set_id": eval_set_id,
        "total_runs": len(comparison),
        "runs": comparison,
        "winner": comparison[0] if comparison else None
    }

# ============================================
# WEEK 3: AUTONOMOUS AGENTS
# ============================================

class AgentAnalysisRequest(BaseModel):
    """Request to analyze eval results"""
    eval_run_id: str

class AgentAnalysisResponse(BaseModel):
    """Claude's analysis of the eval results"""
    eval_run_id: str
    best_config_name: str
    overall_score: float
    strengths: List[str]
    weaknesses: List[str]
    documentation_gaps: List[str]
    recommendations: List[str]
    claude_reasoning: str

@app.post("/agent/analyze", response_model=AgentAnalysisResponse, tags=["agents"])
async def analyze_eval_results(request: AgentAnalysisRequest):
    """
    Phase 1: Analysis Agent
    
    Claude analyzes eval results and provides intelligent insights.
    
    Like a food inspector who reviews all critic scores and explains
    why certain dishes consistently fail.
    """
    
    print(f"\n🧠 Analysis Agent: Analyzing eval run {request.eval_run_id}")
    
    try:
        # Step 1: Fetch the eval run data
        eval_run = supabase.table("eval_runs") \
            .select("*, pipeline_configs(name, parameters), eval_sets(name)") \
            .eq("id", request.eval_run_id) \
            .execute()
        
        if not eval_run.data:
            raise HTTPException(status_code=404, detail="Eval run not found")
        
        run_info = eval_run.data[0]
        
        # Step 2: Fetch detailed results
        results = supabase.table("eval_results") \
            .select("*, eval_examples(question)") \
            .eq("eval_run_id", request.eval_run_id) \
            .execute()
        
        # Step 3: Get questions with no chunks found
    
        no_chunk_questions = []
        
        # Step 4: Build the prompt for Claude
        summary_metrics = run_info["summary_metrics"]
        
        # Format individual results
        results_text = ""
        for i, result in enumerate(results.data, 1):
            results_text += f"""
Question {i}: {result['eval_examples']['question']}
- Scores: Overall={result['score_overall']}, Relevance={result['score_relevance']}, Faithfulness={result['score_faithfulness']}, Completeness={result['score_completeness']}
- Judge Explanation: {result['judge_explanation']}
"""
        
        no_chunks_text = "\n".join([f"- {q}" for q in no_chunk_questions]) if no_chunk_questions else "None"  # type: ignore
        
        analysis_prompt = f"""You are an expert AI systems engineer analyzing the performance of a RAG (Retrieval Augmented Generation) system.

**EVAL RUN SUMMARY:**
- Pipeline Config: {run_info['pipeline_configs']['name']}
- Config Parameters: {json.dumps(run_info['pipeline_configs']['parameters'], indent=2)}
- Overall Score: {summary_metrics.get('avg_overall', 0)}/5.0
- Relevance Score: {summary_metrics.get('avg_relevance', 0)}/5.0
- Faithfulness Score: {summary_metrics.get('avg_faithfulness', 0)}/5.0
- Completeness Score: {summary_metrics.get('avg_completeness', 0)}/5.0
- Questions Answered: {summary_metrics.get('completed', 0)}/{summary_metrics.get('total_examples', 0)}

**QUESTIONS WITH NO CHUNKS FOUND:**
{no_chunks_text}

**DETAILED RESULTS:**
{results_text}

**YOUR TASK:**
Analyze this RAG system's performance and provide actionable insights.

**RESPONSE FORMAT (JSON only, no markdown):**
{{
    "strengths": ["strength 1", "strength 2", ...],
    "weaknesses": ["weakness 1", "weakness 2", ...],
    "documentation_gaps": ["gap 1 with question number", "gap 2 with question number", ...],
    "recommendations": ["recommendation 1", "recommendation 2", ...],
    "reasoning": "Your detailed analysis explaining the patterns you see and why you made these recommendations"
}}

**ANALYSIS GUIDELINES:**
1. **Strengths**: What is working well? (e.g., high faithfulness = no hallucinations)
2. **Weaknesses**: What needs improvement? (e.g., low completeness = answers too brief)
3. **Documentation Gaps**: Which questions have no chunks? What docs are missing?
4. **Recommendations**: Specific, actionable improvements (e.g., "Increase top_k from 5 to 7 to improve completeness")

Provide ONLY the JSON response, no additional text."""

        # Step 5: Call Claude API
        print("   🤖 Calling Claude API for analysis...")
        
        message = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[
                {"role": "user", "content": analysis_prompt}
            ]
        )
        
        # Step 6: Parse Claude's response
        response_text = message.content[0].text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        analysis = json.loads(response_text)
        
        print("   ✅ Analysis complete!")
        print(f"   📊 Strengths: {len(analysis['strengths'])}")
        print(f"   ⚠️  Weaknesses: {len(analysis['weaknesses'])}")
        print(f"   📚 Doc Gaps: {len(analysis['documentation_gaps'])}")
        print(f"   💡 Recommendations: {len(analysis['recommendations'])}")
        
        # Step 7: Return structured response
        return AgentAnalysisResponse(
            eval_run_id=request.eval_run_id,
            best_config_name=run_info['pipeline_configs']['name'],
            overall_score=summary_metrics.get('avg_overall', 0),
            strengths=analysis["strengths"],
            weaknesses=analysis["weaknesses"],
            documentation_gaps=analysis["documentation_gaps"],
            recommendations=analysis["recommendations"],
            claude_reasoning=analysis["reasoning"]
        )
        
    except json.JSONDecodeError as e:
        print(f"   ❌ Failed to parse Claude's response as JSON: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to parse Claude response: {str(e)}")
    except Exception as e:
        print(f"   ❌ Error in analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
## CONFIG GENERATOR AGENT
# ============================================

class AgentConfigSuggestion(BaseModel):
    """A new config suggested by Claude"""
    name: str
    description: str
    parameters: Dict
    reasoning: str
    expected_improvement: str

class AgentConfigSuggestionsRequest(BaseModel):
    """Request to generate config suggestions"""
    eval_run_id: str
    num_suggestions: int = 3  # How many new configs to generate

class AgentConfigSuggestionsResponse(BaseModel):
    """Claude's suggested configs"""
    eval_run_id: str
    current_config_name: str
    current_score: float
    suggestions: List[AgentConfigSuggestion]
    claude_reasoning: str

@app.post("/agent/suggest-configs", response_model=AgentConfigSuggestionsResponse, tags=["agents"])
async def suggest_pipeline_configs(request: AgentConfigSuggestionsRequest):
    """
    Phase 2: Config Generator Agent
    
    Claude analyzes eval results and automatically proposes new pipeline 
    configurations that should perform better.
    
    Analogy: Like an experimental chef proposing new recipes based on 
    customer feedback about what's missing in the current dish.
    
    How it works:
    1. Fetch eval run data and see what's failing
    2. Ask Claude to propose better configs
    3. Claude returns specific parameter combinations with reasoning
    4. Automatically create those configs in the database
    5. Ready to test in Phase 3!
    """
    
    print(f"\n💡 Config Generator: Creating {request.num_suggestions} new configs based on eval run {request.eval_run_id}")
    
    try:
        # ========================================
        # STEP 1: Fetch the eval run data
        # ========================================
        # We need to know:
        # - What config was tested
        # - What scores it got
        # - Which questions failed
        print("   📊 Fetching eval run data...")
        
        eval_run = supabase.table("eval_runs") \
            .select("*, pipeline_configs(name, parameters), eval_sets(name, workspace_id)") \
            .eq("id", request.eval_run_id) \
            .execute()
        
        if not eval_run.data:
            raise HTTPException(status_code=404, detail="Eval run not found")
        
        run_info = eval_run.data[0]
        
        # ========================================
        # STEP 2: Get detailed question results
        # ========================================
        # We want to know which specific questions scored poorly
        # so Claude can understand what needs fixing
        results = supabase.table("eval_results") \
            .select("*, eval_examples(question)") \
            .eq("eval_run_id", request.eval_run_id) \
            .execute()
        
        summary_metrics = run_info["summary_metrics"]
        current_config = run_info["pipeline_configs"]
        
        # ========================================
        # STEP 3: Format results for Claude
        # ========================================
        # Create a summary of all questions and identify low-scoring ones
        results_summary = ""
        low_scoring = []  # Questions that scored < 3.0
        
        for i, result in enumerate(results.data, 1):
            score = result['score_overall']
            results_summary += f"Q{i} (Score: {score}): {result['eval_examples']['question']}\n"
            
            # Track questions that failed badly
            if score < 3.0:
                low_scoring.append({
                    "question": result['eval_examples']['question'],
                    "score": score,
                    "completeness": result['score_completeness']
                })
        
        # ========================================
        # STEP 4: Build the prompt for Claude
        # ========================================
        # This prompt gives Claude:
        # 1. Current config and its performance
        # 2. Which questions failed
        # 3. Available parameters to tune
        # 4. Instructions to propose new configs
        
        config_prompt = f"""You are an expert AI systems engineer optimizing a RAG system's pipeline configuration.

**CURRENT CONFIGURATION:**
Name: {current_config['name']}
Parameters: {json.dumps(current_config['parameters'], indent=2)}

**CURRENT PERFORMANCE:**
- Overall Score: {summary_metrics.get('avg_overall', 0)}/5.0
- Relevance: {summary_metrics.get('avg_relevance', 0)}/5.0
- Faithfulness: {summary_metrics.get('avg_faithfulness', 0)}/5.0
- Completeness: {summary_metrics.get('avg_completeness', 0)}/5.0
- Questions Answered: {summary_metrics.get('completed', 0)}/{summary_metrics.get('total_examples', 0)}

**LOW-SCORING QUESTIONS (Score < 3.0):**
{json.dumps(low_scoring, indent=2)}

**ALL QUESTION SCORES:**
{results_summary}

**YOUR TASK:**
Based on this performance data, propose {request.num_suggestions} NEW pipeline configurations that could improve the overall score.

**AVAILABLE PARAMETERS TO TUNE:**

1. chunk_size (current: {current_config['parameters'].get('chunk_size', 500)})
   - Smaller (300-400): More precise retrieval, but may fragment context
   - Larger (600-800): More context per chunk, but less precise matching
   
2. chunk_overlap (current: {current_config['parameters'].get('chunk_overlap', 50)})
   - More overlap (75-100): Better context preservation at boundaries
   
3. top_k (current: {current_config['parameters'].get('top_k', 3)})
   - More chunks (5-7): More context for LLM, but may add noise
   - Fewer chunks (2-3): More focused, but may miss important context
   
4. similarity_threshold (current: {current_config['parameters'].get('similarity_threshold', 0.5)})
   - Lower (0.3-0.4): More lenient matching, retrieves more chunks
   - Higher (0.6-0.7): Stricter matching, only very relevant chunks

**RESPONSE FORMAT (JSON only, no markdown):**
{{
    "suggestions": [
        {{
            "name": "Descriptive config name",
            "description": "Brief description of this configuration",
            "parameters": {{
                "chunk_size": <value>,
                "chunk_overlap": <value>,
                "top_k": <value>,
                "similarity_threshold": <value>
            }},
            "reasoning": "Why this config should improve performance based on the data",
            "expected_improvement": "What specific metrics should improve and why"
        }}
    ],
    "overall_reasoning": "Your analysis of the current config's weaknesses and how these suggestions address them"
}}

**IMPORTANT:**
1. Make configs DIFFERENT from the current one - don't just tweak one parameter
2. Base suggestions on ACTUAL patterns in the low-scoring questions
3. Each config should target a specific hypothesis about what's wrong
4. Be specific about expected improvements (e.g., "Should improve completeness from 3.75 to ~4.2")

Provide ONLY the JSON response, no additional text."""

        # ========================================
        # STEP 5: Call Claude API
        # ========================================
        # Send the prompt to Claude and get back config suggestions
        print("   🤖 Calling Claude API to generate configs...")
        
        message = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=3000,  # Need more tokens for multiple configs
            messages=[
                {"role": "user", "content": config_prompt}
            ]
        )
        
        # ========================================
        # STEP 6: Parse Claude's response
        # ========================================
        # Claude should return JSON with config suggestions
        response_text = message.content[0].text.strip()
        
        # Remove markdown code blocks if Claude wrapped the JSON
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        # Parse the JSON
        suggestions_data = json.loads(response_text)
        
        print("   ✅ Config generation complete!")
        print(f"   💡 Generated {len(suggestions_data['suggestions'])} new configs")
        
        # ========================================
        # STEP 7: Save new configs to database
        # ========================================
        # Automatically create the suggested configs in pipeline_configs table
        # so they're ready to test in Phase 3
        
        workspace_id = run_info["eval_sets"].get("workspace_id")
        
        created_configs = []
        for suggestion in suggestions_data["suggestions"]:
            # Insert into pipeline_configs table
            new_config = supabase.table("pipeline_configs").insert({
                "workspace_id": workspace_id,
                "name": suggestion["name"],
                "description": suggestion["description"],
                "parameters": suggestion["parameters"],
                "is_active": False  # Not active yet, needs testing first
            }).execute()
            
            # Build response object
            created_configs.append(AgentConfigSuggestion(
                name=suggestion["name"],
                description=suggestion["description"],
                parameters=suggestion["parameters"],
                reasoning=suggestion["reasoning"],
                expected_improvement=suggestion["expected_improvement"]
            ))
            
            print(f"   ✅ Created config in DB: {suggestion['name']}")
        
        # ========================================
        # STEP 8: Return response
        # ========================================
        # Send back all the suggestions with Claude's reasoning
        return AgentConfigSuggestionsResponse(
            eval_run_id=request.eval_run_id,
            current_config_name=current_config['name'],
            current_score=summary_metrics.get('avg_overall', 0),
            suggestions=created_configs,
            claude_reasoning=suggestions_data["overall_reasoning"]
        )
        
    except json.JSONDecodeError as e:
        print(f"   ❌ Failed to parse Claude's response as JSON: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to parse Claude response: {str(e)}")
    except Exception as e:
        print(f"   ❌ Error generating configs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# PHASE 3: AUTO-TUNER ORCHESTRATOR
# ============================================

class AutoTuneRequest(BaseModel):
    """Request to start autonomous optimization"""
    eval_set_id: str
    workspace_id: str
    max_iterations: int = 5
    min_improvement: float = 0.05  # Stop if improvement < 0.05

class AutoTuneIteration(BaseModel):
    """One iteration of the auto-tuning loop"""
    iteration: int
    config_name: str
    config_id: str
    score: float
    improvement_from_previous: float

class AutoTuneResponse(BaseModel):
    """Results from the auto-tuning session"""
    optimization_complete: bool
    total_iterations: int
    starting_score: float
    final_score: float
    improvement: float
    winning_config_name: str
    winning_config_id: str
    iteration_history: List[AutoTuneIteration]
    final_insights: Dict
    reason_stopped: str

@app.post("/agent/auto-tune", response_model=AutoTuneResponse, tags=["agents"])
async def auto_tune_pipeline(request: AutoTuneRequest):
    """
    Phase 3: Auto-Tuner Orchestrator
    
    Runs a fully autonomous optimization loop:
    1. Find current best config
    2. Run eval on it
    3. Analyze results with Claude (Phase 1)
    4. Generate new configs with Claude (Phase 2)
    5. Test all new configs
    6. Pick the winner
    7. Repeat until no improvement
    
    Analogy: Master chef who keeps tweaking recipes, testing them,
    and only keeping improvements - all automatically while you sleep!
    
    This is the culmination of Weeks 1, 2, and 3.
    """
    
    print(f"\n🤖 AUTO-TUNER: Starting autonomous optimization")
    print(f"   📊 Eval Set: {request.eval_set_id}")
    print(f"   🎯 Max Iterations: {request.max_iterations}")
    print(f"   📈 Min Improvement: {request.min_improvement}")
    
    try:
        # ========================================
        # SETUP: Find the current best config
        # ========================================
        print("\n🔍 Finding current best configuration...")
        
        # Get all completed eval runs for this eval set
        existing_runs = supabase.table("eval_runs") \
            .select("*, pipeline_configs(name, id)") \
            .eq("eval_set_id", request.eval_set_id) \
            .eq("status", "completed") \
            .execute()
        
        if not existing_runs.data:
            raise HTTPException(
                status_code=404, 
                detail="No completed eval runs found. Run at least one eval first."
            )
        
        # Find the run with highest score
        best_run = max(existing_runs.data, key=lambda x: x["summary_metrics"].get("avg_overall", 0))
        starting_score = best_run["summary_metrics"]["avg_overall"]
        current_best_config_id = best_run["pipeline_config_id"]
        current_best_score = starting_score
        
        print(f"   ✅ Starting with: {best_run['pipeline_configs']['name']} (Score: {starting_score})")
        
        # Track iteration history
        iteration_history = []
        iterations_without_improvement = 0
        
        # ========================================
        # MAIN OPTIMIZATION LOOP
        # ========================================
        for iteration in range(1, request.max_iterations + 1):
            print(f"\n{'='*60}")
            print(f"🔄 ITERATION {iteration}/{request.max_iterations}")
            print(f"{'='*60}")
            print(f"   Current Best: {current_best_score:.2f}")
            
            # ----------------------------------------
            # STEP 1: Run eval on current best config
            # ----------------------------------------
            print(f"\n📝 Step 1: Running eval on current best config...")
            
            eval_response = await run_evaluation(EvalRunCreate(
                eval_set_id=request.eval_set_id,
                pipeline_config_id=current_best_config_id
            ))
            
            current_eval_run_id = eval_response["eval_run_id"]
            current_score = eval_response["avg_scores"]["avg_overall"]
            
            print(f"   ✅ Eval complete: Score = {current_score:.2f}")
            
            # ----------------------------------------
            # STEP 2: Analyze with Claude (Phase 1)
            # ----------------------------------------
            print(f"\n🧠 Step 2: Analyzing results with Claude...")
            
            analysis = await analyze_eval_results(AgentAnalysisRequest(
                eval_run_id=current_eval_run_id
            ))
            
            print(f"   ✅ Analysis complete")
            print(f"   📊 Strengths: {len(analysis.strengths)}")
            print(f"   ⚠️  Weaknesses: {len(analysis.weaknesses)}")
            
            # ----------------------------------------
            # STEP 3: Generate new configs (Phase 2)
            # ----------------------------------------
            print(f"\n💡 Step 3: Generating new configs with Claude...")
            
            suggestions = await suggest_pipeline_configs(AgentConfigSuggestionsRequest(
                eval_run_id=current_eval_run_id,
                num_suggestions=3
            ))
            
            print(f"   ✅ Generated {len(suggestions.suggestions)} new configs")
            
            # ----------------------------------------
            # STEP 4: Test all new configs
            # ----------------------------------------
            print(f"\n🧪 Step 4: Testing all {len(suggestions.suggestions)} new configs...")
            
            best_new_score = current_score
            best_new_config_id = current_best_config_id
            best_new_config_name = suggestions.current_config_name
            
            # Get the newly created config IDs from database
            # (They were just created by suggest_pipeline_configs)
            new_configs = supabase.table("pipeline_configs") \
                .select("id, name") \
                .eq("workspace_id", request.workspace_id) \
                .order("created_at", desc=True) \
                .limit(3) \
                .execute()
            
            for i, new_config in enumerate(new_configs.data, 1):
                print(f"\n   Testing config {i}/3: {new_config['name']}...")
                
                # Run eval on this new config
                new_eval = await run_evaluation(EvalRunCreate(
                    eval_set_id=request.eval_set_id,
                    pipeline_config_id=new_config["id"]
                ))
                
                new_score = new_eval["avg_scores"]["avg_overall"]
                print(f"   📊 Score: {new_score:.2f}")
                
                # Track if this is the best new config
                if new_score > best_new_score:
                    best_new_score = new_score
                    best_new_config_id = new_config["id"]
                    best_new_config_name = new_config["name"]
                    print(f"   🌟 New best!")
            
            # ----------------------------------------
            # STEP 5: Compare and decide
            # ----------------------------------------
            print(f"\n🎯 Step 5: Comparing results...")
            
            improvement = best_new_score - current_best_score
            print(f"   Current: {current_best_score:.2f}")
            print(f"   Best New: {best_new_score:.2f}")
            print(f"   Improvement: {improvement:+.2f}")
            
            # Record this iteration
            iteration_history.append(AutoTuneIteration(
                iteration=iteration,
                config_name=best_new_config_name,
                config_id=best_new_config_id,
                score=best_new_score,
                improvement_from_previous=improvement
            ))
            
            # ----------------------------------------
            # STEP 6: Check stopping conditions
            # ----------------------------------------
            
            if improvement >= request.min_improvement:
                # We found improvement! Update current best and continue
                print(f"   ✅ Improvement found! Continuing to iteration {iteration + 1}...")
                current_best_config_id = best_new_config_id
                current_best_score = best_new_score
                iterations_without_improvement = 0
            else:
                # No significant improvement
                iterations_without_improvement += 1
                print(f"   ⚠️  No significant improvement ({iterations_without_improvement} iteration(s))")
                
                if iterations_without_improvement >= 2:
                    print(f"\n🛑 Stopping: No improvement for 2 consecutive iterations")
                    reason_stopped = "No improvement for 2 consecutive iterations"
                    break
            
            # Check if we've reached max iterations
            if iteration == request.max_iterations:
                print(f"\n🛑 Stopping: Reached max iterations ({request.max_iterations})")
                reason_stopped = f"Reached max iterations ({request.max_iterations})"
        else:
            # Loop completed without breaking
            reason_stopped = f"Completed all {request.max_iterations} iterations"
        
        # ========================================
        # FINAL RESULTS
        # ========================================
        print(f"\n{'='*60}")
        print(f"🏁 OPTIMIZATION COMPLETE")
        print(f"{'='*60}")
        
        final_score = iteration_history[-1].score if iteration_history else starting_score
        total_improvement = final_score - starting_score
        
        print(f"   Starting Score: {starting_score:.2f}")
        print(f"   Final Score: {final_score:.2f}")
        print(f"   Total Improvement: {total_improvement:+.2f} ({(total_improvement/starting_score*100):+.1f}%)")
        print(f"   Total Iterations: {len(iteration_history)}")
        print(f"   Reason Stopped: {reason_stopped}")
        
        # Get final analysis and insights
        final_eval_run_id = iteration_history[-1].config_id if iteration_history else current_eval_run_id
        
        # Get final run details for insights
        final_run = supabase.table("eval_runs") \
            .select("*") \
            .eq("pipeline_config_id", iteration_history[-1].config_id if iteration_history else current_best_config_id) \
            .order("created_at", desc=True) \
            .limit(1) \
            .execute()
        
        final_insights = {
            "final_score": final_score,
            "starting_score": starting_score,
            "improvement": total_improvement,
            "iterations_run": len(iteration_history),
            "recommendation": "System has been optimized. Monitor performance on new data."
        }
        
        return AutoTuneResponse(
            optimization_complete=True,
            total_iterations=len(iteration_history),
            starting_score=starting_score,
            final_score=final_score,
            improvement=total_improvement,
            winning_config_name=iteration_history[-1].config_name if iteration_history else best_run['pipeline_configs']['name'],
            winning_config_id=iteration_history[-1].config_id if iteration_history else current_best_config_id,
            iteration_history=iteration_history,
            final_insights=final_insights,
            reason_stopped=reason_stopped
        )
        
    except Exception as e:
        print(f"\n❌ Error in auto-tuning: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# debug endpoint here
@app.get("/test-gemini", tags=["debug"])
async def test_gemini():
    """Debug endpoint to see available Gemini models"""
    try:
        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)
        
        models = genai.list_models()
        available = []
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                available.append({
                    "name": model.name,
                    "display_name": model.display_name
                })
        
        return {"available_models": available}
    except Exception as e:
        return {"error": str(e)}