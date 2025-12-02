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

    Shows which pipeline config performed best,
    and includes where each config came from.
    """

    # Get all runs for this eval set
    runs = (
        supabase.table("eval_runs")
        .select(
            "*, pipeline_configs(name, parameters, origin, created_by_agent_run_id, parent_config_id)"
        )
        .eq("eval_set_id", eval_set_id)
        .eq("status", "completed")
        .order("created_at", desc=True)
        .execute()
    )

    if not runs.data:
        return {"message": "No completed runs found for this eval set", "runs": []}

    # Format comparison
    comparison = []
    for run in runs.data:
        pc = run["pipeline_configs"]
        summary = run["summary_metrics"] or {}

        comparison.append(
            {
                "eval_run_id": run["id"],
                "pipeline_config_id": run["pipeline_config_id"],
                "config_name": pc["name"],
                "config_params": pc["parameters"],
                "origin": pc.get("origin", "manual"),
                "created_by_agent_run_id": pc.get("created_by_agent_run_id"),
                "parent_config_id": pc.get("parent_config_id"),
                "avg_overall": summary.get("avg_overall"),
                "avg_relevance": summary.get("avg_relevance"),
                "avg_faithfulness": summary.get("avg_faithfulness"),
                "avg_completeness": summary.get("avg_completeness"),
                "completed": summary.get("completed"),
                "total": summary.get("total_examples"),
                "created_at": run["created_at"],
            }
        )

    # Sort by avg_overall descending (best first)
    comparison.sort(
        key=lambda x: x["avg_overall"] if x["avg_overall"] is not None else 0,
        reverse=True,
    )

    return {
        "eval_set_id": eval_set_id,
        "total_runs": len(comparison),
        "runs": comparison,
        "winner": comparison[0] if comparison else None,
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

class AutoTuneRequest(BaseModel):
    """Request to run autonomous tuning loop"""
    workspace_id: str
    eval_set_id: str
    max_iterations: int = 3

class AutoTuneHistoryItem(BaseModel):
    """One iteration of the auto-tune loop"""
    iteration: int
    eval_run_id: str
    pipeline_config_id: str
    pipeline_config_name: str
    avg_overall: float

class AutoTuneResponse(BaseModel):
    """Result of auto-tune run"""
    status: str
    total_iterations: int
    final_score: float
    final_config_id: str
    final_config_name: str
    starting_score: Optional[float] = None
    improvement: Optional[float] = None
    history: List[AutoTuneHistoryItem]

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
        eval_run = (
            supabase.table("eval_runs")
            .select(
                "*, pipeline_configs(id, name, parameters), eval_sets(name, workspace_id)"
            )
            .eq("id", request.eval_run_id)
            .execute()
        )
        
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

    How it works:
    1. Fetch eval run data and see what is failing
    2. Ask Claude to propose better configs
    3. Claude returns specific parameter combinations with reasoning
    4. Automatically create those configs in the database
    5. Ready to test in Phase 3
    """
    print(
        f"\n💡 Config Generator: Creating {request.num_suggestions} new configs based on eval run {request.eval_run_id}"
    )

    try:
        # Step 1: Fetch eval run with its config and eval set
        eval_run = (
            supabase.table("eval_runs")
            .select(
                "*, pipeline_configs(id, name, parameters), eval_sets(name, workspace_id)"
            )
            .eq("id", request.eval_run_id)
            .execute()
        )

        if not eval_run.data:
            raise HTTPException(status_code=404, detail="Eval run not found")

        run_info = eval_run.data[0]
        current_config = run_info["pipeline_configs"]
        eval_set = run_info["eval_sets"]
        workspace_id = eval_set["workspace_id"]
        summary_metrics = run_info.get("summary_metrics") or {}

        # Step 2: Fetch detailed question level results for prompt context
        results = (
            supabase.table("eval_results")
            .select("*, eval_examples(question)")
            .eq("eval_run_id", request.eval_run_id)
            .execute()
        )

        results_text = ""
        for i, row in enumerate(results.data, 1):
            results_text += f"""
Question {i}: {row['eval_examples']['question']}
- Scores: Overall={row['score_overall']}, Relevance={row['score_relevance']}, Faithfulness={row['score_faithfulness']}, Completeness={row['score_completeness']}
- Judge Explanation: {row['judge_explanation']}
"""

        # Step 3: Fetch existing configs for this workspace
        existing_configs_result = (
            supabase.table("pipeline_configs")
            .select(
                "id, name, parameters, origin, parent_config_id, created_by_agent_run_id, created_at"
            )
            .eq("workspace_id", workspace_id)
            .execute()
        )
        existing_configs = existing_configs_result.data or []

        # Step 4: Build Claude prompt to propose new configs
        prompt = f"""You are an expert RAG systems engineer designing retrieval configs.

        You are given the current pipeline configuration, its evaluation results, and any existing configs.
        Your job is to propose better retrieval configurations that should improve scores on this eval set.

        CURRENT CONFIG:
        Name: {current_config['name']}
        Parameters (JSON):
        {json.dumps(current_config['parameters'], indent=2)}

        SUMMARY METRICS (0 to 5 scale):
        - Overall: {summary_metrics.get('avg_overall', 0)}
        - Relevance: {summary_metrics.get('avg_relevance', 0)}
        - Faithfulness: {summary_metrics.get('avg_faithfulness', 0)}
        - Completeness: {summary_metrics.get('avg_completeness', 0)}
        - Completed: {summary_metrics.get('completed', 0)}/{summary_metrics.get('total_examples', 0)}

        EXISTING CONFIGS IN WORKSPACE:
        {json.dumps(existing_configs, indent=2)}

        DETAILED QUESTION RESULTS:
        {results_text}

        YOUR TASK:
        Propose {request.num_suggestions} new pipeline config variants that may outperform the current one on this eval set.

        For each suggested config, you MUST provide:
        - name: Short, descriptive name.
        - description: One or two sentence description of what it tries to change.
        - parameters: A JSON object with explicit numeric values for:
            - top_k (int)
            - chunk_size (int)
            - chunk_overlap (int)
            - similarity_threshold (float between 0 and 1)
        - reasoning: Why this configuration should help, tied back to the eval results.
        - expected_improvement: A short statement describing where you expect scores to improve.

        RESPONSE FORMAT (JSON only, no markdown, no extra text):
        {{
        "suggestions": [
            {{
            "name": "Config name",
            "description": "What this config is trying to do",
            "parameters": {{
                "top_k": 6,
                "chunk_size": 600,
                "chunk_overlap": 80,
                "similarity_threshold": 0.52
            }},
            "reasoning": "Why this particular combination should help.",
            "expected_improvement": "Where you expect the scores to improve."
            }}
        ],
        "reasoning": "Overall comparison of the proposed configs and how they relate to the baseline."
        }}"""

        # Step 5: Call Claude to generate suggestions
        print("   🤖 Calling Claude API for config suggestions...")
        message = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = message.content[0].text.strip()

        # Remove markdown code fences if present
        if response_text.startswith("```"):
            response_text = response_text.split("```", 1)[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        payload = json.loads(response_text)
        raw_suggestions = payload.get("suggestions", [])
        overall_reasoning = payload.get("reasoning", "")

        if not isinstance(raw_suggestions, list) or not raw_suggestions:
            raise HTTPException(
                status_code=500,
                detail="Claude did not return any config suggestions",
            )

        saved_suggestions: List[AgentConfigSuggestion] = []

        # Step 6: Insert each suggested config into the database
        for suggestion in raw_suggestions:
            params = suggestion.get("parameters") or {}
            name = suggestion.get("name", "Agent suggested config")
            description = suggestion.get("description", "")
            reasoning = suggestion.get("reasoning", "")
            expected_improvement = suggestion.get("expected_improvement", "")

            insert_result = (
                supabase.table("pipeline_configs")
                .insert(
                    {
                        "workspace_id": workspace_id,
                        "name": name,
                        "description": description,
                        "parameters": params,
                        "is_active": False,
                        "origin": "agent_suggested",
                        "parent_config_id": current_config["id"],
                        "created_by_agent_run_id": request.eval_run_id,
                    }
                )
                .execute()
            )

            new_config = insert_result.data[0]

            saved_suggestions.append(
                AgentConfigSuggestion(
                    name=name,
                    description=description,
                    parameters=params,
                    reasoning=reasoning,
                    expected_improvement=expected_improvement,
                )
            )

            print(
                f"   ✅ Created suggested config '{name}' (id={new_config['id']}) from eval run {request.eval_run_id}"
            )

        print(f"   ✅ Finished creating {len(saved_suggestions)} new configs")

        return AgentConfigSuggestionsResponse(
            eval_run_id=request.eval_run_id,
            current_config_name=current_config["name"],
            current_score=summary_metrics.get("avg_overall", 0.0),
            suggestions=saved_suggestions,
            claude_reasoning=overall_reasoning,
        )

    except json.JSONDecodeError as e:
        print(f"   ❌ Failed to parse Claude's response as JSON: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to parse Claude response: {str(e)}"
        )
    except Exception as e:
        print(f"   ❌ Error in config suggestion agent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================@app.post("/agent/auto-tune", response_model=AutoTuneResponse, tags=["agents"])
async def auto_tune_agent(request: AutoTuneRequest):
    """
    Phase 3: Auto-Tuner Orchestrator
    
    Autonomous loop that:
    1. Runs eval with current best config
    2. Analyzes results
    3. Generates new configs
    4. Tests new configs
    5. Keeps the winner
    """
    print(f"\n🔄 Auto-Tuner: Starting autonomous optimization loop")
    print(f"   Workspace: {request.workspace_id}")
    print(f"   Eval Set: {request.eval_set_id}")
    print(f"   Max Iterations: {request.max_iterations}")
    
    history = []
    
    try:
        # Step 1: Find current best config from existing eval runs
        comparison = await compare_eval_runs(request.eval_set_id)
        runs = comparison.get("runs", [])
        
        if runs:
            best = runs[0]
            current_config_id = best["pipeline_config_id"]
            current_config_name = best["config_name"]
            current_best_score = float(best["avg_overall"] or 0)
            starting_score = current_best_score
            print(f"   📊 Starting from best config: '{current_config_name}' (score: {current_best_score})")
        else:
            baseline = supabase.table("pipeline_configs") \
                .select("*") \
                .eq("workspace_id", request.workspace_id) \
                .eq("is_active", True) \
                .order("created_at", desc=True) \
                .limit(1) \
                .execute()
            
            if not baseline.data:
                raise HTTPException(status_code=400, detail="No pipeline configs found for this workspace")
            
            cfg = baseline.data[0]
            current_config_id = cfg["id"]
            current_config_name = cfg["name"]
            current_best_score = 0.0
            starting_score = 0.0
            print(f"   📊 No existing runs. Starting from config: '{current_config_name}'")
        
        # Step 2: Run the optimization loop
        for iteration in range(1, request.max_iterations + 1):
            print(f"\n   🔁 Iteration {iteration}/{request.max_iterations}")
            
            # 2a. Run eval with current config
            print(f"      Running eval with '{current_config_name}'...")
            eval_result = await run_evaluation(EvalRunCreate(
                eval_set_id=request.eval_set_id,
                pipeline_config_id=current_config_id
            ))
            
            eval_run_id = eval_result["eval_run_id"]
            avg_overall = float(eval_result["avg_scores"]["avg_overall"])
            
            print(f"      ✓ Eval complete. Score: {avg_overall}")
            
            history.append(AutoTuneHistoryItem(
                iteration=iteration,
                eval_run_id=eval_run_id,
                pipeline_config_id=current_config_id,
                pipeline_config_name=current_config_name,
                avg_overall=avg_overall
            ))
            
            if avg_overall > current_best_score:
                current_best_score = avg_overall
                print(f"      🎉 New best score: {current_best_score}")
            
            # 2b. Generate new config suggestions
            print(f"      Generating new config suggestions...")
            suggestions_response = await suggest_pipeline_configs(
                AgentConfigSuggestionsRequest(
                    eval_run_id=eval_run_id,
                    num_suggestions=1
                )
            )
            
            if not suggestions_response.suggestions:
                print(f"      ⚠️ No new configs suggested. Stopping.")
                break
            
            # 2c. Get the newly created config
            newest_config = supabase.table("pipeline_configs") \
                .select("*") \
                .eq("workspace_id", request.workspace_id) \
                .eq("origin", "agent_suggested") \
                .order("created_at", desc=True) \
                .limit(1) \
                .execute()
            
            if not newest_config.data:
                print(f"      ⚠️ Could not find newly created config. Stopping.")
                break
            
            candidate = newest_config.data[0]
            current_config_id = candidate["id"]
            current_config_name = candidate["name"]
            print(f"      ✓ Next iteration will test: '{current_config_name}'")
        
        # Step 3: Calculate improvement
        improvement = current_best_score - starting_score if starting_score else None
        
        print(f"\n✅ Auto-tune complete!")
        print(f"   Final score: {current_best_score}")
        print(f"   Improvement: {improvement if improvement else 'N/A'}")
        
        return AutoTuneResponse(
            status="completed",
            total_iterations=len(history),
            final_score=current_best_score,
            final_config_id=current_config_id,
            final_config_name=current_config_name,
            starting_score=starting_score,
            improvement=improvement,
            history=history
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in auto-tune: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))