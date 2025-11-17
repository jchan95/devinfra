from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from datetime import datetime
from typing import Optional, List
from supabase import create_client, Client
from typing import Optional, List
import time
import re
from openai import OpenAI

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
    
    class Config:
        env_file = ".env"

settings = Settings()

# Database connection
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

# Initialize OpenAI client
openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

# FastAPI app
app = FastAPI(title="DevInfra RAG API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
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
    
    class Config:
        # This tells Pydantic to ignore extra fields from the database
        extra = "ignore"


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
        
        # Step 4: Call Gemini LLM
        print(f"🤖 Calling Gemini to generate answer...")
        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)
        
        # Use the correct model name format
        model = genai.GenerativeModel('models/gemini-2.5-flash')
        response = model.generate_content(prompt)
        answer = response.text
        
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
            "model_name": "gemini-2.5-flash",
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