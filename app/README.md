# DevInfra RAG System

A production-ready RAG (Retrieval Augmented Generation) system for ingesting technical documentation and providing AI-powered question answering.

## Features

- **Multi-workspace management** - Organize documents by project/team
- **Intelligent document ingestion** - Automatic chunking and embedding generation
- **Vector similarity search** - Fast semantic search using pgvector
- **LLM-powered answers** - Context-aware responses using Gemini
- **Query logging** - Track all questions and answers for analysis
- **Interactive API docs** - Auto-generated Swagger UI at `/docs`

## Tech Stack

- **Backend:** FastAPI (Python)
- **Database:** Supabase (PostgreSQL + pgvector)
- **Embeddings:** OpenAI (text-embedding-3-small)
- **LLM:** Google Gemini 2.5 Flash
- **Deployment:** Local development (production deployment coming soon)

## Setup

### Prerequisites

- Python 3.9+
- Supabase account
- OpenAI API key
- Google Gemini API key

### Installation

1. Clone the repository
```bash
git clone <your-repo-url>
cd devinfra-api
```

2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up environment variables

Create a `.env` file:
```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
```

5. Set up database

Run the SQL scripts in Supabase SQL Editor to create tables and the vector search function.

### Running the Server
```bash
uvicorn app.main:app --reload
```

Visit `http://localhost:8000/docs` for interactive API documentation.

## API Endpoints

### Workspaces
- `POST /workspaces` - Create a workspace
- `GET /workspaces` - List all workspaces
- `GET /workspaces/{id}` - Get specific workspace
- `DELETE /workspaces/{id}` - Delete workspace

### Documents
- `POST /documents` - Upload and ingest a document
- `GET /documents` - List documents (filterable by workspace)
- `GET /documents/{id}` - Get specific document

### Query (RAG)
- `POST /query` - Ask a question and get AI-powered answer

## How It Works

1. **Document Ingestion:**
   - Upload document text
   - System chunks text into ~500 character pieces with overlap
   - Each chunk converted to vector embedding (1536 dimensions)
   - Embeddings stored in pgvector for fast similarity search

2. **RAG Query:**
   - User asks a question
   - Question converted to vector embedding
   - pgvector finds 3 most similar chunks
   - Chunks + question sent to Gemini LLM
   - AI generates answer based only on provided context

## Project Structure
```
devinfra-api/
├── app/
│   ├── main.py           # FastAPI app and all endpoints
│   └── __init__.py
├── seed_data/            # Sample documents for testing
├── venv/                 # Virtual environment
├── .env                  # Environment variables (not in git)
├── .env.example          # Template for .env
├── .gitignore
├── requirements.txt
└── README.md
```

## Current Status

**Week 1 Complete ✅**
- [x] Workspace management
- [x] Document ingestion with chunking
- [x] Vector embeddings and storage
- [x] RAG query endpoint
- [x] Query/answer logging

**Coming Next (Week 2)**
- [ ] Evaluation harness
- [ ] Pipeline configurations
- [ ] Incident simulator
- [ ] Auto-tuning agents
- [ ] React frontend

## Use Cases

- **Engineering teams:** Search across API docs, runbooks, and READMEs
- **Customer support:** Quick answers from product documentation
- **Onboarding:** Help new team members find information faster
- **Incident response:** Rapidly surface relevant troubleshooting docs

## Resume Talking Points

- Built FastAPI backend with RESTful endpoints
- Implemented vector similarity search using pgvector
- Integrated OpenAI embeddings and Google Gemini LLM
- Designed multi-document RAG system with source attribution
- Automatic chunking strategy with semantic boundaries

## License

MIT
```

---

## Step 2: Make Sure .gitignore is Correct

Check your `.gitignore` file has:
```
# Python
venv/
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
dist/
*.egg-info/

# Environment
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo
.cursor/

# OS
.DS_Store
Thumbs.db

# Database
*.db
*.sqlite