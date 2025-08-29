import warnings
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from rag import RAGPipeline   # import the class, not functions

warnings.filterwarnings("ignore")

app = FastAPI(title="Simple RAG API", version="1.0")

# Initialize pipeline once (singleton style)
rag_pipeline = RAGPipeline()


# --- Request & Response Models ---
class RAGRequest(BaseModel):
    query: str
    documents: List[str] = []


class RAGResponse(BaseModel):
    answer: str
    sources: List[str]


# --- API Endpoint ---
@app.post("/rag", response_model=RAGResponse)
async def rag_endpoint(request: RAGRequest):
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="Query cannot be empty")

    try:
        # Load new documents (if provided)
        if request.documents:
            rag_pipeline.load_documents(request.documents)

        # Ask RAG
        result = rag_pipeline.ask(request.query)

        return RAGResponse(answer=result["answer"], sources=result["sources"])

    except Exception as e:
        # Log the actual error if you want
        print(f"RAG pipeline error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
