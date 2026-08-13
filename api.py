from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import subprocess
import chromadb
import re
from langchain_ollama import OllamaEmbeddings, ChatOllama
from rank_bm25 import BM25Okapi
from link_extractor import get_links_for_file

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "source"
LLM_MODEL = "llama3.1:8b"
TOP_K = 6

PROMPT_TEMPLATE = """You are a research assistant answering questions using only the provided context from documents.

Context:
{context}

Question: {question}

Instructions:
- Answer using only the information in the context above.
- If the context contains multiple different values that could answer the question, mention all of them and clearly specify which table or section each comes from.
- If the context does not contain enough information to answer, say so clearly.
- After your answer, cite the sources used in this format: [source: filename, chunk X]

Answer:"""

app = FastAPI(title="Local RAG Research Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_state = {}

#Global state
_state = {}


def get_state():
    if not _state:
        _state["embeddings"] = OllamaEmbeddings(model="nomic-embed-text")
        _state["llm"] = ChatOllama(model=LLM_MODEL, temperature=0.2)
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _state["collection"] = client.get_or_create_collection(name=COLLECTION_NAME)
        rebuild_bm25()
    return _state


def rebuild_bm25():
    collection = _state["collection"]
    all_items = collection.get(limit=10000, include=["documents", "metadatas"])
    _state["bm25_docs"] = all_items["documents"]
    _state["bm25_metas"] = all_items["metadatas"]
    _state["bm25_ids"] = all_items["ids"]
    tokenized = [re.findall(r'\w+', doc.lower()) for doc in all_items["documents"]]
    _state["bm25"] = BM25Okapi(tokenized) if tokenized else None


#Retrieval logic

def retrieve_chunks_hybrid(query, allowed_sources=None, k=TOP_K):
    s = get_state()
    query_vector = s["embeddings"].embed_query(query)
    dense_results = s["collection"].query(
        query_embeddings=[query_vector],
        n_results=k * 3,
        include=["documents", "metadatas", "distances"]
    )
    dense_ids = dense_results["ids"][0]

    tokenized_query = re.findall(r'\w+', query.lower())
    bm25_scores = s["bm25"].get_scores(tokenized_query)
    bm25_ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k * 3]
    bm25_top_ids = [s["bm25_ids"][i] for i in bm25_ranked]

    rrf_scores = {}
    rrf_k = 60
    for rank, doc_id in enumerate(dense_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (rrf_k + rank + 1)
    for rank, doc_id in enumerate(bm25_top_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (rrf_k + rank + 1)

    id_to_doc = {i: (d, m) for i, d, m in zip(s["bm25_ids"], s["bm25_docs"], s["bm25_metas"])}
    merged_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    chunks = []
    for doc_id in merged_ids:
        if doc_id not in id_to_doc:
            continue
        doc, meta = id_to_doc[doc_id]
        if allowed_sources and meta.get("source") not in allowed_sources:
            continue
        chunks.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "chunk_index": meta.get("chunk_index", "unknown")
        })
        if len(chunks) >= k:
            break
    return chunks


def build_context(chunks):
    return "\n\n---\n\n".join(
        f"[source: {c['source']}, chunk {c['chunk_index']}]\n{c['text']}" for c in chunks
    )

#Request/Response models

class ChatRequest(BaseModel):
    question: str
    sources: Optional[List[str]] = None

class ChatResponse(BaseModel):
    answer: str
    chunks_used: List[dict]

#Endpoints

@app.get("/documents")
def list_documents():
    files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.pdf','.docx'))]
    return {"documents": files}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    s = get_state()
    chunks = retrieve_chunks_hybrid(req.question, allowed_sources=req.sources)
    context = build_context(chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=req.question)
    response = s["llm"].invoke(prompt)
    return {"answer": response.content, "chunks_used": chunks}

@app.get("/links/{filename}")
def get_links(filename: str):
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path) or not filename.lower().endswith(".pdf"):
        return {"links": []}
    return {"links": get_links_for_file(file_path, filename)}

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    saved = []
    for file in files:
        path = os.path.join(DATA_DIR, file.filename)
        with open(path, "wb") as f:
            f.write(await file.read())
        saved.append(file.filename)
    return {"uploaded": saved}

@app.post("/ingest")
def run_ingestion():
    result = subprocess.run(["python3", "ingest.py"], capture_output=True, text=True)
    get_state()
    rebuild_bm25()
    return {"output": result.stdout[-2000:], "sucess": result.returncode == 0}

