from langchain_ollama import OllamaEmbeddings, ChatOllama
from rank_bm25 import BM25Okapi
import re
import chromadb

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "source"
LLM_MODEL = "llama3.1:8b"
TOP_K = 6

PROMPT_TEMPLATE = """You are a research assistant answering questions only based on the provided context from documents.

Context:
{context} 

Question: {question}

Instructions:
- Answer using only the information in the context above.
- If the context contains multiple different values that could answer the question (e.g., the same metric appears in different tables or experiments), mention all of them and clearly specify which table or section each comes from.
- If the context does not contain enough information to answer, say so clearly.
- After your answer, cite the sources used in this format: [source: filename, chunk X]

Answer:"""

def build_bm25_index(collection):
    all_items = collection.get(limit=10000, include=["documents", "metadatas"])
    documents = all_items["documents"]
    metadatas = all_items["metadatas"]
    ids = all_items["ids"]

    tokenized = [re.findall(r'\w+', doc.lower()) for doc in documents]
    bm25 = BM25Okapi(tokenized)

    return bm25, documents, metadatas, ids

def retrieve_chunks_hybrid(query, embeddings, collection,bm25, bm25_docs, bm25_metas, bm25_ids, k=TOP_K):
    #dense retreival
    query_vector = embeddings.embed_query(query)
    dense_results = collection.query(
        query_embeddings=[query_vector],
        n_results=k*2,
        include=["documents", "metadatas","distances"]
    )
    dense_ids = dense_results["ids"][0]

    #BM25 retrieval
    tokenized_query = re.findall(r'\w+', query.lower())
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k*2]
    bm25_top_ids = [bm25_ids[i] for i in bm25_ranked]

    #RRF (Reciprocal Rank Fusion): combine both rankings
    rrf_scores = {}
    rrf_k = 60
    for rank, doc_id in enumerate(dense_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (rrf_k + rank + 1)
    for rank, doc_id in enumerate(bm25_top_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (rrf_k + rank + 1)

    merged_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:k]

    id_to_doc = {doc_id: (doc, meta) for doc_id, doc, meta in zip(bm25_ids, bm25_docs, bm25_metas)}
    chunks = []
    for doc_id in merged_ids:
        doc, meta = id_to_doc[doc_id]
        chunks.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "chunk_index": meta.get("chunk_index", "unknown")
        })
    return chunks

def build_context(chunks):
    parts = []
    for c in chunks:
        parts.append(f"Source: {c['source']}, Chunk: {c['chunk_index']}\n{c['text']}")
    return "\n\n".join(parts)

def main():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    llm = ChatOllama(model=LLM_MODEL, temperature=0.2)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    print("Building BM25 index...")
    bm25, bm25_docs, bm25_metas, bm25_ids = build_bm25_index(collection)

    print("Welcome to the RAG Chatbot! Type 'exit' to quit.")

    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit"]:
            break
        if not question:
            continue

        chunks = retrieve_chunks_hybrid(question, embeddings, collection, bm25, bm25_docs, bm25_metas, bm25_ids)
        context = build_context(chunks)

        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        response = llm.invoke(prompt)

        print(f"\nAssistant: {response.content}\n")

if __name__ == "__main__":
    main()