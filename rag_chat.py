from langchain_ollama import OllamaEmbeddings, ChatOllama
import chromadb

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "source"
LLM_MODEL = "llama3.1:8b"
TOP_K = 4

PROMPT_TEMPLATE = """You are a research assistant answering questions only based on the provided context from documents.

Context:
{context} 

Question: {question}

Instructions:
-Answer using only the information provided in the context.
-If the context doesnot contain enough information to answer, say so clearly.
-After your answer, cite the sources used in this format: (source: <source_name>, chunk: <chunk_index>).

Answer:"""

def retrieve_chunks(query, embeddings, collection, k=TOP_K):
    query_vector = embeddings.embed_query(query)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        include=["documents", "metadatas","distances"]
    )
    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
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

    print("Welcome to the RAG Chatbot! Type 'exit' to quit.")

    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit"]:
            break
        if not question:
            continue

        chunks = retrieve_chunks(question, embeddings, collection)
        context = build_context(chunks)

        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        response = llm.invoke(prompt)

        print(f"\nAssistant: {response.content}\n")

if __name__ == "__main__":
    main()