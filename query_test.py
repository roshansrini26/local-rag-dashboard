from langchain_ollama import OllamaEmbeddings
import chromadb

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "source"

def main():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    query = "What is self-attention"
    query_vector = embeddings.embed_query(query)

    results = collection.query(
        query_embeddings = [query_vector],
        n_results = 3
    )

    print(f"Query: {query}\n")
    print(results.get("metadatas"))
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i] if results.get("metadatas") else {}
        source = meta.get("source", "unknown") if meta else "unknown"
        chunk_idx = meta.get("chunk_index", "unknown") if meta else "unknown"

        print(f"Answer {i+1} (source: {source}, chunk {chunk_idx}) -- ")
        print(doc[:300])
        print()

if __name__ == "__main__":
    main()