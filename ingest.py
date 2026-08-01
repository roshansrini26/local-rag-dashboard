import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import chromadb
import os

PDF_PATH = "data/1706.03762v7.pdf" #text file
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "source"

def load_pdf_text(path):
    doc = fitz.open(path)
    full_text = ""
    for page_num, page in enumerate(doc):
        text = page.get_text()
        full_text += f"\n[page {page_num + 1}]\n{text}"
    doc.close()
    return full_text

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)

def main():
    print(f"Loading {PDF_PATH}...")
    text = load_pdf_text(PDF_PATH)
    print(f"Extracted characters {len(text)}")

    print("Chunking...")
    chunks = chunk_text(text)
    print(f"created {len(chunks)} chunks")

    print("Setting up embeddings (Ollama nomic-embed-text)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print("Connencting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    filename = os.path.basename(PDF_PATH)

    print("Embedding and storing chunks ...")
    for i, chunk in enumerate(chunks):
        vector = embeddings.embed_query(chunk)
        collection.add(
            ids=[f"{filename}_chunk_{i}"],
            embeddings=[vector],
            documents=[chunk],
            metadatas=[{"source": filename, "chunk_index":i}]
        )
        if (i+1)%10 == 0:
            print(f"{i+1}/{len(chunks)} chunks stored")

    print(f"\n Finished. {len(chunks)} chunks stored in ChromaDB collection '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()
