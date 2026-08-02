import fitz
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
import chromadb
import os
import base64
from pathlib import Path

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "source"
VISION_MODEL = "llava:7b"
TEMP_IMG_DIR = "temp_images"

os.makedirs(TEMP_IMG_DIR, exist_ok=True)

#PDF handling

def load_pdf_text_and_images(path):
    doc = fitz.open(path)
    full_text = ""
    images = []

    for page_num, page in enumerate(doc):
        text = page.get_text()
        full_text += f"\n[page {page_num + 1}]\n{text}"

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]

            img_filename = f"{Path(path).stem}_page{page_num + 1}_img{img_index + 1}.{ext}"
            img_path = os.path.join(TEMP_IMG_DIR, img_filename)

            with open(img_path, "wb") as img_file:
                img_file.write(image_bytes)

            images.append((img_path, page_num + 1))

    doc.close()
    return full_text, images


#DOCX handling

def load_docx_text_and_images(path):
    """"""
    doc = Document(path)
    full_text = "\n".join([para.text for para in doc.paragraphs])

    images= []
    img_index = 0
    for rel in doc.part.rels.values():
        if "image" in rel.target_ref:
            image_bytes = rel.target_part.blob
            ext = rel.target_ref.split(".")[-1]
            img_filename = f"{Path(path).stem}_img{img_index}.{ext}"
            img_path = os.path.join(TEMP_IMG_DIR, img_filename)
            with open(img_path, "wb") as img_file:
                img_file.write(image_bytes)
            images.append((img_path, None))
            img_index += 1

    return full_text, images

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
