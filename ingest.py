import fitz
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
import chromadb
import os
import base64
from pathlib import Path
import ollama as ollama_client

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "source"
VISION_MODEL = "qwen2.5vl:7b"
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
            bbox = page.get_image_bbox(img)

            mat = fitz.Matrix(3, 3)  # 3x zoom for higher resolution
            pix = page.get_pixmap(matrix=mat, clip=bbox)

            img_filename = f"{Path(path).stem}_page{page_num + 1}_img{img_index + 1}.png"
            img_path = os.path.join(TEMP_IMG_DIR, img_filename)
            pix.save(img_path)

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

#Image description via vision model

def describe_image(image_path, vision_llm=None):
    response = ollama_client.chat(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": "Describe this figure/diagram/chart in detail. Explain what it shows, "
                        "any labels, axes, or data trends visible. Be specific and factual.",
            "images": [image_path]  # pass the file path directly, not base64
        }]
    )
    return response["message"]["content"]

#chunking

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)


#main ingestion
def ingest_file(path, embeddings, collection, vision_llm):
    ext = Path(path).suffix.lower()
    filename = os.path.basename(path)

    print(f"\nLoading {filename}...")
    if ext == ".pdf":
        text, images = load_pdf_text_and_images(path)
    elif ext == ".docx":
        text, images = load_docx_text_and_images(path)
    else:
        print(f"Unsupported file type: {filename}. Skipping.")
        return

    print(f"Extracted characters {len(text)} and {len(images)} images.")

    #Text chunks
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} text chunks.")

    for i, chunk in enumerate(chunks):
        vector = embeddings.embed_query(chunk)
        collection.add(
            ids=[f"{filename}_text_{i}"],
            embeddings=[vector],
            documents=[chunk],
            metadatas=[{
                "source": filename, 
                "chunk_index": i,
                "content_type": "text"
                }]
        )
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(chunks)} text chunks stored.")

    #Image chunks (described via vision model, embedded and stored)

    if images:
        print(f"Describing {len(images)} images with {VISION_MODEL}...")
        for i, (img_path, page_num) in enumerate(images):
            try:
                description = describe_image(img_path, vision_llm)
                figure_text = f"[Figure from {filename}, page {page_num}]\n{description}"
                vector = embeddings.embed_query(figure_text)
                collection.add(
                    ids=[f"{filename}_image_{i}"],
                    embeddings=[vector],
                    documents=[figure_text],
                    metadatas=[{
                        "source": filename,
                        "chunk_index": i,
                        "content_type": "image",
                        "page_number": page_num if page_num else -1
                    }]
                )
                print(f"Image {i + 1}/{len(images)} described and stored.")
            except Exception as e:
                print(f"Error describing image {img_path}: {e}")
        print(f"Done describing and storing {filename}")




def main():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vision_llm = ChatOllama(model=VISION_MODEL, temperature=0.1)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.pdf', '.docx'))]
    if not files:
        print(f"No PDF or DOCX files found in {DATA_DIR}.")
        return
    

    for filename in files:
        file_path = os.path.join(DATA_DIR, filename)
        if os.path.isfile(file_path):
            ingest_file(file_path, embeddings, collection, vision_llm)

    print("\nAll files ingested and stored in ChromaDB.")



if __name__ == "__main__":
    main()
