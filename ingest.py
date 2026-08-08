import fitz
from docx import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from docling.document_converter import DocumentConverter
import chromadb
import os
import base64
from pathlib import Path
import ollama as ollama_client

import hashlib
import json
import re



DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "source"
VISION_MODEL = "qwen2.5vl:7b"
TEMP_IMG_DIR = "temp_images"
MANIFEST_FILE = "manifest.json"

os.makedirs(TEMP_IMG_DIR, exist_ok=True)

def get_file_hash(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def load_manifest():
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "r") as f:
            return json.load(f)
    return {}

def save_manifest(manifest):
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)


def needs_docling(quick_text):
    """Heuristic: route to Docling only if document likely has tables or heavy math"""
    table_signal = "table " in quick_text.lower()
    math_symbols = ["∑", "∫", "√", "≈", "≤", "≥", "±", "∂", "∇", "×10", "·10"]
    math_signal = any(s in quick_text for s in math_symbols)
    return table_signal or math_signal


def split_markdown_tables(markdown_text):
    lines = markdown_text.split("\n")
    text_lines = []
    tables = []
    current_table = []
    in_table = False
    last_nonempty_line = ""

    for line in lines:
        if line.strip().startswith("|"):
            if not in_table:
                # starting a new table — grab the line right before it as a likely caption
                current_table.append(f"CAPTION: {last_nonempty_line}")
            current_table.append(line)
            in_table = True
        else:
            if in_table:
                if len(current_table) >= 2:
                    tables.append("\n".join(current_table))
                current_table = []
                in_table = False
            text_lines.append(line)
            if line.strip():
                last_nonempty_line = line.strip()

    if in_table and len(current_table) >= 2:
        tables.append("\n".join(current_table))

    full_text = "\n".join(text_lines)
    return full_text, tables


#PDF handling

def load_pdf_text_and_images(path):
    doc = fitz.open(path)

    # Fast pass: quick text extraction for heuristic check + images
    quick_text = ""
    images = []
    for page_num, page in enumerate(doc):
        quick_text += page.get_text()
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            bbox = page.get_image_bbox(img)
            mat = fitz.Matrix(3, 3)
            pix = page.get_pixmap(matrix=mat, clip=bbox)
            img_filename = f"{Path(path).stem}_page{page_num + 1}_img{img_index + 1}.png"
            img_path = os.path.join(TEMP_IMG_DIR, img_filename)
            pix.save(img_path)
            images.append((img_path, page_num + 1))
    doc.close()

    if needs_docling(quick_text):
        print("  Table/formula indicators found — using Docling for accurate extraction...")
        converter = DocumentConverter()
        result = converter.convert(path)
        markdown = result.document.export_to_markdown()
        full_text, tables = split_markdown_tables(markdown)
        tables_with_pages = [(t, None) for t in tables]
    else:
        print("  No table/formula indicators — using fast text extraction...")
        full_text = ""
        for page_num, page in enumerate(fitz.open(path)):
            full_text += f"\n[page {page_num + 1}]\n{page.get_text()}"
        tables_with_pages = []

    return full_text, images, tables_with_pages


#DOCX handling

def load_docx_text_and_images(path):
    doc = Document(path)
    quick_text = "\n".join([para.text for para in doc.paragraphs])

    images = []
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

    if needs_docling(quick_text):
        print("  Table/formula indicators found — using Docling for accurate extraction...")
        converter = DocumentConverter()
        result = converter.convert(path)
        markdown = result.document.export_to_markdown()
        full_text, tables = split_markdown_tables(markdown)
        tables_with_pages = [(t, None) for t in tables]
    else:
        print("  No table/formula indicators — using fast text extraction...")
        full_text = quick_text
        tables_with_pages = []

    return full_text, images, tables_with_pages


#Image description via vision model

def describe_image(image_path, vision_llm=None):
    response = ollama_client.chat(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": "Describe this figure/diagram/chart in detail. Explain what it shows, "
                        "any labels, axes, or data trends visible. Be specific and factual.",
            "images": [image_path]
        }]
    )
    return response["message"]["content"]


#chunking

def chunk_by_sentence(text, max_chars=800, overlap_sentences=2):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s.strip()]
    chunks = []
    current = []
    current_len = 0

    for sentence in sentences:
        if current_len + len(sentence) > max_chars and current:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_len = sum(len(s) for s in current)
        current.append(sentence)
        current_len += len(sentence)

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_text(text):
    headers_to_split_on = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    try:
        sections = md_splitter.split_text(text)
    except Exception:
        sections = None

    all_chunks = []

    if sections:
        for section in sections:
            content = section.page_content.strip()
            if not content:
                continue
            header_prefix = "" 
            if section.metadata:
                header_prefix = " > ".join(section.metadata.values()) + "\n"
            section_chunks = chunk_by_sentence(content, max_chars=800, overlap_sentences=2)
            for c in section_chunks:
                all_chunks.append(header_prefix + c if header_prefix else c)
    else:
        all_chunks = chunk_by_sentence(text, max_chars=800, overlap_sentences=2)
    return all_chunks


#main ingestion
def ingest_file(path, embeddings, collection, vision_llm):
    ext = Path(path).suffix.lower()
    filename = os.path.basename(path)

    print(f"\nLoading {filename}...")
    if ext == ".pdf":
        text, images, tables = load_pdf_text_and_images(path)
    elif ext == ".docx":
        text, images, tables = load_docx_text_and_images(path)
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

    #Table chunks
    if tables:
        print(f"Storing {len(tables)} tables...")
        for i, (table_md, page_num) in enumerate(tables):
            table_text = f"[Table from {filename}, page {page_num}]\n{table_md}"
            vector = embeddings.embed_query(table_text)
            collection.add(
                ids=[f"{filename}_table_{i}"],
                embeddings=[vector],
                documents=[table_text],
                metadatas=[{
                    "source": filename,
                    "chunk_index": i,
                    "content_type": "table",
                    "page_number": page_num if page_num else -1
                }]
            )
            print(f"Table {i + 1}/{len(tables)} stored.")
        print(f"Done storing tables from {filename}.")

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

    manifest = load_manifest()

    files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.pdf', '.docx'))]
    if not files:
        print(f"No PDF or DOCX files found in {DATA_DIR}.")
        return

    for filename in files:
        file_path = os.path.join(DATA_DIR, filename)
        if not os.path.isfile(file_path):
            continue

        file_hash = get_file_hash(file_path)

        if filename in manifest and manifest[filename]["hash"] == file_hash:
            print(f"Skipping {filename}: already ingested and unchanged.")
            continue

        ingest_file(file_path, embeddings, collection, vision_llm)
        manifest[filename] = {"hash": file_hash}
        save_manifest(manifest)

    print("\nAll files ingested and stored in ChromaDB.")


if __name__ == "__main__":
    main()