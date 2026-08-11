import fitz
import re
import os
import json

LINKS_DIR = "links"
os.makedirs(LINKS_DIR, exist_ok=True)

REFERENCE_HEADERS = ["reference", "bibliography", "works cited"]

def find_reference_start_page(doc):
    for page_num, page in enumerate(doc):
        text = page.get_text().strip().lower()
        for line in text.split("\n")[:5]:
            line_clean = line.strip().lower()
            if line_clean in REFERENCE_HEADERS or any(
                line_clean.startswith(h) for h in REFERENCE_HEADERS
            ):
                return page_num
    return None

def extract_links(pdf_path):
    doc = fitz.open(pdf_path)
    ref_start = find_reference_start_page(doc)

    links_by_normalized = {}

    url_pattern = re.compile(r'https?://[^\s\)\]]+')

    for page_num, page in enumerate(doc):
        if ref_start is not None and page_num >= ref_start:
            break

        candidates = []

        for link in page.get_links():
            uri = link.get("uri")
            if uri:
                candidates.append(uri)

        text = page.get_text()
        for match in url_pattern.findall(text):
            candidates.append(match.rstrip('.,;'))

        for url in candidates:
            key = url.rstrip('/.,;').lower()
            key = re.sub(r'\.\.\.$', '', key)  
            if key not in links_by_normalized or len(url) > len(links_by_normalized[key]["url"]):
                links_by_normalized[key] = {"url": url, "page": page_num + 1}

    doc.close()
    return list(links_by_normalized.values())

def get_links_for_file(pdf_path, filename):
    cache_path = os.path.join(LINKS_DIR, f"{filename}.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)

    links = extract_links(pdf_path)
    with open(cache_path, "w") as f:
        json.dump(links, f, indent=2)
    return links