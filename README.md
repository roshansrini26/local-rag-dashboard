# Local Document Assistant

A private, fully local assistant that answers questions across a collection of documents and returns answers grounded in cited source passages.

Everything runs on the host machine. No document content, query, or embedding is sent to an external service, which makes the system usable in settings where confidentiality rules out cloud-hosted assistants.

---

## Why this exists

Searching finds documents. It does not answer questions that span several of them.

When a body of material grows past a few dozen files, the practical problem is no longer locating a file — it is that the answer to a given question lives in three different documents that share no common vocabulary. Keyword search returns a list to read. This system returns an answer, with a pointer to the exact passage behind each claim.

Three properties drove the design:

- **Local execution** — the corpus may be confidential, so nothing leaves the machine.
- **Verifiable answers** — every response cites the source passages it drew on, so claims can be checked rather than trusted.
- **Controlled updates** — when a tracked source changes, the change is surfaced for review rather than silently absorbed.

---

## Capabilities

**Multi-format ingestion** — PDF and DOCX files, plus live web pages tracked by URL.

**Multimodal content** — text, tables, and figures are each handled distinctly. Tables are extracted as structured markdown rather than flattened prose; figures and diagrams are described by a local vision-language model and indexed as searchable text.

**Hybrid retrieval** — dense (semantic) and sparse (keyword) retrieval run in parallel and are merged, so both paraphrased questions and exact terminology are matched.

**Cited answers** — responses reference the source document and chunk behind each statement.

**Source change detection** — tracked URLs are re-checked on demand. Content changes are flagged and held for approval; nothing enters the index automatically.

**Incremental ingestion** — content is hashed, so re-running ingestion processes only new or modified material.

**Scoped querying** — retrieval can be restricted to a chosen subset of sources.

---

## Architecture

```
                    ┌─────────────────────────────┐
                    │   React frontend (Vite)     │
                    └──────────────┬──────────────┘
                                   │  HTTP
                    ┌──────────────▼──────────────┐
                    │      FastAPI backend        │
                    └──────────────┬──────────────┘
                 ┌─────────────────┼─────────────────┐
                 │                 │                 │
        ┌────────▼────────┐ ┌──────▼──────┐ ┌────────▼────────┐
        │   Ingestion     │ │  Retrieval  │ │   Generation    │
        │                 │ │             │ │                 │
        │ Docling         │ │ ChromaDB    │ │ llama3.1:8b     │
        │ PyMuPDF         │ │  (dense)    │ │  via Ollama     │
        │ python-docx     │ │ BM25        │ │                 │
        │ qwen2.5vl:7b    │ │  (sparse)   │ │                 │
        │ nomic-embed-text│ │ RRF merge   │ │                 │
        └─────────────────┘ └─────────────┘ └─────────────────┘
```

### Ingestion

Documents are routed by content. A fast text pass checks for indicators of structural complexity (tables, mathematical notation); documents that show them go through Docling's layout-aware pipeline, which uses a trained table-structure model. Simpler documents take a faster direct-extraction path. This routing exists because full layout analysis is significantly slower, and most documents do not need it.

Embedded images are rendered from the page at 3× resolution — rather than extracted as raw embedded objects, which are often too low-resolution to interpret — and passed to a local vision-language model, which produces a factual description that is indexed alongside the text.

Tables are extracted as markdown with their captions attached, so a retrieved table carries the context needed to identify which table it is.

### Chunking

Text is first split on markdown section headers, then divided within each section at sentence boundaries up to a size limit, with a two-sentence overlap between consecutive chunks.

Splitting on headers first prevents a chunk from spanning two unrelated sections. Splitting on sentences rather than character counts prevents chunks from terminating mid-thought. The section path is prepended to each chunk, so retrieved passages carry their position in the document.

### Retrieval

Two retrievers run per query:

- **Dense** — the query is embedded with the same model used at ingestion and matched against chunk vectors in ChromaDB (HNSW index, cosine similarity).
- **Sparse** — BM25 over the same chunks.

Results are merged with Reciprocal Rank Fusion. RRF is rank-based rather than score-based, because BM25 scores and cosine distances are not on a comparable scale.

Final selection interleaves across sources so that a single document cannot occupy every slot. This was added after a cross-document query returned results exclusively from one document, despite directly relevant content existing in another.

### Generation

Retrieved passages are assembled into a prompt that constrains the model to the supplied context: answer only from what is given, quote exact wording where precision matters, present conflicting sources rather than silently selecting one, and state plainly when the context is insufficient.

### Change tracking

Tracked URLs are hashed on their **extracted text**, not raw HTML — raw markup contains dynamic elements that change on every request and would produce constant false positives.

On check, each source is re-fetched and re-hashed. A differing hash sets the source to `changed` and stores the new hash as pending. The index is untouched.

Approval deletes the source's existing chunks, re-ingests the new content, and rebuilds the keyword index. Rejection leaves the index as-is and updates the comparison baseline so the same change is not re-flagged.

---

## Stack

**Models** (all local, served by Ollama)

| Model | Role |
|---|---|
| `llama3.1:8b` | Answer generation |
| `nomic-embed-text` | Embeddings — documents and queries |
| `qwen2.5vl:7b` | Figure and diagram description |

**Processing** — Docling, PyMuPDF, python-docx, requests

**Retrieval** — ChromaDB, rank_bm25, Reciprocal Rank Fusion

**Backend** — FastAPI, uvicorn, LangChain

**Frontend** — React, Vite, react-markdown

**Supporting** — watchdog (filesystem monitoring), SHA-256 content hashing

---

## Setup

**Requirements** — Python 3.11, Node.js, [Ollama](https://ollama.com), roughly 12 GB free disk for models.

```bash
# Models
ollama pull llama3.1:8b
ollama pull nomic-embed-text
ollama pull qwen2.5vl:7b

# Backend
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Frontend
cd frontend && npm install
```

**Running**

```bash
# Terminal 1
uvicorn api:app --reload --port 8000

# Terminal 2
cd frontend && npm run dev
```

The interface is served at `http://localhost:5173`; interactive API documentation at `http://localhost:8000/docs`.

**Ingesting**

Place documents in `data/` and run `python3 ingest.py`, or add them through the interface. Ingestion is incremental — unchanged files are skipped.

---

## Project structure

```
ingest.py            Ingestion pipeline: parsing, chunking, embedding,
                     URL tracking, change detection
api.py               FastAPI service and retrieval logic
link_extractor.py    Extracts in-document links, excluding bibliographies
watcher.py           Filesystem monitor for automatic ingestion
rag_chat.py          Command-line interface
frontend/            React application
data/                Source documents (not tracked in version control)
chroma_db/           Vector store (generated)
manifest.json        Ingestion state and source tracking (generated)
```

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /sources` | All tracked sources with status |
| `POST /chat` | Query, optionally scoped to selected sources |
| `POST /upload` | Upload documents |
| `POST /ingest` | Run ingestion over new or modified files |
| `POST /add-url` | Track a web page as a source |
| `POST /check-updates` | Re-check tracked URLs for content changes |
| `POST /approve-update` | Accept a change and re-index the source |
| `POST /reject-update` | Keep the indexed version |
| `GET /links/{filename}` | Links extracted from a document |

---

## Design decisions

**Layout-aware extraction over direct text extraction.** Heuristic table detection missed real tables in test documents, which produced incomplete answers on questions whose answers lived in those tables. A trained layout model caught them. The cost is ingestion speed, which is a one-time per-document cost rather than a per-query one.

**Page rendering over raw image extraction.** Images embedded in PDFs are frequently stored at resolutions too low for a vision model to interpret. Rendering the page region at higher resolution produces usable input.

**Hybrid retrieval over dense-only.** Semantic retrieval alone missed exact terminology; keyword retrieval alone missed paraphrased questions. Both are needed.

**Content hashing over timestamps.** File modification times change without content changing. Hashing content means work is only repeated when the content genuinely differs.

**Review before indexing.** A knowledge base that silently updates is a knowledge base whose users cannot know what it currently contains. Changes are surfaced and held.

---

## Known limitations

**Reasoning depth** is bounded by the 8B local model. Factual retrieval and grounded summarisation are reliable; complex multi-step inference across many sources is not. This is a deliberate trade — a larger model would improve it at the cost of hardware requirements that conflict with local execution.

**Mathematical notation** is detected but not transcribed. Docling supports formula recognition, but enabling it disables GPU acceleration on Apple Silicon and has reported stability issues, so it remains off. Formulas appear in the index as placeholders.

**No formal evaluation set.** Retrieval quality has been validated against specific test questions rather than a labelled benchmark, so improvements are assessed qualitatively. A held-out question set with known answers is the natural next step.

**No re-ranking stage.** Retrieved candidates go directly to generation. A cross-encoder re-ranking pass is the single most likely source of further retrieval improvement.

**Round-robin source selection** guarantees each source is represented but can promote a weakly relevant passage from an under-represented source. A score-aware variant would be more precise.

**Rejected changes update the comparison baseline**, so subsequent checks compare against the rejected version rather than the indexed one. Tracking indexed and reviewed hashes separately would be more correct.

---

## Possible extensions

- Automatic summaries generated per document at ingestion
- Monitoring index pages to detect newly published documents
- Scheduled rather than manual source checking
- Retrieval evaluation against a labelled question set
- Cross-encoder re-ranking
