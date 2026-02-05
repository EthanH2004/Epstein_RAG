# Epstein_RAG

A working end-to-end Retrieval-Augmented Generation (RAG) system for exploring DOJ Epstein disclosure PDFs.

This project indexes PDFs into a FAISS vector database, retrieves relevant text chunks for a query, and generates cited answers using an LLM. It includes both a command-line interface and a Streamlit web UI.

---

## Features

- PDF ingestion with page-level token chunking
- OpenAI embeddings + FAISS vector search
- Source-cited answers (PDF name, page, chunk)
- Interactive CLI for querying
- Streamlit web interface with filters and Top-K control
- Incremental indexing using file hashing
- Fully configurable via environment variables

---

## Repository Structure

src/
  ingest/
    build_index.py      # Builds/updates FAISS index from PDFs
  rag/
    ask.py              # CLI-based RAG interface
    web_app.py          # Streamlit web UI

data/
  raw_pdfs/             # PDFs to index (gitignored)
  index/                # FAISS index + metadata (gitignored)

.env.example            # Environment variable template
.vscode/launch.json     # VS Code run configuration (optional)

---

## Setup

Install dependencies:

py -m pip install pypdf faiss-cpu numpy python-dotenv openai tiktoken streamlit

Configure environment variables:

cp .env.example .env

Edit .env and add your OpenAI API key:

OPENAI_API_KEY=sk-...

---

## Index PDFs

Place PDFs in:

data/raw_pdfs/

Build or update the index:

py src/ingest/build_index.py

This process extracts text from PDFs, chunks by token count with overlap, embeds chunks, and stores vectors plus metadata in FAISS.

---

## Query via CLI

py src/rag/ask.py

Optional flags:

py src/rag/ask.py --top-k 12
py src/rag/ask.py --filter EFTA02205655

---

## Run Web UI (Streamlit)

py -m streamlit run src/rag/web_app.py

The web UI supports adjustable Top-K retrieval, PDF name filtering, cited answers, and source inspection. VS Code users can also run via the Play button if launch.json is configured.

---

## Notes

.env, PDFs, and FAISS index files are intentionally not committed. This system retrieves the most relevant chunks rather than the entire corpus and is intended for research and exploratory analysis.

---
