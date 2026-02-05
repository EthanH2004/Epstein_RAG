from pathlib import Path
import os
import json
import hashlib
from typing import List, Dict, Tuple

from dotenv import load_dotenv
import tiktoken
import faiss
import numpy as np
from pypdf import PdfReader
from openai import OpenAI

load_dotenv()

# -------- Config --------
PDF_DIR = Path(os.getenv("PDF_DIR", "data/raw_pdfs"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", "data/index"))

CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))

# -------- Paths --------
FAISS_INDEX_PATH = INDEX_DIR / "index.faiss"
METADATA_PATH = INDEX_DIR / "metadata.jsonl"
STATE_FILE_PATH = INDEX_DIR / "state.json"

# -------- Setup --------
INDEX_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
enc = tiktoken.get_encoding("cl100k_base")


def get_file_hash(filepath: Path) -> str:
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()


def load_state() -> Dict:
    if STATE_FILE_PATH.exists():
        with open(STATE_FILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"file_hashes": {}, "next_id": 0}


def save_state(state: Dict) -> None:
    with open(STATE_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def extract_text_from_pdf(pdf_path: Path) -> Dict[int, str]:
    page_texts: Dict[int, str] = {}
    reader = PdfReader(str(pdf_path))
    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            page_texts[page_idx] = text
    return page_texts


def chunk_text_by_tokens(text: str) -> List[str]:
    tokens = enc.encode(text)
    if not tokens:
        return []

    stride = max(1, CHUNK_TOKENS - CHUNK_OVERLAP)
    chunks: List[str] = []

    for i in range(0, len(tokens), stride):
        chunk_tokens = tokens[i : i + CHUNK_TOKENS]
        if not chunk_tokens:
            continue
        chunk_str = enc.decode(chunk_tokens)
        chunks.append(chunk_str)

    return chunks


def embed_chunks_batch(chunks: List[str]) -> List[List[float]]:
    embeddings: List[List[float]] = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        print(
            f"  Embedding batch {i // BATCH_SIZE + 1}/"
            f"{(len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE}..."
        )
        resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=batch)

        # preserve order
        for item in sorted(resp.data, key=lambda x: x.index):
            embeddings.append(item.embedding)

    return embeddings


def ensure_idmap_index(index: faiss.Index, dim: int) -> faiss.Index:
    """
    Always use an ID-mapped index so FAISS returns our chunk IDs.
    - New: IndexIDMap2(IndexFlatL2)
    - Existing: keep if already IDMap; otherwise wrap.
    """
    if index is None:
        base = faiss.IndexFlatL2(dim)
        return faiss.IndexIDMap2(base)

    # If it's already an IDMap, keep it
    if isinstance(index, faiss.IndexIDMap) or isinstance(index, faiss.IndexIDMap2):
        return index

    # Wrap existing (rare in your case, but safe)
    return faiss.IndexIDMap2(index)


def process_pdf(
    pdf_path: Path,
    pdf_name: str,
    state: Dict,
) -> Tuple[List[np.ndarray], List[int], List[Dict]]:
    """
    Returns: (embeddings_np_list, ids_list, metadata_list)
    """
    file_hash = get_file_hash(pdf_path)
    current_hash = state["file_hashes"].get(pdf_name)

    if current_hash == file_hash:
        print(f"  Skipping {pdf_name} (unchanged)")
        return [], [], []

    print(f"  Processing {pdf_name}...")

    page_texts = extract_text_from_pdf(pdf_path)

    new_embeddings: List[np.ndarray] = []
    new_ids: List[int] = []
    new_metadata: List[Dict] = []

    for page_num, page_text in page_texts.items():
        chunks = chunk_text_by_tokens(page_text)
        if not chunks:
            continue

        embeddings = embed_chunks_batch(chunks)
        if len(embeddings) != len(chunks):
            raise RuntimeError("Embedding count mismatch vs chunk count")

        for chunk_idx, (chunk_str, emb) in enumerate(zip(chunks, embeddings)):
            chunk_id = int(state["next_id"])
            state["next_id"] += 1

            new_ids.append(chunk_id)
            new_embeddings.append(np.array(emb, dtype=np.float32))

            preview = chunk_str[:200].replace("\n", " ").strip()
            if len(chunk_str) > 200:
                preview += "..."

            new_metadata.append(
                {
                    "id": chunk_id,
                    "pdf_name": pdf_name,
                    "page": int(page_num),
                    "chunk": int(chunk_idx),
                    "text_preview": preview,
                    "text": chunk_str,
                }
            )

    state["file_hashes"][pdf_name] = file_hash
    print(f"    Added {len(new_ids)} chunks")
    return new_embeddings, new_ids, new_metadata


def main() -> None:
    print("Building RAG index...")

    state = load_state()
    print(f"State: {state['next_id']} chunks indexed so far")

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs")
    if not pdfs:
        print("No PDFs found. Exiting.")
        return

    # Load existing index if present
    index = None
    if FAISS_INDEX_PATH.exists():
        print("Loading existing FAISS index...")
        index = faiss.read_index(str(FAISS_INDEX_PATH))

    all_new_vecs: List[np.ndarray] = []
    all_new_ids: List[int] = []
    all_new_meta: List[Dict] = []

    for pdf_path in pdfs:
        vecs, ids, meta = process_pdf(pdf_path, pdf_path.name, state)
        all_new_vecs.extend(vecs)
        all_new_ids.extend(ids)
        all_new_meta.extend(meta)

    if not all_new_vecs:
        print("No new chunks to index. Exiting.")
        save_state(state)
        return

    embeddings_array = np.vstack(all_new_vecs).astype(np.float32)
    ids_array = np.array(all_new_ids, dtype=np.int64)

    dim = embeddings_array.shape[1]
    index = ensure_idmap_index(index, dim)

    print(f"\nAdding {len(all_new_ids)} vectors to FAISS...")
    # ID-mapped index requires add_with_ids
    index.add_with_ids(embeddings_array, ids_array)

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"Saved FAISS index to {FAISS_INDEX_PATH}")

    # Append metadata (do NOT overwrite)
    with open(METADATA_PATH, "a", encoding="utf-8") as f:
        for m in all_new_meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"Appended metadata to {METADATA_PATH}")

    save_state(state)
    print(f"State saved with {state['next_id']} total chunks")
    print("Done.")


if __name__ == "__main__":
    main()