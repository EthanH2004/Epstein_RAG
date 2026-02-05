import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set

import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# -------- Config --------
INDEX_DIR = Path(os.getenv("INDEX_DIR", "data/index"))
FAISS_INDEX_PATH = INDEX_DIR / "index.faiss"
METADATA_PATH = INDEX_DIR / "metadata.jsonl"

OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
TOP_K_DEFAULT = int(os.getenv("TOP_K", "8"))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_index_and_metadata_by_id() -> Tuple[faiss.Index, Dict[int, Dict]]:
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}")

    print("Loading FAISS index...")
    index = faiss.read_index(str(FAISS_INDEX_PATH))

    print("Loading metadata...")
    metadata_by_id: Dict[int, Dict] = {}
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            metadata_by_id[int(m["id"])] = m

    print(f"Loaded {len(metadata_by_id)} chunks")
    return index, metadata_by_id


def embed_question(question: str) -> np.ndarray:
    resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=[question])
    vec = np.array(resp.data[0].embedding, dtype=np.float32)
    return vec.reshape(1, -1)


def search_faiss(
    index: faiss.Index,
    metadata_by_id: Dict[int, Dict],
    query_vec: np.ndarray,
    top_k: int,
    filter_pdf: str = None,
) -> List[Dict]:
    distances, ids = index.search(query_vec, top_k)

    results: List[Dict] = []
    for chunk_id in ids[0]:
        if int(chunk_id) == -1:
            continue
        meta = metadata_by_id.get(int(chunk_id))
        if not meta:
            continue
        if filter_pdf and filter_pdf.lower() not in meta.get("pdf_name", "").lower():
            continue
        results.append(meta)

    return results


def build_prompt(question: str, retrieved: List[Dict]) -> str:
    ctx_lines: List[str] = []
    for i, chunk in enumerate(retrieved, 1):
        citation = f"[{chunk['pdf_name']} p.{chunk['page']} chunk{chunk['chunk']}]"
        text = chunk.get("text") or chunk.get("text_preview", "")
        ctx_lines.append(f"{i}. {text}\n   {citation}")

    context = "\n\n".join(ctx_lines)

    return (
        "You must answer using ONLY the provided context. "
        "If the context is insufficient, say you don't have enough information.\n"
        "Cite sources inline exactly like: [pdf_name p.# chunk#].\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n"
    )


def extract_citations(answer: str, retrieved: List[Dict]) -> Set[str]:
    citations: Set[str] = set()
    ans_lower = answer.lower()
    for chunk in retrieved:
        citation = f"[{chunk['pdf_name']} p.{chunk['page']} chunk{chunk['chunk']}]"
        if citation.lower() in ans_lower:
            citations.add(citation)
    return citations


def query_llm(prompt: str) -> str:
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Answer only from the provided context and always cite sources."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=900,
    )
    return resp.choices[0].message.content


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG CLI: Query your indexed PDFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Only use chunks from PDFs containing this substring (case-insensitive).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K_DEFAULT,
        help=f"Number of chunks to retrieve (default: {TOP_K_DEFAULT})",
    )
    args = parser.parse_args()

    try:
        index, metadata_by_id = load_index_and_metadata_by_id()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run the indexer first: py src/ingest/build_index.py")
        return

    print("\nRAG CLI - Type 'exit' or 'quit' to exit")
    print("=" * 60)

    while True:
        try:
            question = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        try:
            print("Embedding question...")
            qvec = embed_question(question)

            print(f"Searching top {args.top_k} chunks...")
            retrieved = search_faiss(
                index=index,
                metadata_by_id=metadata_by_id,
                query_vec=qvec,
                top_k=args.top_k,
                filter_pdf=args.filter,
            )

            if not retrieved:
                print("No relevant chunks found.")
                continue

            prompt = build_prompt(question, retrieved)

            print("Querying LLM...")
            answer = query_llm(prompt)
            citations = sorted(extract_citations(answer, retrieved))

            print("\nAnswer:")
            print(answer)

            if citations:
                print("\nCited sources:")
                for c in citations:
                    print(f"  {c}")

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()