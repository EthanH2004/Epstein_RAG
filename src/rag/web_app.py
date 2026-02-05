import os
import json
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
import faiss
import streamlit as st
from openai import OpenAI

load_dotenv()

INDEX_DIR = Path(os.getenv("INDEX_DIR", "data/index"))
FAISS_INDEX_PATH = INDEX_DIR / "index.faiss"
METADATA_PATH = INDEX_DIR / "metadata.jsonl"

OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
TOP_K_DEFAULT = int(os.getenv("TOP_K", "8"))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@st.cache_resource
def load_index_and_meta():
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    meta_by_id = {}
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                m = json.loads(line)
                meta_by_id[int(m["id"])] = m
    return index, meta_by_id

def embed_question(q: str) -> np.ndarray:
    r = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=[q])
    v = np.array(r.data[0].embedding, dtype=np.float32)
    return v.reshape(1, -1)

def search(index, meta_by_id, qvec, top_k, filter_pdf=None):
    D, I = index.search(qvec, top_k)
    hits = []
    for cid in I[0]:
        if int(cid) == -1:
            continue
        m = meta_by_id.get(int(cid))
        if not m:
            continue
        if filter_pdf and filter_pdf.lower() not in m["pdf_name"].lower():
            continue
        hits.append(m)
    return hits

def build_prompt(question, retrieved):
    blocks = []
    for ch in retrieved:
        cite = f"[{ch['pdf_name']} p.{ch['page']} chunk{ch['chunk']}]"
        text = ch.get("text") or ch.get("text_preview", "")
        blocks.append(f"{text}\n{cite}")
    context = "\n\n---\n\n".join(blocks)

    return (
        "Answer using ONLY the context. If insufficient, say so. "
        "Cite sources inline exactly like [pdf_name p.# chunk#].\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n"
    )

def ask_llm(prompt: str) -> str:
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are a careful assistant who always cites sources."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=900,
    )
    return resp.choices[0].message.content

def main():
    st.set_page_config(page_title="Epstein RAG", layout="wide")
    
    # Hide Streamlit's default loading indicators and customize spinner size
    st.markdown("""
        <style>
        /* Hide top-right loading indicators */
        .stAppToolbar, [data-testid="stToolbar"] { display: none !important; }
        
        /* Make spinner bigger */
        .stSpinner > div { scale: 2; }
        .stSpinner { display: flex; justify-content: center; align-items: center; }
        
        /* Hide spinner text */
        .stSpinner > div:nth-child(2) { display: none !important; }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Epstein RAG")

    index, meta_by_id = load_index_and_meta()

    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Top K", 3, 30, TOP_K_DEFAULT)
        filter_pdf = st.text_input("Filter PDF name contains", "")
        filter_pdf = filter_pdf.strip() or None

    col1, col2 = st.columns([0.9, 0.1])
    with col1:
        q = st.text_area("Ask a question", height=100)
    with col2:
        st.write("")
        st.write("")
        cancel_clicked = st.button("Cancel", use_container_width=True)
    
    if cancel_clicked:
        st.session_state.clear()
        st.rerun()
    
    if st.button("Ask") and q.strip():
        with st.spinner():
            qvec = embed_question(q.strip())
        
        with st.spinner():
            retrieved = search(index, meta_by_id, qvec, top_k, filter_pdf=filter_pdf)

        if not retrieved:
            st.warning("No chunks retrieved. Try increasing Top K or removing filter.")
            return

        prompt = build_prompt(q.strip(), retrieved)
        
        with st.spinner():
            answer = ask_llm(prompt)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Answer")
            st.markdown(answer)

        with col2:
            st.subheader("Sources (retrieved)")
            for ch in retrieved:
                st.write(f"- {ch['pdf_name']} p.{ch['page']} chunk{ch['chunk']}")

if __name__ == "__main__":
    main()
