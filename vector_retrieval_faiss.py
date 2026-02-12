import os
import re
import time
import uuid
import argparse
import hashlib
import sqlite3
from typing import Optional, Dict, List, Tuple

import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # dim 384


# -----------------------------
# Utils
# -----------------------------
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def pdf_to_text_by_page(pdf_path: str) -> List[str]:
    doc = fitz.open(pdf_path)
    pages = []
    for p in doc:
        pages.append(p.get_text("text") or "")
    doc.close()
    return pages

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    text = text.replace("\r", "\n")
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) <= 1:
        blocks = [b.strip() for b in re.split(r"(?<=[\.\?\!])\s+", norm_space(text)) if b.strip()]

    chunks = []
    buf = ""
    for b in blocks:
        if not buf:
            buf = b
        elif len(buf) + 1 + len(b) <= max_chars:
            buf = buf + " " + b
        else:
            chunks.append(buf)
            tail = buf[-overlap:] if overlap > 0 else ""
            buf = (tail + " " + b).strip()
    if buf:
        chunks.append(buf)

    return [norm_space(c) for c in chunks if norm_space(c)]

SECTION_SPLIT_RE = re.compile(
    r"(?i)\b(\d+\s+(Conductor|Conductor Shielding|Insulation|Insulation Shielding|Metallic Screen|Cabling|Inner Sheath|Armour|Outer Sheath|Final Test|Packing))\b"
)

def chunk_by_sections(page_text: str, max_chars: int = 650, overlap: int = 120) -> List[str]:
    """
    Pecah text per halaman menjadi segment berdasarkan section header angka,
    lalu chunk kecil-kecil per segment.
    """
    t = " ".join((page_text or "").split())  # 1 paragraf per halaman
    if not t:
        return []

    # cari semua posisi section header
    hits = [(m.start(), m.group(1)) for m in SECTION_SPLIT_RE.finditer(t)]
    if not hits:
        return chunk_text(t, max_chars=max_chars, overlap=overlap)

    # segmentasi dari hit ke hit berikutnya
    segs = []
    for i, (pos, hdr) in enumerate(hits):
        end = hits[i+1][0] if i < len(hits)-1 else len(t)
        seg = t[pos:end].strip()
        if seg:
            segs.append(seg)

    # chunk tiap segment biar tidak kepanjangan
    out = []
    for seg in segs:
        out.extend(chunk_text(seg, max_chars=max_chars, overlap=overlap))
    return out



# -----------------------------
# SQLite metadata
# -----------------------------
def init_meta_db(db_path: str):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS files (
            file_hash TEXT PRIMARY KEY,
            file_path TEXT NOT NULL,
            indexed_at INTEGER NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            file_hash TEXT NOT NULL,
            file_path TEXT NOT NULL,
            page_no INTEGER NOT NULL,
            chunk_no INTEGER NOT NULL,
            chunk_text TEXT NOT NULL
        )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS file_paths (
        file_path TEXT PRIMARY KEY,
        file_hash TEXT NOT NULL,
        indexed_at INTEGER NOT NULL
    )
""")
    conn.commit()
    conn.close()

def is_file_indexed(db_path: str, file_hash: str) -> bool:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM files WHERE file_hash=? LIMIT 1", (file_hash,))
    ok = cur.fetchone() is not None
    conn.close()
    return ok

def register_file(db_path: str, file_hash: str, file_path: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO files (file_hash, file_path, indexed_at) VALUES (?,?,?)",
        (file_hash, file_path, int(time.time()))
    )
    conn.commit()
    conn.close()

def register_file_path(db_path: str, file_hash: str, file_path: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO file_paths (file_path, file_hash, indexed_at) VALUES (?,?,?)",
        (file_path, file_hash, int(time.time()))
    )
    conn.commit()
    conn.close()


def get_next_chunk_id(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COALESCE(MAX(id),0) FROM chunks")
    mx = cur.fetchone()[0]
    conn.close()
    return int(mx) + 1

def insert_chunk_meta(db_path: str, chunk_id: int, file_hash: str, file_path: str, page_no: int, chunk_no: int, chunk_text: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO chunks (id, file_hash, file_path, page_no, chunk_no, chunk_text) VALUES (?,?,?,?,?,?)",
        (chunk_id, file_hash, file_path, page_no, chunk_no, chunk_text)
    )
    conn.commit()
    conn.close()

def get_chunk_meta(db_path: str, chunk_id: int) -> Optional[Dict]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id,file_hash,file_path,page_no,chunk_no,chunk_text FROM chunks WHERE id=?", (chunk_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row[0],
        "file_hash": row[1],
        "file_path": row[2],
        "page_no": row[3],
        "chunk_no": row[4],
        "chunk_text": row[5],
    }

def get_file_chunk_ids(db_path: str, file_hash: str) -> set:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id FROM chunks WHERE file_hash=?", (file_hash,))
    ids = {r[0] for r in cur.fetchall()}
    conn.close()
    return ids

def keyword_score(text: str, keywords: List[str]) -> int:
    t = (text or "").lower()
    return sum(1 for k in keywords if k.lower() in t)

SECTION_KEYWORDS = {
    "Packing": ["packing", "wooden", "drum", "end cap", "standard length", "net", "gross", "weight"],
    "Final Test": ["final test", "resistance", "voltage", "partial discharge", "dimension", "visual", "ok"],
    "Outer Sheath": ["outer sheath", "sheath", "uv", "overall diameter"],
    # tambah kalau perlu
}


# -----------------------------
# FAISS index (cosine via inner product on normalized vectors)
# -----------------------------
def load_or_create_faiss(index_path: str, dim: int) -> faiss.Index:
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    if os.path.exists(index_path):
        return faiss.read_index(index_path)
    # inner product index; must use normalized vectors for cosine similarity
    return faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

def save_faiss(index: faiss.Index, index_path: str):
    faiss.write_index(index, index_path)

def normalize_vecs(x: np.ndarray) -> np.ndarray:
    # x shape: (n, dim)
    faiss.normalize_L2(x)
    return x


# -----------------------------
# Indexing
# -----------------------------
def index_pdf_folder(
    pdf_dir: str,
    model_name: str,
    meta_db: str,
    index_file: str,
    limit: int = 0,
    chunk_chars: int = 1200,
    overlap: int = 200
):
    init_meta_db(meta_db)
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()

    index = load_or_create_faiss(index_file, dim)

    pdfs = []
    for root, _, files in os.walk(pdf_dir):
        for fn in files:
            if fn.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, fn))
    pdfs.sort()
    if limit:
        pdfs = pdfs[:limit]

    print(f"[INFO] PDFs found: {len(pdfs)}")
    next_id = get_next_chunk_id(meta_db)

    for i, pdf_path in enumerate(pdfs, start=1):
        try:
            fh = sha256_file(pdf_path)
            # ✅ selalu catat path -> hash (walau duplikat)
            register_file_path(meta_db, fh, pdf_path)
            if is_file_indexed(meta_db, fh):
                print(f"[SKIP] {i}/{len(pdfs)} duplicate content (same hash): {os.path.basename(pdf_path)}")
                continue


            pages = pdf_to_text_by_page(pdf_path)

            all_chunks = []
            metas = []
            for pno, ptxt in enumerate(pages, start=1):
                if not ptxt.strip():
                    continue
                chunks = chunk_by_sections(ptxt, max_chars=chunk_chars, overlap=overlap)
                for cno, c in enumerate(chunks, start=1):
                    all_chunks.append(c)
                    metas.append((pno, cno, c))

            if not all_chunks:
                print(f"[FAIL] {i}/{len(pdfs)} no text chunks: {os.path.basename(pdf_path)}")
                continue

            vecs = model.encode(all_chunks, normalize_embeddings=True)
            vecs = np.asarray(vecs, dtype=np.float32)
            vecs = normalize_vecs(vecs)

            ids = np.arange(next_id, next_id + len(all_chunks), dtype=np.int64)
            index.add_with_ids(vecs, ids)

            for cid, (pno, cno, ctext) in zip(ids.tolist(), metas):
                insert_chunk_meta(meta_db, int(cid), fh, pdf_path, pno, cno, ctext)

            register_file(meta_db, fh, pdf_path)
            next_id += len(all_chunks)

            print(f"[OK] {i}/{len(pdfs)} indexed: {os.path.basename(pdf_path)} chunks={len(all_chunks)}")

        except Exception as e:
            print(f"[ERR] {pdf_path} -> {e}")

    save_faiss(index, index_file)
    print("[DONE] index saved:", index_file)


# -----------------------------
# Search
# -----------------------------
def search(
    query: str,
    model_name: str,
    meta_db: str,
    index_file: str,
    top_k: int = 5,
    file_filter: Optional[str] = None
):
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    index = load_or_create_faiss(index_file, dim)

    qv = model.encode([query], normalize_embeddings=True)
    qv = np.asarray(qv, dtype=np.float32)
    qv = normalize_vecs(qv)

    # faiss returns (scores, ids)
    scores, ids = index.search(qv, top_k * 10)
    ids = ids[0].tolist()
    scores = scores[0].tolist()

    results = []
    for cid, score in zip(ids, scores):
        if cid == -1:
            continue
        meta = get_chunk_meta(meta_db, int(cid))
        if not meta:
            continue
        if file_filter and file_filter not in meta["file_path"]:
            continue
        results.append((float(score), meta))
        if len(results) >= top_k:
            break

    for rank, (score, meta) in enumerate(results, start=1):
        print("=" * 90)
        print(f"#{rank} score={score:.4f} file={os.path.basename(meta['file_path'])} page={meta['page_no']} chunk={meta['chunk_no']} id={meta['id']}")
        print(meta["chunk_text"][:900])


SECTION_QUERIES = [
    ("Conductor", "Conductor number/diameter of wire conductor shape construction OD/LP/DL 1st layer 2nd layer"),
    ("Conductor Shielding", "Conductor Shielding average thickness semiconductive peroxide kg/km"),
    ("Insulation", "Insulation thickness mm minimum thickness xlpe peroxide quantity kg/km"),
    ("Insulation Shielding", "Insulation Shielding average thickness strippable semiconductive OS PEROXIDE quantity"),
    ("Metallic Screen", "Metallic Screen copper tape size n/mm core identification"),
    ("Cabling", "Cabling centre filler lay length outer diameter"),
    ("Inner Sheath", "Inner Sheath nominal thickness PVC filler outer diameter"),
    ("Armour", "Armour steel tape galvanized dimension"),
    ("Outer Sheath", "Outer Sheath nominal thickness PVC sheath outer diameter"),
    ("Final Test", "Final Test max dc conductor resistance ac voltage test visual all layer dimension ok"),
    ("Packing", "Packing wooden drum end cap standard length net weight gross weight"),
]

def extract_top_sections_for_pdf(
    pdf_path: str,
    model_name: str,
    meta_db: str,
    index_file: str,
    top_k_per_section: int = 3
):
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    index = load_or_create_faiss(index_file, dim)

    fh = sha256_file(pdf_path)
    file_ids = get_file_chunk_ids(meta_db, fh)
    if not file_ids:
        print("[INFO] file not indexed yet. Run index first.")
        return

    out = {}
    for sec, q in SECTION_QUERIES:
        qv = model.encode([q], normalize_embeddings=True)
        qv = np.asarray(qv, dtype=np.float32)
        qv = normalize_vecs(qv)

        scores, ids = index.search(qv, top_k_per_section * 30)
        ids = ids[0].tolist()
        scores = scores[0].tolist()

        cands = []
        for cid, score in zip(ids, scores):
            if cid == -1:
                continue
            if int(cid) not in file_ids:
                continue
            meta = get_chunk_meta(meta_db, int(cid))
            if not meta:
                continue
            cands.append((float(score), meta))
            if len(cands) >= 20:
                break

        best = None
        if cands:
            kws = SECTION_KEYWORDS.get(sec, [])
            # score akhir = faiss_score + 0.05 * keyword_score
            best = max(
                cands,
                key=lambda x: x[0] + 0.05 * keyword_score(x[1]["chunk_text"], kws)
    )


        if best:
            out[sec] = best

    for sec, (score, meta) in out.items():
        print("\n" + "#" * 90)
        print(f"[{sec}] score={score:.4f} page={meta['page_no']} chunk={meta['chunk_no']} id={meta['id']}")
        print(meta["chunk_text"][:900])


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index")
    p_index.add_argument("--pdf_dir", required=True)
    p_index.add_argument("--model", default=DEFAULT_MODEL)
    p_index.add_argument("--meta_db", default="./vector_store/meta/chunks.db")
    p_index.add_argument("--index_file", default="./vector_store/index/faiss.index")
    p_index.add_argument("--limit", type=int, default=0)
    p_index.add_argument("--chunk_chars", type=int, default=1200)
    p_index.add_argument("--overlap", type=int, default=200)

    p_search = sub.add_parser("search")
    p_search.add_argument("--query", required=True)
    p_search.add_argument("--model", default=DEFAULT_MODEL)
    p_search.add_argument("--meta_db", default="./vector_store/meta/chunks.db")
    p_search.add_argument("--index_file", default="./vector_store/index/faiss.index")
    p_search.add_argument("--top_k", type=int, default=5)
    p_search.add_argument("--file_filter", default=None)

    p_extract = sub.add_parser("extract")
    p_extract.add_argument("--pdf", required=True)
    p_extract.add_argument("--model", default=DEFAULT_MODEL)
    p_extract.add_argument("--meta_db", default="./vector_store/meta/chunks.db")
    p_extract.add_argument("--index_file", default="./vector_store/index/faiss.index")
    p_extract.add_argument("--top_k_per_section", type=int, default=3)

    args = ap.parse_args()

    if args.cmd == "index":
        index_pdf_folder(
            pdf_dir=args.pdf_dir,
            model_name=args.model,
            meta_db=args.meta_db,
            index_file=args.index_file,
            limit=args.limit,
            chunk_chars=args.chunk_chars,
            overlap=args.overlap
        )
    elif args.cmd == "search":
        search(
            query=args.query,
            model_name=args.model,
            meta_db=args.meta_db,
            index_file=args.index_file,
            top_k=args.top_k,
            file_filter=args.file_filter
        )
    elif args.cmd == "extract":
        extract_top_sections_for_pdf(
            pdf_path=args.pdf,
            model_name=args.model,
            meta_db=args.meta_db,
            index_file=args.index_file,
            top_k_per_section=args.top_k_per_section
        )

if __name__ == "__main__":
    main()
