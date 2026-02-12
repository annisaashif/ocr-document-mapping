import os
import argparse
import hashlib
import re
import sqlite3
from typing import Dict, Optional, Tuple, List

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text as sql_text

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SECTION_QUERIES = [
    ("Conductor", "1 Conductor Number/diameter of wire conductor shape OD/LP/DL 1st Layer 2nd Layer"),
    ("Conductor Shielding", "Conductor Shielding Average thickness Semiconductive PEROXIDE quantity"),
    ("Insulation", "Insulation Nom thickness Minimum thickness XLPE PEROXIDE quantity"),
    ("Insulation Shielding", "Insulation Shielding Average thickness Strippable Semiconductive OS PEROXIDE quantity 105,3"),
    ("Metallic Screen", "Metallic Screen copper tape Min OL/DL Core identification Outer diameter"),
    ("Cabling", "Cabling Centre filler OD/LP/DL Polyester tape Outer diameter"),
    ("Inner Sheath", "Inner Sheath Nom thickness Min thickness Outer diameter PVC"),
    ("Armour", "Armour Steel Tape Galv Max OG/DL Min OV Outer diameter"),
    ("Outer Sheath", "Outer Sheath Nom thickness UV Resistant Overall diameter Cable Marking by Embossed"),
    ("Final Test", "Final Test DC Conductor Resistance AC voltage test Partial discharge Insulation resistant Visual Ok Dimension Ok"),
    ("Packing", "Packing Wooden drum End cap Standard length Net Weight Gross weight"),
]

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def normalize_vecs(x: np.ndarray) -> np.ndarray:
    faiss.normalize_L2(x)
    return x

def load_index(index_file: str) -> faiss.Index:
    return faiss.read_index(index_file)

def get_chunk_meta(meta_db: str, chunk_id: int) -> Optional[Dict]:
    conn = sqlite3.connect(meta_db)
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

def get_file_chunk_ids(meta_db: str, file_hash: str) -> set:
    conn = sqlite3.connect(meta_db)
    cur = conn.cursor()
    cur.execute("SELECT id FROM chunks WHERE file_hash=?", (file_hash,))
    ids = {r[0] for r in cur.fetchall()}
    conn.close()
    return ids

def ensure_table(engine):
    with engine.begin() as conn:
        conn.execute(sql_text("""
            CREATE TABLE IF NOT EXISTS tds_retrieval_hits (
              id BIGINT AUTO_INCREMENT PRIMARY KEY,
              file_hash CHAR(64) NOT NULL,
              file_path VARCHAR(1024) NOT NULL,
              section VARCHAR(64) NOT NULL,
              score DOUBLE NOT NULL,
              page_no INT,
              chunk_no INT,
              chunk_id BIGINT,
              chunk_text LONGTEXT,
              created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
              UNIQUE KEY uq_file_section (file_hash, section)
            )
        """))

def clean_chunk_for_section(section: str, text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "")).strip()
    if not t:
        return t

    sec = section.lower()

    def find_pos(pattern: str, start_at: int = 0) -> Optional[int]:
        m = re.search(pattern, t[start_at:], flags=re.I)
        return (start_at + m.start()) if m else None

    # ===== Packing cleaner =====
    if sec == "packing":
        # start at "Packing" (with optional number)
        m_start = re.search(r"(?i)\b(\d+\s+Packing|Packing)\b", t)
        if not m_start:
            return t

        s = m_start.start()

        # end candidates (ambil yang paling awal setelah start)
        ends = []

        # stop at next section
        for pat in [
            r"(?i)\b\d+\s+Final\s+Test\b",
            r"(?i)\bFinal\s+Test\b",
            r"(?i)\b\d+\s+Design\b",
            r"(?i)\bDESIGN\s+CHANGE\b",
            r"(?i)\bPrepared\s+by\b",
        ]:
            p = find_pos(pat, s)
            if p is not None:
                ends.append(p)

        # stop before cable marking embossed (sering nyasar ke packing chunk)
        p = find_pos(r"(?i)\bCable\s+Marking\b", s)
        if p is not None:
            ends.append(p)

        # stop before IEC/JEMBO block kalau muncul setelah Gross weight
        # (heuristik biar tidak potong Ref Spec di awal dokumen)
        p_gross = find_pos(r"(?i)\bGross\s+weight\b", s)
        if p_gross is not None:
            p_iec = find_pos(r"(?i)\bIEC\s+\d+\b", p_gross)
            if p_iec is not None:
                ends.append(p_iec)
            p_jembo = find_pos(r"(?i)\bJEMBO\s+CABLE\b", p_gross)
            if p_jembo is not None:
                ends.append(p_jembo)

        e = min(ends) if ends else len(t)
        out = t[s:e].strip()

        # rapihin sedikit: hapus "Packing :" jika ada
        out = re.sub(r"(?i)\b\d+\s+Packing\b\s*[:\-]?\s*", "Packing - ", out)
        out = re.sub(r"(?i)\bPacking\b\s*[:\-]?\s*", "Packing - ", out, count=1)

        return out.strip()

    # default: return apa adanya
    return t

def sha256_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


def upsert_hit(engine, file_hash: str, file_path: str, section: str,
               score: float, page_no: int, chunk_no: int, chunk_id: int, chunk_text: str):

    fph = sha256_text(file_path)  # ✅ path hash

    with engine.begin() as conn:
        conn.execute(sql_text("""
            INSERT INTO tds_retrieval_hits
              (file_hash, file_path_hash, file_path, section, score, page_no, chunk_no, chunk_id, chunk_text)
            VALUES
              (:fh, :fph, :fp, :sec, :score, :page, :cno, :cid, :ctext)
            ON DUPLICATE KEY UPDATE
              file_hash=VALUES(file_hash),
              file_path=VALUES(file_path),
              score=VALUES(score),
              page_no=VALUES(page_no),
              chunk_no=VALUES(chunk_no),
              chunk_id=VALUES(chunk_id),
              chunk_text=VALUES(chunk_text),
              created_at=CURRENT_TIMESTAMP
        """), {
            "fh": file_hash,
            "fph": fph,
            "fp": file_path,
            "sec": section,
            "score": score,
            "page": page_no,
            "cno": chunk_no,
            "cid": chunk_id,
            "ctext": chunk_text
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", required=True)
    ap.add_argument("--mysql_server", required=True, help='mysql+pymysql://root:123456@192.168.10.40:3306')
    ap.add_argument("--db_name", required=True)
    ap.add_argument("--meta_db", default=r".\vector_store\meta\chunks.db")
    ap.add_argument("--index_file", default=r".\vector_store\index\faiss.index")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top_candidates", type=int, default=30)
    args = ap.parse_args()

    # load model + index
    model = SentenceTransformer(DEFAULT_MODEL)
    index = load_index(args.index_file)

    engine = create_engine(f"{args.mysql_server}/{args.db_name}", pool_pre_ping=True)
    ensure_table(engine)

    pdfs = []
    for root, _, files in os.walk(args.pdf_dir):
        for fn in files:
            if fn.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, fn))
    pdfs.sort()
    if args.limit:
        pdfs = pdfs[:args.limit]

    for i, pdf_path in enumerate(pdfs, start=1):
        fh = sha256_file(pdf_path)
        file_ids = get_file_chunk_ids(args.meta_db, fh)
        if not file_ids:
            print(f"[SKIP] not indexed: {os.path.basename(pdf_path)}")
            continue

        for sec, q in SECTION_QUERIES:
            qv = model.encode([q], normalize_embeddings=True)
            qv = np.asarray(qv, dtype=np.float32)
            qv = normalize_vecs(qv)

            scores, ids = index.search(qv, args.top_candidates)
            ids = ids[0].tolist()
            scores = scores[0].tolist()

            best = None
            for cid, score in zip(ids, scores):
                if cid == -1:
                    continue
                if int(cid) not in file_ids:
                    continue
                meta = get_chunk_meta(args.meta_db, int(cid))
                if not meta:
                    continue
                best = (float(score), meta)
                break

            if best:
                score, meta = best
                cleaned = clean_chunk_for_section(sec, meta["chunk_text"])
                upsert_hit(engine, fh, pdf_path, sec, score, meta["page_no"], meta["chunk_no"], meta["id"], cleaned)


        print(f"[OK] {i}/{len(pdfs)} saved -> {os.path.basename(pdf_path)}")

    print("DONE")

if __name__ == "__main__":
    main()
