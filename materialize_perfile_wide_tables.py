import os
import re
import argparse
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import create_engine, text as sql_text

# =========================
# CONFIG
# =========================
SECTIONS = [
    "Conductor",
    "Covering",
    "Final Test",
    "Packing",
    "Conductor Shielding",
    "Insulation",
    "Insulation Shielding",
    "Metallic Screen",
    "Cabling",
    "Inner Sheath",
    "Armour",
    "Outer Sheath",
]

UNITS = [
    "Ohm.mm²/km", "kV/min.", "Ohm/km", "kg/mm²", "M.Ohm.km",
    "kg/km", "n/mm", "pcs/dtex",
    "mm/-", "%/-",
    "mm", "m", "kg", "%", "-", "pC",
]

# =========================
# utils
# =========================
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def safe_table_name(file_path: str) -> str:
    base = os.path.splitext(os.path.basename(file_path))[0].lower()
    base = re.sub(r"[^a-z0-9_]+", "_", base)
    base = re.sub(r"_+", "_", base).strip("_") or "file"
    return ("tds_" + base)[:64]

def find_unit(text: str) -> Optional[str]:
    for u in sorted(UNITS, key=len, reverse=True):
        if re.search(rf"(?<!\w){re.escape(u)}(?!\w)", text):
            return u
    return None

def split_items_safe(section_text: str) -> List[str]:
    """
    Split pakai " - " hanya jika setelahnya huruf (anti pecah angka '0,51 - 0.01').
    """
    t = norm(section_text)
    if not t:
        return []
    parts = re.split(r"\s-\s(?=[A-Za-z])", t)
    return [p.strip() for p in parts if p.strip()]

def pick_global_type_qty(text: str, patterns: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Cari material type + qty yg besar (untuk conductor/insulation shielding dll).
    """
    t = text or ""
    cands = []
    for pat in patterns:
        for m in re.finditer(pat, t, flags=re.I):
            typ = norm(m.group(1))
            qty = norm(m.group(2))
            cands.append((typ, qty))
    if not cands:
        return None, None

    best = None
    best_val = -1.0
    for typ, qty in cands:
        try:
            v = float(qty.replace(",", "."))
        except:
            continue
        if v > best_val:
            best_val = v
            best = (typ, qty)
    return best if best else (None, None)

def parse_section_rows(section: str, chunk_text: str) -> pd.DataFrame:
    """
    Parse chunk_text (1 section) -> rows item/unit/specified/type/quantity
    Heuristik sederhana, mirip yang kamu pakai kemarin.
    """
    items = split_items_safe(chunk_text)

    g_type = None
    g_qty = None

    # global type/qty: conductor wire / insulation qty besar / shielding qty besar / metallic screen tape qty besar
    sec_l = section.lower()
    if sec_l == "conductor":
        g_type, g_qty = pick_global_type_qty(chunk_text, [r"([A-Za-z][A-Za-z\s]+Wire)\s+(\d+(?:[.,]\d+)?)"])
    elif sec_l == "insulation":
        # contoh: XLPE 503,6
        g_type, g_qty = pick_global_type_qty(chunk_text, [r"(XLPE(?:\s+\w+)*)\s+(\d+(?:[.,]\d+)?)"])
    elif "shielding" in sec_l:
        # contoh: Semiconductive 47,0 / Strippable Semiconductive 105,3
        g_type, g_qty = pick_global_type_qty(chunk_text, [r"((?:Strippable\s+)?Semiconductive(?:\s+\w+)*)\s+(\d+(?:[.,]\d+)?)"])
    elif sec_l in ("metallic screen", "cabling", "armour", "outer sheath", "inner sheath"):
        # ambil material umum + qty besar (PVC ... 599,1 / Galv. Steel Tape 747,4 / dll)
        g_type, g_qty = pick_global_type_qty(chunk_text, [r"([A-Za-z][A-Za-z\s\.\(\)]+)\s+(\d+(?:[.,]\d+)?)"])

    rows = []
    i = 0
    while i < len(items):
        raw = norm(items[i])
        unit = find_unit(raw)
        desc = raw
        specified = None
        typ = g_type
        qty = g_qty

        if unit:
            m = re.search(rf"(?<!\w){re.escape(unit)}(?!\w)", raw)
            if m:
                desc = raw[:m.start()].strip()
                tail = raw[m.end():].strip()

                # jika global type+qty kebawa di tail, hapus
                if typ and qty:
                    tail = re.sub(rf"\b{re.escape(typ)}\b\s+{re.escape(qty)}\b", "", tail, flags=re.I).strip()

                specified = tail or None
        else:
            # kasus seperti "Conductor shape - Compacted Circular Stranded"
            if sec_l == "conductor" and desc.lower() == "conductor shape" and i + 1 < len(items):
                nxt = norm(items[i + 1])
                if nxt and find_unit(nxt) is None:
                    specified = nxt
                    unit = "-"
                    i += 1
            else:
                # split angka pertama
                mnum = re.search(r"\b\d", raw)
                if mnum:
                    left = raw[:mnum.start()].strip()
                    right = raw[mnum.start():].strip()
                    if left:
                        desc = left
                        specified = right

        desc = desc.strip(" :-")

        rows.append({
            "item": desc or None,
            "unit": unit,
            "specified": specified,
            "type": typ,
            "quantity": qty,
        })
        i += 1

    return pd.DataFrame(rows)

def df_map_to_wide(df_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    max_len = 0
    for df in df_map.values():
        if df is not None and not df.empty:
            max_len = max(max_len, len(df))
    if max_len == 0:
        return pd.DataFrame()

    out_rows = []
    for i in range(max_len):
        row = {"row_no": i + 1}
        for sec, df in df_map.items():
            if df is None or df.empty or i >= len(df):
                continue
            r = df.iloc[i]
            row[sec] = r.get("item")
            row[f"{sec} Unit"] = r.get("unit")
            row[f"{sec} Specified"] = r.get("specified")
            row[f"{sec} Type"] = r.get("type")
            row[f"{sec} Quantity"] = r.get("quantity")
        out_rows.append(row)
    return pd.DataFrame(out_rows)

# =========================
# MySQL table per file
# =========================
def column_exists(conn, table_name: str, col_name: str) -> bool:
    q = sql_text("""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_schema = DATABASE()
          AND table_name = :tbl
          AND column_name = :col
    """)
    return conn.execute(q, {"tbl": table_name, "col": col_name}).scalar() > 0

def ensure_perfile_table(conn, table_name: str, present_sections: List[str]):
    # base table
    conn.execute(sql_text(f"""
        CREATE TABLE IF NOT EXISTS `{table_name}` (
          id BIGINT AUTO_INCREMENT PRIMARY KEY,
          row_no INT NOT NULL,
          created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
          UNIQUE KEY uq_rowno (row_no)
        )
    """))

    # add columns only for sections that exist in this file
    for sec in present_sections:
        cols = [
            (sec, "TEXT"),
            (f"{sec} Unit", "VARCHAR(50)"),
            (f"{sec} Specified", "TEXT"),
            (f"{sec} Type", "TEXT"),
            (f"{sec} Quantity", "VARCHAR(50)"),
        ]
        for col, typ in cols:
            if not column_exists(conn, table_name, col):
                conn.execute(sql_text(f"ALTER TABLE `{table_name}` ADD COLUMN `{col}` {typ}"))



def insert_wide_rows(conn, table_name: str, wide_df: pd.DataFrame):
    if wide_df.empty:
        return
    cols = list(wide_df.columns)  # includes row_no and dynamic cols
    col_sql = ", ".join([f"`{c}`" for c in cols])
    val_sql = ", ".join([f":{c.replace(' ', '_')}" for c in cols])
    stmt = sql_text(f"INSERT INTO `{table_name}` ({col_sql}) VALUES ({val_sql})")

    for _, r in wide_df.iterrows():
        params = {}
        for c in cols:
            key = c.replace(" ", "_")
            v = r.get(c)
            if pd.isna(v):
                v = None
            params[key] = v
        conn.execute(stmt, params)

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mysql_server", required=True, help='mysql+pymysql://root:123456@192.168.10.40:3306')
    ap.add_argument("--db_name", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0=all files")
    args = ap.parse_args()

    engine = create_engine(f"{args.mysql_server}/{args.db_name}", pool_pre_ping=True)

    # ambil semua file unik dari hits
    with engine.begin() as conn:
        rows = conn.execute(sql_text("""
            SELECT DISTINCT file_path
            FROM tds_retrieval_hits
            ORDER BY file_path
        """)).fetchall()

    file_paths = [r[0] for r in rows]
    if args.limit:
        file_paths = file_paths[:args.limit]

    total = len(file_paths)
    for idx, file_path in enumerate(file_paths, start=1):
        with engine.begin() as conn:
            hit_rows = conn.execute(sql_text("""
                SELECT section, chunk_text
                FROM tds_retrieval_hits
                WHERE file_path = :fp
                  AND chunk_text IS NOT NULL
                  AND chunk_text <> ''
            """), {"fp": file_path}).fetchall()

        # ✅ ini present sections beneran dari file tsb
        sec_text = {sec: (txt or "") for sec, txt in hit_rows}
        present_sections = sorted(sec_text.keys())

        df_map = {}
        for sec in present_sections:
            df_map[sec] = parse_section_rows(sec, sec_text[sec])


        wide_df = df_map_to_wide(df_map)
        if wide_df.empty:
            print(f"[SKIP] {os.path.basename(file_path)} wide_df empty")
            continue

        table_name = safe_table_name(file_path)

        with engine.begin() as conn:
            # ensure table + columns
            ensure_perfile_table(conn, table_name, present_sections)
            # refresh
            conn.execute(sql_text(f"TRUNCATE TABLE `{table_name}`"))
            insert_wide_rows(conn, table_name, wide_df)

        print(f"[OK] {idx}/{total} -> {table_name} rows={len(wide_df)}")

    print("DONE")

if __name__ == "__main__":
    main()
