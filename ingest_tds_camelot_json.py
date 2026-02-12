import os
import re
import hashlib
import argparse
from typing import Dict, List, Optional, Tuple

from sqlalchemy import create_engine, text
import camelot
import pdfplumber


# ----------------------------
# Utils
# ----------------------------
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(s) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


def df_contains(df, keyword: str) -> bool:
    kw = keyword.lower()
    for r in range(df.shape[0]):
        for c in range(df.shape[1]):
            if kw in norm(df.iat[r, c]).lower():
                return True
    return False


def dump_tables_debug(pdf_path: str, dfs: List, out_dir: str = "debug"):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    for i, df in enumerate(dfs, start=1):
        df.to_csv(os.path.join(out_dir, f"{base}_table_{i}.csv"), index=False, header=False)


def split_type_qty_last_number(s: str) -> Tuple[Optional[str], Optional[str]]:
    """
    "Carbon Black 2.64" -> ("Carbon Black", "2.64")
    "Wooden Drum 120 1" -> ("Wooden Drum 120", "1")
    """
    s = norm(s)
    if not s:
        return None, None

    m = re.match(r"^(.*?)([-+]?\d+(?:\.\d+)?)\s*$", s)
    if m:
        left = norm(m.group(1))
        num = norm(m.group(2))
        if left:
            return left, num
    return s, None


def clean_desc(desc: str) -> str:
    d = norm(desc)
    d = re.sub(r"^\-\s*", "", d)
    return d


def is_marking_text(s: str) -> bool:
    t = (s or "").lower()
    return ("jembo" in t and "cable" in t) or ("lmk" in t) or ("<>" in t)


def extract_marking_from_page_text(pdf_path: str) -> Optional[str]:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            t = (pdf.pages[0].extract_text() or "")
        m = re.search(r"Marking of Cable by Roll Printing\s*:\s*(.+)", t, re.I)
        if m:
            return norm(m.group(1))
    except Exception:
        pass
    return None


def extract_packing_from_text(pdf_path: str) -> List[Tuple[str, str]]:
    """
    Fallback paling pasti untuk Packing dari text PDF (PDF text-based).
    Target:
      Wooden Drum 120 | 1
      End Cap | 2
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text_all = "\n".join([(p.extract_text() or "") for p in pdf.pages])
    except Exception:
        return []

    t = re.sub(r"[ \t]+", " ", text_all)

    # ambil blok dekat "4 Packing"
    m = re.search(r"4\s+Packing(.+?)(?:\n\s*\d+\s+\w+|\Z)", t, re.I | re.S)
    block = m.group(1) if m else t

    wd = re.search(r"Wooden\s+Drum\s+(\d+)\s+(\d+)", block, re.I)  # 120 1
    ec = re.search(r"End\s+Cap\s+(\d+)", block, re.I)             # 2

    out = []
    if wd:
        out.append((f"Wooden Drum {wd.group(1)}", wd.group(2)))
    if ec:
        out.append(("End Cap", ec.group(1)))

    # dedup + Wooden Drum first
    dedup = []
    for p in out:
        if p not in dedup:
            dedup.append(p)

    ordered = [p for p in dedup if "wooden drum" in p[0].lower()]
    for p in dedup:
        if p not in ordered:
            ordered.append(p)

    return ordered


def find_value_next_to_label_scan_right(df, label: str) -> Optional[str]:
    lab = label.lower()
    for r in range(df.shape[0]):
        for c in range(df.shape[1]):
            cell = norm(df.iat[r, c]).lower()
            if lab == cell or lab in cell:
                for cc in range(c + 1, df.shape[1]):
                    v = norm(df.iat[r, cc])
                    if v:
                        return v
                for rr in range(r + 1, min(r + 4, df.shape[0])):
                    for cc in range(c + 1, df.shape[1]):
                        v = norm(df.iat[rr, cc])
                        if v:
                            return v
    return None


def extract_ref_specs_from_header_table(df) -> List[str]:
    specs = []
    for r in range(df.shape[0]):
        row_join = " ".join(norm(df.iat[r, c]) for c in range(df.shape[1])).lower()
        if "ref. spec" in row_join or "ref spec" in row_join:
            for c in range(df.shape[1]):
                if "ref" in norm(df.iat[r, c]).lower():
                    for cc in range(c + 1, df.shape[1]):
                        v = norm(df.iat[r, cc])
                        if v:
                            specs.append(v)
                            break
                    rr = r + 1
                    while rr < df.shape[0]:
                        left = norm(df.iat[rr, c])
                        if left:
                            break
                        vv = None
                        for cc in range(c + 1, df.shape[1]):
                            v = norm(df.iat[rr, cc])
                            if v:
                                vv = v
                                break
                        if vv:
                            specs.append(vv)
                        rr += 1

                    out = []
                    for s in specs:
                        s = norm(s)
                        if not s:
                            continue
                        if "raw material" in s.lower():
                            continue
                        if "spln" in s.lower():
                            if s not in out:
                                out.append(s)
                    return out
    return []


# ----------------------------
# Auto-create DB & tables
# (Quantity -> Packing Quantity)
# ----------------------------
def ensure_database_and_tables(mysql_server_url: str, db_name: str):
    if not re.match(r"^[A-Za-z0-9_]+$", db_name):
        raise ValueError("db_name hanya boleh huruf/angka/underscore. Contoh: tds_db")

    eng_server = create_engine(mysql_server_url, pool_pre_ping=True)
    with eng_server.begin() as conn:
        conn.execute(text(
            f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        ))

    eng_db = create_engine(f"{mysql_server_url}/{db_name}", pool_pre_ping=True)
    with eng_db.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS tds_documents (
              id BIGINT AUTO_INCREMENT PRIMARY KEY,
              file_path VARCHAR(1024) NOT NULL,
              file_hash CHAR(64) NOT NULL,
              parser_name VARCHAR(64) NOT NULL,
              parsed_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
              UNIQUE KEY uq_filehash (file_hash)
            )
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS tds_rows (
              id BIGINT AUTO_INCREMENT PRIMARY KEY,
              document_id BIGINT NOT NULL,
              row_no INT NOT NULL,

              `Type` VARCHAR(100),
              `Size` VARCHAR(100),
              `Rate voltage` VARCHAR(100),
              `Ref. Spec.` VARCHAR(200),
              `Reff. Doc.` VARCHAR(100),

              `Conductor` TEXT,
              `Conductor Unit` VARCHAR(50),
              `Conductor Specified` VARCHAR(200),
              `Conductor Type` VARCHAR(200),
              `Conductor Quantity kg/km` VARCHAR(50),

              `Covering` TEXT,
              `Covering Unit` VARCHAR(50),
              `Covering Specified` VARCHAR(200),
              `Covering Type` VARCHAR(200),
              `Covering Quantity kg/km` VARCHAR(50),

              `Marking of Cable by Roll Printing` TEXT,

              `Final Test` TEXT,
              `Final Test Unit` VARCHAR(50),
              `Final Test Specified` VARCHAR(200),
              `Final Test Type` VARCHAR(200),
              `Final Test Quantity kg/km` VARCHAR(50),

              `Packing` TEXT,
              `Packing Unit` VARCHAR(50),
              `Packing Specified` VARCHAR(200),
              `Packing Type` VARCHAR(200),
              `Packing Quantity` VARCHAR(50),

              created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

              KEY idx_doc_row (document_id, row_no),
              CONSTRAINT fk_rows_doc FOREIGN KEY (document_id) REFERENCES tds_documents(id)
            )
        """))

        # auto rename old column if exists
        try:
            conn.execute(text("""
                ALTER TABLE tds_rows
                CHANGE COLUMN `Quantity` `Packing Quantity` VARCHAR(50)
            """))
        except Exception:
            pass

    return eng_db


# ----------------------------
# Parsing helpers
# ----------------------------
def df_score_for_main_table(df) -> int:
    score = 0
    if df_contains(df, "DESCRIPTION"):
        score += 5
    if df_contains(df, "UNIT"):
        score += 3
    if df_contains(df, "SPECIFIED"):
        score += 5
    if df_contains(df, "Conductor"):
        score += 2
    if df_contains(df, "Covering"):
        score += 2
    if df_contains(df, "Final Test"):
        score += 2
    if df_contains(df, "Packing"):
        score += 2

    signature_terms = ["prepared by", "checked by", "approved by", "engineer", "supervisor"]
    for r in range(df.shape[0]):
        for c in range(df.shape[1]):
            if any(t in norm(df.iat[r, c]).lower() for t in signature_terms):
                score -= 10
    return score


def choose_main_table(dfs: List) -> Optional[object]:
    if not dfs:
        return None
    scored = [(df_score_for_main_table(df), df) for df in dfs]
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_df = scored[0]

    if df_contains(best_df, "Conductor") or df_contains(best_df, "Covering") or df_contains(best_df, "Final Test") or df_contains(best_df, "Packing"):
        return best_df

    if best_score < 5:
        return None
    return best_df


def parse_header_from_tables(dfs: List) -> Dict:
    header = {
        "Type": None,
        "Size": None,
        "Rate voltage": None,
        "Reff. Doc.": None,
        "Ref. Spec.": [],
    }

    for df in dfs:
        if df_contains(df, "Type") and df_contains(df, "Size"):
            header["Type"] = header["Type"] or find_value_next_to_label_scan_right(df, "Type")
            header["Size"] = header["Size"] or find_value_next_to_label_scan_right(df, "Size")
            header["Rate voltage"] = header["Rate voltage"] or find_value_next_to_label_scan_right(df, "Rate voltage")

            rs = extract_ref_specs_from_header_table(df)
            if rs:
                header["Ref. Spec."] = rs

        if df_contains(df, "Reff. Doc"):
            header["Reff. Doc."] = header["Reff. Doc."] or find_value_next_to_label_scan_right(df, "Reff. Doc.")

    return header


def find_col_idx(df, keywords: List[str], search_rows=(0, 1, 2)) -> Optional[int]:
    ks = [k.lower() for k in keywords]
    for rr in search_rows:
        if rr >= df.shape[0]:
            continue
        for c in range(df.shape[1]):
            cell = norm(df.iat[rr, c]).lower()
            if any(k in cell for k in ks):
                return c
    return None


def parse_sections_from_main_table(df) -> Tuple[Dict[str, List[Dict]], Optional[str]]:
    sections = {"Conductor": [], "Covering": [], "Final Test": [], "Packing": []}
    marking = None

    desc_c = find_col_idx(df, ["description"]) or 0
    unit_c = find_col_idx(df, ["unit"]) or 1

    type_c = find_col_idx(df, ["raw material", "type", "material"])
    qty_c = find_col_idx(df, ["quantity"])

    if type_c is None or qty_c is None:
        if df.shape[1] >= 2:
            type_c = df.shape[1] - 2
            qty_c = df.shape[1] - 1

    current = None
    start_r = 1
    if df.shape[0] > 1 and "description" in norm(df.iat[1, desc_c]).lower():
        start_r = 2

    noise_keywords = ["prepared by", "checked by", "approved by", "engineer", "supervisor", "signature"]

    for r in range(start_r, df.shape[0]):
        row_all = " ".join(norm(df.iat[r, c]) for c in range(df.shape[1]))
        if any(k in row_all.lower() for k in noise_keywords):
            continue

        desc_raw = norm(df.iat[r, desc_c])
        desc = clean_desc(desc_raw)
        unit = norm(df.iat[r, unit_c]) if unit_c is not None else ""

        msec = re.match(r"^\s*\d+\s+(Conductor|Covering|Final Test|Packing)\b", desc_raw, re.I)
        if msec:
            current = msec.group(1).title()
            continue

        if "marking of cable by roll printing" in desc_raw.lower():
            marking = norm(row_all)
            continue

        if current not in sections:
            continue

        specified_parts = []
        left = min(max(unit_c + 1, 0), df.shape[1] - 1)
        right = max(min(type_c - 1, df.shape[1] - 1), left)
        for cc in range(left, right + 1):
            v = norm(df.iat[r, cc])
            if v and v.lower() not in ["-", "—"]:
                specified_parts.append(v)
        specified = norm(" ".join(specified_parts))

        if specified and is_marking_text(specified):
            if not marking:
                marking = specified
            specified = ""

        raw_type_cell = norm(df.iat[r, type_c]) if type_c is not None else ""
        raw_qty_cell = norm(df.iat[r, qty_c]) if qty_c is not None else ""

        sections[current].append({
            "desc": desc if desc else None,
            "unit": unit if unit else None,
            "specified": specified if specified else None,
            "raw_type_cell": raw_type_cell,
            "raw_qty_cell": raw_qty_cell,
            "type": None,
            "qty": None,
        })

    return sections, marking


# ----------------------------
# Postprocess (Packing fallback to text)
# ----------------------------
def repair_conductor_desc_merge_layers(cond_items: List[Dict]) -> List[Dict]:
    out = []
    base = None
    for it in cond_items:
        d = norm(it.get("desc"))
        if d.lower() == "od/lp/dl of outer layer":
            base = "OD/LP/DL of outer layer"
            continue

        if base and (d.lower().startswith("1st layer") or d.lower().startswith("2nd layer")):
            it = dict(it)
            it["desc"] = f"{base} {d}"
            out.append(it)
            continue

        out.append(it)
    return out


def apply_type_qty_from_raw_cell(it: Dict) -> Dict:
    it = dict(it)

    rt = norm(it.get("raw_type_cell"))
    rq = norm(it.get("raw_qty_cell"))

    candidates = [rt, rq]
    best_t = None
    best_q = None

    for s in candidates:
        t, q = split_type_qty_last_number(s)
        if t and q:
            best_t, best_q = t, q
            break

    if best_t is None:
        if rt:
            best_t = rt
        elif rq:
            best_t = rq

    it["type"] = best_t if best_t else None
    it["qty"] = best_q if best_q else None
    return it


def fix_spec_suffix_SZ(cond_items: List[Dict]) -> List[Dict]:
    out = []
    for it in cond_items:
        it = dict(it)
        spec = norm(it.get("specified"))
        rt = norm(it.get("raw_type_cell"))
        if rt in ("S", "Z") and spec and spec.endswith("/"):
            it["specified"] = norm(spec + " " + rt)
            it["raw_type_cell"] = ""
            it["raw_qty_cell"] = ""
        out.append(it)
    return out


def normalize_marking(marking: Optional[str]) -> Optional[str]:
    if not marking:
        return None
    m = re.sub(r"(?i).*Marking of Cable by Roll Printing\s*:\s*", "", marking).strip()
    return norm(m)


def postprocess_sections(header: Dict, sections: Dict[str, List[Dict]], marking: Optional[str], pdf_path: str) -> Tuple[Dict, Dict, Optional[str]]:
    marking = normalize_marking(marking)

    # --- Conductor ---
    cond = sections.get("Conductor", [])
    cond = fix_spec_suffix_SZ(cond)
    cond = repair_conductor_desc_merge_layers(cond)

    conductor_type = None
    conductor_qty = None
    for it in cond:
        it2 = apply_type_qty_from_raw_cell(it)
        t = norm(it2.get("type"))
        q = norm(it2.get("qty"))
        if t and q and ("wire" in t.lower()):
            conductor_type = t
            conductor_qty = q
            break

    fixed_cond = []
    for it in cond:
        it = dict(it)
        it["desc"] = clean_desc(it.get("desc") or "")
        it["type"] = conductor_type if conductor_type else None
        it["qty"] = conductor_qty if conductor_qty else None
        fixed_cond.append(it)
    sections["Conductor"] = fixed_cond

    # --- Covering ---
    cov = sections.get("Covering", [])
    cov_materials: List[Tuple[str, str]] = []
    for it in cov:
        it2 = apply_type_qty_from_raw_cell(it)
        t = norm(it2.get("type"))
        q = norm(it2.get("qty"))
        if t and q:
            cov_materials.append((t, q))
    cov_m2 = []
    for p in cov_materials:
        if p not in cov_m2:
            cov_m2.append(p)

    fixed_cov = []
    for it in cov:
        it = dict(it)
        it["desc"] = clean_desc(it.get("desc") or "")
        spec = norm(it.get("specified"))
        if spec and is_marking_text(spec):
            if not marking:
                marking = spec
            it["specified"] = None
        it["type"] = None
        it["qty"] = None
        fixed_cov.append(it)

    for i, (t, q) in enumerate(cov_m2):
        if i < len(fixed_cov):
            fixed_cov[i]["type"] = t
            fixed_cov[i]["qty"] = q
    sections["Covering"] = fixed_cov

    # --- Packing ---
    pk = sections.get("Packing", [])
    pk_materials: List[Tuple[str, str]] = []
    for it in pk:
        it2 = apply_type_qty_from_raw_cell(it)
        t = norm(it2.get("type"))
        q = norm(it2.get("qty"))
        if t and q:
            pk_materials.append((t, q))
    pk_m2 = []
    for p in pk_materials:
        if p not in pk_m2:
            pk_m2.append(p)

    # ✅ fallback: kalau Wooden Drum tidak ada, ambil dari text (pasti ada)
    if not any("wooden drum" in t.lower() for t, _ in pk_m2):
        pk_m2 = extract_packing_from_text(pdf_path)

    fixed_pk = []
    for it in pk:
        it = dict(it)
        it["desc"] = clean_desc(it.get("desc") or "")
        it["type"] = None
        it["qty"] = None
        fixed_pk.append(it)

    for i, (t, q) in enumerate(pk_m2):
        if i < len(fixed_pk):
            fixed_pk[i]["type"] = t
            fixed_pk[i]["qty"] = q
    sections["Packing"] = fixed_pk

    # --- Final Test (biarkan) ---
    ft = sections.get("Final Test", [])
    fixed_ft = []
    for it in ft:
        it = dict(it)
        it["desc"] = clean_desc(it.get("desc") or "")
        it["type"] = None
        it["qty"] = None
        fixed_ft.append(it)
    sections["Final Test"] = fixed_ft

    rs = header.get("Ref. Spec.", []) or []
    rs = [x for x in rs if x and ("spln" in x.lower()) and ("raw material" not in x.lower())]
    header["Ref. Spec."] = rs

    return header, sections, marking


# ----------------------------
# Main parse pipeline
# ----------------------------
def parse_tds(pdf_path: str, debug: bool = False):
    dfs = []

    try:
        tbs = camelot.read_pdf(pdf_path, pages="1", flavor="lattice")
        dfs += [t.df for t in tbs]
    except Exception:
        pass

    try:
        tbs2 = camelot.read_pdf(pdf_path, pages="1", flavor="stream")
        dfs += [t.df for t in tbs2]
    except Exception:
        pass

    if debug:
        dump_tables_debug(pdf_path, dfs)
        print(f"[DEBUG] dumped {len(dfs)} tables into ./debug")

    header = parse_header_from_tables(dfs)
    main_df = choose_main_table(dfs)
    if main_df is None:
        marking = extract_marking_from_page_text(pdf_path)
        return header, {}, marking

    sections, marking_tbl = parse_sections_from_main_table(main_df)

    marking_txt = extract_marking_from_page_text(pdf_path)
    marking = marking_txt or marking_tbl

    header, sections, marking = postprocess_sections(header, sections, marking, pdf_path)
    return header, sections, marking


def build_excel_like_rows(header: Dict, sections: Dict[str, List[Dict]], marking: Optional[str]) -> List[Dict]:
    ref_specs = header.get("Ref. Spec.", []) or []

    conductor = sections.get("Conductor", []) or []
    covering = sections.get("Covering", []) or []
    final_test = sections.get("Final Test", []) or []
    packing = sections.get("Packing", []) or []

    N = max(len(ref_specs), len(conductor), len(covering), len(final_test), len(packing), 1)

    rows = []
    for i in range(N):
        c = conductor[i] if i < len(conductor) else {}
        cv = covering[i] if i < len(covering) else {}
        ft = final_test[i] if i < len(final_test) else {}
        pk = packing[i] if i < len(packing) else {}

        ref_spec = ref_specs[i] if i < len(ref_specs) else None
        if ref_spec and "raw material" in ref_spec.lower():
            ref_spec = None

        rows.append({
            "row_no": i + 1,
            "Type": header.get("Type"),
            "Size": header.get("Size"),
            "Rate voltage": header.get("Rate voltage"),
            "Ref. Spec.": ref_spec,
            "Reff. Doc.": header.get("Reff. Doc."),

            "Conductor": c.get("desc"),
            "Conductor Unit": c.get("unit"),
            "Conductor Specified": c.get("specified"),
            "Conductor Type": c.get("type"),
            "Conductor Quantity kg/km": c.get("qty"),

            "Covering": cv.get("desc"),
            "Covering Unit": cv.get("unit"),
            "Covering Specified": cv.get("specified"),
            "Covering Type": cv.get("type"),
            "Covering Quantity kg/km": cv.get("qty"),

            "Marking of Cable by Roll Printing": marking if i == 0 else None,

            "Final Test": ft.get("desc"),
            "Final Test Unit": ft.get("unit"),
            "Final Test Specified": ft.get("specified"),
            "Final Test Type": ft.get("type"),
            "Final Test Quantity kg/km": ft.get("qty"),

            "Packing": pk.get("desc"),
            "Packing Unit": pk.get("unit"),
            "Packing Specified": pk.get("specified"),
            "Packing Type": pk.get("type"),
            "Packing Quantity": pk.get("qty"),
        })

    return rows


# ----------------------------
# DB insert (Packing Quantity)
# ----------------------------
def insert_document_and_rows(engine, pdf_path: str, file_hash: str, parser_name: str, rows: List[Dict]) -> int:
    with engine.begin() as conn:
        existing = conn.execute(
            text("SELECT id FROM tds_documents WHERE file_hash=:h"),
            {"h": file_hash}
        ).fetchone()
        if existing:
            return existing[0]

        res = conn.execute(
            text("""
                INSERT INTO tds_documents (file_path, file_hash, parser_name)
                VALUES (:p, :h, :pn)
            """),
            {"p": pdf_path, "h": file_hash, "pn": parser_name}
        )
        doc_id = res.lastrowid

        for r in rows:
            conn.execute(
                text("""
                    INSERT INTO tds_rows (
                      document_id, row_no,
                      `Type`, `Size`, `Rate voltage`, `Ref. Spec.`, `Reff. Doc.`,
                      `Conductor`, `Conductor Unit`, `Conductor Specified`, `Conductor Type`, `Conductor Quantity kg/km`,
                      `Covering`, `Covering Unit`, `Covering Specified`, `Covering Type`, `Covering Quantity kg/km`,
                      `Marking of Cable by Roll Printing`,
                      `Final Test`, `Final Test Unit`, `Final Test Specified`, `Final Test Type`, `Final Test Quantity kg/km`,
                      `Packing`, `Packing Unit`, `Packing Specified`, `Packing Type`, `Packing Quantity`
                    ) VALUES (
                      :document_id, :row_no,
                      :Type, :Size, :Rate_voltage, :Ref_Spec, :Reff_Doc,
                      :Conductor, :Conductor_Unit, :Conductor_Specified, :Conductor_Type, :Conductor_Quantity,
                      :Covering, :Covering_Unit, :Covering_Specified, :Covering_Type, :Covering_Quantity,
                      :Marking,
                      :Final_Test, :Final_Test_Unit, :Final_Test_Specified, :Final_Test_Type, :Final_Test_Quantity,
                      :Packing, :Packing_Unit, :Packing_Specified, :Packing_Type, :Packing_Quantity
                    )
                """),
                {
                    "document_id": doc_id,
                    "row_no": r.get("row_no"),

                    "Type": r.get("Type"),
                    "Size": r.get("Size"),
                    "Rate_voltage": r.get("Rate voltage"),
                    "Ref_Spec": r.get("Ref. Spec."),
                    "Reff_Doc": r.get("Reff. Doc."),

                    "Conductor": r.get("Conductor"),
                    "Conductor_Unit": r.get("Conductor Unit"),
                    "Conductor_Specified": r.get("Conductor Specified"),
                    "Conductor_Type": r.get("Conductor Type"),
                    "Conductor_Quantity": r.get("Conductor Quantity kg/km"),

                    "Covering": r.get("Covering"),
                    "Covering_Unit": r.get("Covering Unit"),
                    "Covering_Specified": r.get("Covering Specified"),
                    "Covering_Type": r.get("Covering Type"),
                    "Covering_Quantity": r.get("Covering Quantity kg/km"),

                    "Marking": r.get("Marking of Cable by Roll Printing"),

                    "Final_Test": r.get("Final Test"),
                    "Final_Test_Unit": r.get("Final Test Unit"),
                    "Final_Test_Specified": r.get("Final Test Specified"),
                    "Final_Test_Type": r.get("Final Test Type"),
                    "Final_Test_Quantity": r.get("Final Test Quantity kg/km"),

                    "Packing": r.get("Packing"),
                    "Packing_Unit": r.get("Packing Unit"),
                    "Packing_Specified": r.get("Packing Specified"),
                    "Packing_Type": r.get("Packing Type"),
                    "Packing_Quantity": r.get("Packing Quantity"),
                }
            )

        return doc_id


# ----------------------------
# Runner
# ----------------------------
def iter_pdfs(pdf_dir: str):
    for root, _, files in os.walk(pdf_dir):
        for fn in files:
            if fn.lower().endswith(".pdf"):
                yield os.path.join(root, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", required=True)
    ap.add_argument("--mysql_server", required=True)
    ap.add_argument("--db_name", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    engine = ensure_database_and_tables(args.mysql_server, args.db_name)

    count = ok = fail = 0
    for pdf_path in iter_pdfs(args.pdf_dir):
        count += 1
        if args.limit and count > args.limit:
            break

        try:
            fh = sha256_file(pdf_path)
            header, sections, marking = parse_tds(pdf_path, debug=args.debug)

            if not sections:
                print(f"[FAIL] {pdf_path} - tabel utama tidak kebaca")
                fail += 1
                continue

            rows = build_excel_like_rows(header, sections, marking)
            insert_document_and_rows(engine, pdf_path, fh, "tds_v4_excel_exact", rows)
            ok += 1
            print(f"[OK] {os.path.basename(pdf_path)} -> rows={len(rows)}")

        except Exception as e:
            print(f"[ERR] {pdf_path} -> {e}")
            fail += 1

    print(f"DONE total={count}, ok={ok}, fail={fail}")


if __name__ == "__main__":
    main()
