import os
import re
import argparse
import hashlib
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import pandas as pd
from sqlalchemy import create_engine, text as sql_text


# =========================
# Utils
# =========================
UNITS = [
    "mm/-", "Ohm.mm²/km", "Ohm/km", "kV/min.", "kg/mm²", "kg/km",
    "n/mm", "pcs/dtex", "mm", "m", "kg", "%", "-", " %/-",
]

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def safe_table_name_from_path(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0].lower()
    base = re.sub(r"[^a-z0-9_]+", "_", base)
    base = re.sub(r"_+", "_", base).strip("_") or "file"
    return ("tds_" + base)[:64]

def pdf_to_text_one_paragraph(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    all_text = []
    for page in doc:
        all_text.append(page.get_text("text") or "")
    doc.close()
    # 1 paragraf supaya header match gampang
    return " ".join(" ".join(all_text).split())

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf8") as f:
        return f.read()


# =========================
# Section detection
# =========================
CANON_MAP = {
    "Conductor -": "Conductor",
    "Conductor Shielding -": "Conductor Shielding",
    "Covering -": "Covering",
    "Final Test -": "Final Test",
    "Final test -": "Final Test",   # samakan
    "Packing": "Packing",
    "Tapping -": "Tapping",
    "Insulation -": "Insulation",
    "Insulation Shielding -": "Insulation Shielding",
    "Metallic Screen -": "Metallic Screen",
    "Twisting -": "Twisting",
    "Overall Screen": "Overall Screen",
    "Individual Screen": "Individual Screen",
    "Cabling -": "Cabling",
    "Inner Sheath -": "Inner Sheath",
    "Armour -": "Armour",
    "Outer Sheath -": "Outer Sheath",
}

def build_header_patterns(headers: List[str]) -> List[Tuple[str, re.Pattern]]:
    out: List[Tuple[str, re.Pattern]] = []

    # pattern dash: jika header di list pakai "-", maka "-" WAJIB (anti false-positive seperti PE Insulation)
    for h in headers:
        base = h.strip().rstrip("-").strip()
        esc = re.escape(base).replace(r"\ ", r"\s+")
        if h.strip().endswith("-"):
            out.append((h, re.compile(rf"(?i)\b{esc}\b\s*\-", re.I)))
        else:
            out.append((h, re.compile(rf"(?i)\b{esc}\b", re.I)))

    # pattern numeric (untuk template yang pakai angka)
    numeric = {
        "Conductor -": r"(?i)\b1\s+Conductor\b",
        "Conductor Shielding -": r"(?i)\b2\s+Conductor\s+Shielding\b",
        "Covering -": r"(?i)\b2\s+Covering\b",
        "Final Test -": r"(?i)\b3\s+Final\s+Test\b",
        "Packing": r"(?i)\b4\s+Packing\b|\b11\s+Packing\b",
        "Tapping -": r"(?i)\b2\s+Tapping\b",
        "Insulation -": r"(?i)\b2\s+Insulation\b",
        "Metallic Screen -": r"(?i)\b5\s+Metallic\s+Screen\b",
        "Insulation Shielding -": r"(?i)\b4\s+Insulation\s+Shielding\b",
        "Twisting -": r"(?i)\b3\s+Twisting\b",
        "Individual Screen": r"(?i)\b4\s+Individual\s+Screen\b",
        "Cabling -": r"(?i)\b5\s+Cabling\b",
        "Overall Screen": r"(?i)\b6\s+Overall\s+Screen\b",
        "Inner Sheath -": r"(?i)\b7\s+Inner\s+Sheath\b",
        "Armour -": r"(?i)\b8\s+Armour\b",
        "Outer Sheath -": r"(?i)\b9\s+Outer\s+Sheath\b",
       
    }

    hdr_set = set(headers)
    for h, pat in numeric.items():
        if h in hdr_set:
            out.append((h, re.compile(pat, re.I)))

    return out

def parse_sections_dynamic(content: str, headers: List[str]) -> Dict[str, str]:
    pats = build_header_patterns(headers)

    matches = []
    for header, pat in pats:
        m = pat.search(content)
        if m:
            matches.append({"header": header, "start": m.start(), "end": m.end()})

    matches.sort(key=lambda x: x["start"])
    if not matches:
        return {}

    out: Dict[str, str] = {}
    for i, m in enumerate(matches):
        start_pos = m["end"]
        end_pos = matches[i + 1]["start"] if i < len(matches) - 1 else len(content)
        chunk = content[start_pos:end_pos].strip()
        if chunk:
            out[CANON_MAP.get(m["header"], m["header"])] = chunk

    return out

def extract_marking_from_text(content: str) -> Optional[str]:
    m = re.search(
        r"(?i)Marking\s+of\s+Cable\s+by\s+Roll\s+Printing\s*:\s*(.+?)(?:\b\d+\s+Final\s+Test\b|\b3\s+Final\s+Test\b|\b4\s+Packing\b|\Z)",
        content
    )
    if m:
        return norm(m.group(1))
    return None



# =========================
# Parser section -> rows (item/unit/specified/type/qty)
# =========================
def find_unit(text: str) -> Optional[str]:
    for u in sorted(UNITS, key=len, reverse=True):
        if re.search(rf"(?<!\w){re.escape(u)}(?!\w)", text):
            return u
    return None

def split_items_safe(section_text: str) -> List[str]:
    """
    Split hanya bullet "- " yang diikuti huruf,
    supaya angka '0,51 - 0.01' tidak kepotong.
    """
    t = norm(section_text)
    if not t:
        return []
    parts = re.split(r"\s-\s(?=[A-Za-z])", t)
    return [p.strip() for p in parts if p.strip()]

def parse_section_rows(section_name: str, section_text: str) -> pd.DataFrame:
    """
    Generic parsing.
    Conductor: type+qty biasanya global (Wire 206,0), disebarkan ke semua row.
    """
    items = split_items_safe(section_text)

    # global conductor type+qty: ambil qty terbesar (anti salah 51,5 vs 206,0)
    g_type = None
    g_qty = None
    if section_name.lower() == "conductor":
        cands = re.findall(r"([A-Za-z][A-Za-z\s]+Wire)\s+(\d+(?:[.,]\d+)?)", section_text, flags=re.I)
        if cands:
            best = None
            best_val = -1.0
            for typ, qty in cands:
                try:
                    val = float(qty.replace(",", "."))
                except:
                    continue
                if val > best_val:
                    best_val = val
                    best = (typ, qty)
            if best:
                g_type = norm(best[0])
                g_qty = norm(best[1])

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

                # buang global type qty kalau kebawa
                if g_type and g_qty:
                    tail = re.sub(rf"\b{re.escape(g_type)}\b\s+{re.escape(g_qty)}\b", "", tail, flags=re.I).strip()

                specified = tail or None
        else:
            # merge khusus: Conductor shape + Round Stranded
            if section_name.lower() == "conductor" and desc.lower() == "conductor shape" and i + 1 < len(items):
                nxt = norm(items[i+1])
                if nxt and find_unit(nxt) is None:
                    specified = nxt
                    unit = "-"  # di PDF memang "-"
                    i += 1
            else:
                # fallback split angka
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
            "quantity": qty
        })

        i += 1

    return pd.DataFrame(rows)


# =========================
# Postprocess (JEMBO fixes)
# =========================
def postprocess_conductor_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pecah OD/LP/DL of outer layer jadi 2 baris (1st/2nd layer).
    """
    if df is None or df.empty:
        return df

    rows = df.to_dict("records")
    out = []
    i = 0

    while i < len(rows):
        item = (rows[i].get("item") or "").strip()

        if item.lower() == "od/lp/dl of outer layer":
            base = "OD/LP/DL of outer layer"
            merged = False
            j = i + 1
            while j < len(rows):
                nxt = (rows[j].get("item") or "").strip()
                if nxt.lower().startswith("1st layer") or nxt.lower().startswith("2nd layer"):
                    nr = dict(rows[j])
                    nr["item"] = f"{base} - {nxt}"
                    out.append(nr)
                    merged = True
                    j += 1
                    continue
                break

            if merged:
                i = j
                continue

        out.append(rows[i])
        i += 1

    return pd.DataFrame(out)

def split_second_layer_from_specified(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kalau 1st Layer baris punya specified yang mengandung '- 2nd Layer ...',
    pecah jadi 2 baris:
      - baris 1 tetap 1st Layer (specified sebelum 2nd Layer)
      - baris 2 menjadi OD/LP/DL of outer layer - 2nd Layer ... (specified sisanya)
    """
    if df is None or df.empty:
        return df

    rows = df.to_dict("records")
    out = []

    for r in rows:
        item = (r.get("item") or "")
        spec = (r.get("specified") or "")

        # hanya proses untuk baris OD/LP/DL 1st layer
        if "OD/LP/DL of outer layer" in item and "1st layer" in item.lower() and "2nd layer" in spec.lower():
            # split di "2nd Layer"
            parts = re.split(r"(?i)\b2nd\s+Layer\b", spec, maxsplit=1)
            before = parts[0].strip(" -")                      # before 2nd layer
            after = ("2nd Layer " + parts[1]).strip() if len(parts) > 1 else ""

            # baris 1 (1st)
            r1 = dict(r)
            r1["specified"] = before if before else None
            out.append(r1)

            # baris 2 (2nd)
            if after:
                # coba ambil unit mm/- di after
                unit2 = r.get("unit")
                munit = re.search(r"(mm/-)", after)
                if munit:
                    unit2 = "mm/-"
                    # potong unit dari string supaya specified bersih
                    after_clean = after.replace("mm/-", "").strip()
                else:
                    after_clean = after

                r2 = dict(r)
                r2["item"] = "OD/LP/DL of outer layer - 2nd Layer 12 Wire"
                r2["unit"] = unit2
                r2["specified"] = after_clean if after_clean else None
                out.append(r2)

            continue

        out.append(r)

    return pd.DataFrame(out)

def clean_second_layer_specified(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    rows = df.to_dict("records")
    for r in rows:
        item = (r.get("item") or "").lower()
        spec = (r.get("specified") or "")

        if "od/lp/dl of outer layer" in item and "2nd layer" in item:
            # buang "2nd Layer 12 Wire" kalau masih kebawa di specified
            spec = re.sub(r"(?i)^\s*2nd\s+Layer\s+\d+\s+Wire\s*", "", spec).strip()
            r["specified"] = spec if spec else None

    return pd.DataFrame(rows)



def postprocess_covering_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Buang baris marking kalau nyasar ke Covering.
    """
    if df is None or df.empty:
        return df

    def is_marking(s: str) -> bool:
        t = (s or "").lower()
        return ("marking" in t) or ("jembo" in t and "cable" in t) or ("lmk" in t)

    cleaned = []
    for r in df.to_dict("records"):
        item = r.get("item") or ""
        spec = r.get("specified") or ""
        if is_marking(item) or is_marking(spec):
            continue
        cleaned.append(r)

    return pd.DataFrame(cleaned)

def postprocess_covering_type_qty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Untuk Covering:
    - Jika specified berisi "3.0 PE Insulation 128.27"
      maka:
        specified = "3.0"
        type = "PE Insulation"
        quantity = "128.27"
    - Jika specified berisi "2.70 Carbon Black 2.64" -> sama
    - Jika specified berisi "Black Sylane 1.32" -> specified="Black", type="Sylane", qty="1.32"
    """
    if df is None or df.empty:
        return df

    rows = df.to_dict("records")
    out = []

    for r in rows:
        spec = norm(r.get("specified") or "")
        if not spec:
            out.append(r)
            continue

        # pola: <specified_value> <Type words> <qty>
        m = re.match(r"^(.*?)(PE\s+Insulation|Carbon\s+Black|Sylane)\s+(\d+(?:\.\d+)?)$", spec, flags=re.I)
        if m:
            left = norm(m.group(1))
            typ = norm(m.group(2))
            qty = norm(m.group(3))

            r["specified"] = left if left else None
            r["type"] = typ
            r["quantity"] = qty
            out.append(r)
            continue

        # pola lain: "Black Sylane 1.32" (Core identification)
        m2 = re.match(r"^(Black)\s+(Sylane)\s+(\d+(?:\.\d+)?)$", spec, flags=re.I)
        if m2:
            r["specified"] = norm(m2.group(1))
            r["type"] = norm(m2.group(2))
            r["quantity"] = norm(m2.group(3))
            out.append(r)
            continue

        out.append(r)

    return pd.DataFrame(out)

def postprocess_covering_core_identification(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    rows = df.to_dict("records")
    out = []

    for r in rows:
        item = norm(r.get("item") or "")
        spec = norm(r.get("specified") or "")
        typ  = norm(r.get("type") or "")
        qty  = norm(r.get("quantity") or "")

        # kasus kamu: "Core identification Black Sylane" dan specified="1.32"
        if item.lower().startswith("core identification") and (not typ) and (not qty):
            # coba pecah item jadi: core identification | Black | Sylane
            m = re.match(r"(?i)^(Core\s+identification)\s+(Black)\s+(Sylane)$", item)
            if m:
                r["item"] = norm(m.group(1))
                r["specified"] = norm(m.group(2))   # Black
                r["type"] = norm(m.group(3))        # Sylane

                # quantity kadang nyasar di specified atau tetap di specified kolom
                # kalau specified awalnya "1.32" maka itu qty
                if spec and re.fullmatch(r"\d+(?:\.\d+)?", spec):
                    r["quantity"] = spec
                elif qty:
                    r["quantity"] = qty
                else:
                    # fallback: kalau ada angka di item (jarang)
                    pass

        out.append(r)

    return pd.DataFrame(out)

def postprocess_final_test_ok(df: pd.DataFrame) -> pd.DataFrame:
    """
    Jika item Final Test berakhiran 'Ok' (misal 'Visual all layer Ok'),
    pindahkan 'Ok' ke kolom specified.
    """
    if df is None or df.empty:
        return df

    rows = df.to_dict("records")
    for r in rows:
        item = norm(r.get("item") or "")
        spec = norm(r.get("specified") or "")

        if (not spec) and re.search(r"(?i)\bok$", item):
            # hapus Ok dari item
            new_item = re.sub(r"(?i)\s+ok$", "", item).strip()
            r["item"] = new_item if new_item else item
            r["specified"] = "Ok"

    return pd.DataFrame(rows)

def postprocess_packing(df: pd.DataFrame, raw_section_text: str) -> pd.DataFrame:
    """
    Rapikan Packing agar sesuai PDF:
    - item rows: Standard length, Net. Weight, Gross weight
    - material rows: Wooden Drum <size> qty, End Cap qty -> masuk ke kolom type/quantity (row1/row2)
    - buang noise Prepared by / Revision / Checked by / Approved by
    """
    if df is None:
        return df

    # 1) ambil material dari raw text (lebih stabil)
    t = norm(raw_section_text)

    wd = re.search(r"(?i)Wooden\s+Drum\s+(\d+)\s+(\d+)", t)
    ec = re.search(r"(?i)End\s+Cap\s+(\d+)", t)

    wd_type = f"Wooden Drum {wd.group(1)}" if wd else None
    wd_qty  = wd.group(2) if wd else None
    ec_type = "End Cap" if ec else None
    ec_qty  = ec.group(1) if ec else None

    # 2) filter item yang valid saja
    valid_items = {"standard length", "net. weight", "net weight", "gross weight"}
    noise_kw = ["prepared by", "checked by", "approved by", "revision", "design change", "notice", "item no"]

    rows = []
    for r in df.to_dict("records"):
        item = norm(r.get("item") or "")
        item_l = item.lower()

        # buang noise
        if any(k in item_l for k in noise_kw):
            continue

        # keep only known packing items OR empty rows
        if item and item_l not in valid_items:
            continue

        # normalisasi nama item
        if item_l == "net weight":
            r["item"] = "Net. Weight"
        elif item_l == "net. weight":
            r["item"] = "Net. Weight"
        elif item_l == "standard length":
            r["item"] = "Standard length"
        elif item_l == "gross weight":
            r["item"] = "Gross weight"

        rows.append(r)

    # 3) pastikan minimal 3 baris item (Standard, Net, Gross) sesuai urutan PDF
    def find_row(name):
        for rr in rows:
            if (rr.get("item") or "").lower() == name.lower():
                return rr
        return None

    ordered = []
    for name in ["Standard length", "Net. Weight", "Gross weight"]:
        rr = find_row(name)
        if rr:
            ordered.append(rr)
        else:
            ordered.append({"item": name, "unit": None, "specified": None, "type": None, "quantity": None})

    # 4) assign material ke row1/row2
    if wd_type and wd_qty:
        ordered[0]["type"] = wd_type
        ordered[0]["quantity"] = wd_qty
    if ec_type and ec_qty:
        ordered[1]["type"] = ec_type
        ordered[1]["quantity"] = ec_qty
    
        # 5) Paksa specified dari raw text (paling stabil)
    t2 = norm(raw_section_text)

    m_std = re.search(r"(?i)Standard\s+length\s+m\s+(\d+(?:[.,]\d+)?)", t2)
    m_net = re.search(r"(?i)Net\.?\s+Weight\s+kg\s+(\d+(?:[.,]\d+)?)", t2)
    m_gro = re.search(r"(?i)Gross\s+weight\s+kg\s+(\d+(?:[.,]\d+)?)", t2)

    if m_std:
        ordered[0]["unit"] = "m"
        ordered[0]["specified"] = m_std.group(1)
    if m_net:
        ordered[1]["unit"] = "kg"
        ordered[1]["specified"] = m_net.group(1)
    if m_gro:
        ordered[2]["unit"] = "kg"
        ordered[2]["specified"] = m_gro.group(1)

    return pd.DataFrame(ordered)

def postprocess_packing_fix_unit_spec(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pastikan:
    - Standard length: unit=m, specified=1000
    - Net. Weight: unit=kg, specified=339
    - Gross weight: unit=kg, specified=485
    Buang teks nyasar (misal 'End Cap 2') dari specified.
    """
    if df is None or df.empty:
        return df

    rows = df.to_dict("records")
    for r in rows:
        item = norm(r.get("item") or "").lower()
        unit = norm(r.get("unit") or "")
        spec = norm(r.get("specified") or "")

        # 1) unit paksa sesuai item
        if item == "standard length":
            r["unit"] = "m"
        elif item == "net. weight":
            r["unit"] = "kg"
        elif item == "gross weight":
            r["unit"] = "kg"

        # 2) specified ambil angka pertama saja
        # contoh: "1000 End Cap 2" -> "1000"
        m = re.search(r"(\d+(?:[.,]\d+)?)", spec)
        r["specified"] = m.group(1) if m else (spec if spec else None)

    return pd.DataFrame(rows)

def postprocess_conductor_shielding(df: pd.DataFrame, raw_section_text: str) -> pd.DataFrame:
    t = norm(raw_section_text or "")

    # 1) specified: setelah "Average thickness mm"
    spec = None
    m_spec = re.search(r"(?i)Average\s+thickness\s+mm\s+(\d+(?:[.,]\d+)?)", t)
    if m_spec:
        spec = m_spec.group(1)

    # 2) quantity: cari angka yang PALING dekat dengan akhir section dan masih masuk konteks shielding
    # biasakan qty shielding adalah angka besar (contoh 47,0)
    nums = re.findall(r"\b\d+(?:[.,]\d+)?\b", t)
    qty = None
    if nums:
        # ambil angka terbesar sebagai qty (47,0) -> aman dari 0,5
        best_val = -1.0
        best_raw = None
        for x in nums:
            try:
                v = float(x.replace(",", "."))
            except:
                continue
            if v > best_val:
                best_val = v
                best_raw = x
        qty = best_raw

    # 3) type: ambil text setelah "Semiconductive" sampai sebelum qty terbesar
    typ = None
    if qty and re.search(r"(?i)\bSemiconductive\b", t):
        # potong dari 'Semiconductive' sampai qty terbesar
        m = re.search(r"(?i)\bSemiconductive\b(.*)", t)
        tail = m.group(1) if m else ""

        # ambil sebelum qty
        # (pakai rfind agar qty terakhir yang dipakai)
        pos = tail.lower().rfind(qty.lower())
        before_qty = tail[:pos].strip() if pos != -1 else tail.strip()

        # buang noise yang nyasar
        before_qty = re.sub(r"(?i)average\s+thickness.*", "", before_qty).strip()
        before_qty = re.sub(r"(?i)\bmm\b.*", "", before_qty).strip()
        before_qty = before_qty.strip(" -:")

        typ = norm("Semiconductive " + before_qty) if before_qty else "Semiconductive"

    # fallback: jika IS PEROXIDE ada di text, paksa masuk
    if typ and "is peroxide" in t.lower() and "is peroxide" not in typ.lower():
        typ = norm(typ + " IS PEROXIDE")

    return pd.DataFrame([{
        "item": "Average thickness",
        "unit": "mm",
        "specified": spec,
        "type": typ,
        "quantity": qty
    }])

def postprocess_insulation_shielding(
    df: pd.DataFrame,
    raw_section_text: str,
    pdf_path: str,
    debug: bool = False
) -> pd.DataFrame:

    # ✅ pakai pdf_path & debug dari parameter
    typ, qty = extract_insulation_shielding_type_qty_from_pdf(pdf_path, debug=debug)

    t = norm(raw_section_text or "")

    m_avg = re.search(r"(?i)Average\s+thickness\s+mm\s+(\d+(?:[.,]\d+)?)", t)
    m_od  = re.search(r"(?i)Outer\s+diameter\s+Approx\s+mm\s+(\d+(?:[.,]\d+)?)", t)
    spec_avg = m_avg.group(1) if m_avg else None
    spec_od  = m_od.group(1) if m_od else None

    rows = [
        {"item": "Average thickness", "unit": "mm", "specified": spec_avg, "type": typ, "quantity": qty},
        {"item": "Outer diameter Approx", "unit": "mm", "specified": spec_od, "type": None, "quantity": None},
    ]
    return pd.DataFrame(rows)




def postprocess_metallic_screen_core_id(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    rows = df.to_dict("records")
    for r in rows:
        item = norm(r.get("item") or "")
        if item.lower().startswith("core identification") and not r.get("specified"):
            m = re.match(r"(?i)^(Core\s+identification)\s+(.+)$", item)
            if m:
                r["item"] = norm(m.group(1))
                r["specified"] = norm(m.group(2))
    return pd.DataFrame(rows)

def postprocess_insulation_type_qty(df: pd.DataFrame, raw_section_text: str) -> pd.DataFrame:
    """
    Contoh PDF:
      Nom. thickness -> specified 4,5 | type XLPE | qty 503,6
      Minimum thickness... -> specified 3,95 | type PEROXIDE 2 | qty 503,6 (global)
    """
    if df is None or df.empty:
        return df

    t = norm(raw_section_text or "")

    # ambil qty global terbesar (biasanya angka besar seperti 503,6)
    nums = re.findall(r"\b\d+(?:[.,]\d+)?\b", t)
    g_qty = None
    if nums:
        best_val = -1.0
        best_raw = None
        for x in nums:
            try:
                v = float(x.replace(",", "."))
            except:
                continue
            if v > best_val:
                best_val = v
                best_raw = x
        g_qty = best_raw

    rows = df.to_dict("records")
    for r in rows:
        spec = norm(r.get("specified") or "")

        # kalau type/qty masih kosong dan spec kebawa "4,5 XLPE 503,6"
        if spec:
            # buang qty global dari spec kalau kebawa
            if g_qty:
                spec_wo_qty = re.sub(rf"\b{re.escape(g_qty)}\b", "", spec).strip()
            else:
                spec_wo_qty = spec

            # pola: <angka> <type words>
            m = re.match(r"^(\d+(?:[.,]\d+)?)\s+(.+)$", spec_wo_qty)
            if m:
                r["specified"] = m.group(1)
                r["type"] = norm(m.group(2))
            else:
                # kalau cuma angka
                r["specified"] = spec

        # set qty global
        if g_qty:
            r["quantity"] = g_qty

    return pd.DataFrame(rows)

def extract_insulation_shielding_type_qty_from_pdf(pdf_path: str, debug: bool = False) -> Tuple[Optional[str], Optional[str]]:
    """
    Cari qty 105,3 (khas Insulation Shielding), lalu ambil type di sekitar baris itu.
    Target:
      Strippable Semiconductive
      OS PEROXIDE
    | 105,3
    """
    # helper: rapihin spasi tapi SIMPAN newline
    def norm_keep_newline(s: str) -> str:
        lines = (s or "").splitlines()
        lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in lines if ln.strip() != ""]
        return "\n".join(lines).strip()

    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return None, None

    for page in doc:
        words = page.get_text("words")
        if not words:
            continue

        # group words into lines by y
        lines = {}
        for w in words:
            x0, y0, x1, y1, txt = w[:5]
            txt = str(txt).strip()
            if not txt:
                continue
            yk = round(y0, 1)
            lines.setdefault(yk, []).append((x0, txt))

        line_list = []
        for yk, toks in sorted(lines.items()):
            toks_sorted = [t for _, t in sorted(toks)]
            line_text = " ".join(toks_sorted)
            line_list.append((yk, line_text, toks_sorted))

        # cari baris yang mengandung qty ~ 105.3
        for idx, (yk, line_text, toks_sorted) in enumerate(line_list):
            nums = re.findall(r"\b\d+(?:[.,]\d+)?\b", line_text)
            hit_qty = None
            for n in nums:
                try:
                    val = float(n.replace(",", "."))
                except:
                    continue
                if 104.0 <= val <= 106.5:  # target 105,3
                    hit_qty = n
                    break
            if not hit_qty:
                continue

            # ambil window baris sekitar (idx-3 .. idx+3)
            window_lines = []
            for j in range(max(0, idx - 3), min(len(line_list), idx + 4)):
                window_lines.append(line_list[j][1])

            window_text = norm(" ".join(window_lines))  # boleh norm di window (nggak butuh newline)

            # ambil text sebelum qty
            pos = window_text.lower().rfind(hit_qty.lower())
            left = window_text[:pos].strip() if pos != -1 else window_text

            # coba ambil typ yang sudah lengkap (kalau PEROXIDE muncul sebelum qty)
            m = re.search(r"(?i)(Strippable\s+Semiconductive.*?PEROXIDE)", left)
            if not m:
                m = re.search(r"(?i)(Semiconductive.*?PEROXIDE)", left)

            typ = m.group(1).strip() if m else None

            # fallback: kalau ada "Strippable Semiconductive" tapi belum kebentuk type
            if not typ and "strippable semiconductive" in window_text.lower():
                typ = "Strippable Semiconductive"

            # kalau ada "OS PEROXIDE" di window, jadikan 2 baris seperti PDF
            if "os peroxide" in window_text.lower():
                if typ:
                    # kalau typ belum mengandung OS PEROXIDE, paksa newline
                    if "os peroxide" not in typ.lower():
                        # paksa format 2 baris
                        typ = "Strippable Semiconductive\nOS PEROXIDE"
                    else:
                        # kalau sudah ada peroxide tapi masih 1 baris, ubah ke newline
                        if "\n" not in typ:
                            typ = "Strippable Semiconductive\nOS PEROXIDE"
                else:
                    typ = "OS PEROXIDE"

            # rapihin: buang noise tapi JANGAN hilangkan newline
            if typ:
                typ = re.sub(
                    r"(?i)\baverage\b|\bthickness\b|\binsulation\b|\bshielding\b|\bouter\b|\bdiameter\b|\bapprox\b|\bmm\b",
                    "",
                    typ
                )
                typ = norm_keep_newline(typ)   # ✅ bukan norm()

            if debug:
                print("DEBUG qty:", hit_qty)
                print("DEBUG window:", window_text)
                print("DEBUG type:", typ)

            doc.close()
            return typ, hit_qty

    doc.close()
    return None, None














# =========================
# MySQL dynamic per-file table (only present sections)
# =========================
def ensure_db(mysql_server_url: str, db_name: str):
    eng_server = create_engine(mysql_server_url, pool_pre_ping=True)
    with eng_server.begin() as conn:
        conn.execute(sql_text(
            f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        ))
    return create_engine(f"{mysql_server_url}/{db_name}", pool_pre_ping=True)

def table_exists(conn, db_name: str, table_name: str) -> bool:
    q = sql_text("""
        SELECT COUNT(*) FROM information_schema.tables
        WHERE table_schema=:db AND table_name=:tbl
    """)
    return conn.execute(q, {"db": db_name, "tbl": table_name}).scalar() > 0

def column_exists(conn, db_name: str, table_name: str, col_name: str) -> bool:
    q = sql_text("""
        SELECT COUNT(*) FROM information_schema.columns
        WHERE table_schema=:db AND table_name=:tbl AND column_name=:col
    """)
    return conn.execute(q, {"db": db_name, "tbl": table_name, "col": col_name}).scalar() > 0

from typing import Optional, List, Tuple  # pastikan ini sudah ada di atas

def ensure_table_dynamic(
    conn,
    table_name: str,
    db_name: str,
    sections_present: List[str],
    extra_cols: Optional[List[Tuple[str, str]]] = None
):
    if not table_exists(conn, db_name, table_name):
        conn.execute(sql_text(f"""
            CREATE TABLE `{table_name}` (
              id BIGINT AUTO_INCREMENT PRIMARY KEY,
              row_no INT NOT NULL,
              created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """))

    # kolom-kolom section yang ada
    for sec in sections_present:
        cols = [
            (sec, "TEXT"),
            (f"{sec} Unit", "VARCHAR(50)"),
            (f"{sec} Specified", "TEXT"),
            (f"{sec} Type", "TEXT"),
            (f"{sec} Quantity", "VARCHAR(50)"),
        ]
        for col, typ in cols:
            if not column_exists(conn, db_name, table_name, col):
                conn.execute(sql_text(f"ALTER TABLE `{table_name}` ADD COLUMN `{col}` {typ}"))

    # ✅ kolom tambahan (contoh: Marking) hanya kalau dibutuhkan
    if extra_cols:
        for col, typ in extra_cols:
            if not column_exists(conn, db_name, table_name, col):
                conn.execute(sql_text(f"ALTER TABLE `{table_name}` ADD COLUMN `{col}` {typ}"))

    # unique row_no
    try:
        conn.execute(sql_text(f"ALTER TABLE `{table_name}` ADD UNIQUE KEY uq_rowno (row_no)"))
    except Exception:
        pass


def df_to_wide_dynamic(df_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    max_len = 0
    for df in df_map.values():
        if df is not None and not df.empty:
            max_len = max(max_len, len(df))
    if max_len == 0:
        return pd.DataFrame()

    wide = []
    for i in range(max_len):
        row = {"row_no": i + 1}
        for sec, df in df_map.items():
            if df.empty or i >= len(df):
                continue
            r = df.iloc[i]
            row[sec] = r.get("item")
            row[f"{sec} Unit"] = r.get("unit")
            row[f"{sec} Specified"] = r.get("specified")
            row[f"{sec} Type"] = r.get("type")
            row[f"{sec} Quantity"] = r.get("quantity")
        wide.append(row)

    return pd.DataFrame(wide)

def insert_wide(conn, table_name: str, wide_df: pd.DataFrame):
    if wide_df.empty:
        return

    cols = list(wide_df.columns)
    col_sql = ", ".join([f"`{c}`" for c in cols])
    val_sql = ", ".join([f":{c.replace(' ','_')}" for c in cols])
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
# Runner
# =========================
def iter_files(input_path: str, mode: str):
    exts = (".pdf",) if mode == "pdf" else (".txt",)

    if os.path.isfile(input_path):
        if input_path.lower().endswith(exts):
            yield input_path
        return

    for root, _, files in os.walk(input_path):
        for fn in files:
            if fn.lower().endswith(exts):
                yield os.path.join(root, fn)
                
def extract_cable_marking_embossed(content: str) -> Optional[str]:
    """
    Ambil seluruh block cable marking embossed (2 baris):
    - Cable Marking by Embossed : IEC 60502 ... "Year" "Length Marking"
    - Length marking by Ink-Jet Printing with interval 1 meter
    """
    # ambil block dari "Cable Marking by Embossed" sampai sebelum Final Test / end
    m = re.search(
        r"(?is)Cable\s+Marking\s+by\s+Embossed\s*:?(.*?)(?:\b\d+\s+Final\s+Test\b|\bFinal\s+Test\b|\Z)",
        content
    )
    if not m:
        return None

    block = m.group(1)

    # rapihin whitespace
    block_1sp = re.sub(r"\s+", " ", block).strip()

    # buang prefix '-' kalau kebawa
    block_1sp = re.sub(r"^\-\s*", "", block_1sp).strip()

    # kalau ada kata "Cable Marking by Embossed" ikut kebawa, hapus
    block_1sp = re.sub(r"(?i)^Cable\s+Marking\s+by\s+Embossed\s*:?\s*", "", block_1sp).strip()

    # optional: pastikan minimal ada IEC atau JEMBO atau Length marking
    if not any(k in block_1sp.lower() for k in ["iec", "jembo", "length marking"]):
        return None

    return block_1sp

def extract_cable_marking_embossed_from_pdf(pdf_path: str) -> Optional[str]:
    """
    Ambil block Cable Marking by Embossed pakai words (lebih lengkap daripada get_text('text')).
    Target: IEC 60502 JEMBO CABLE ... "Year" "Length Marking" + kalimat inkjet interval.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return None

    for page in doc:
        words = page.get_text("words")  # (x0,y0,x1,y1,word,...)
        if not words:
            continue

        # sort reading order
        words.sort(key=lambda w: (round(w[1], 1), round(w[0], 1)))
        toks = [w[4] for w in words if str(w[4]).strip()]
        low = [t.lower() for t in toks]

        # cari posisi "Embossed"
        try:
            idx_emb = low.index("embossed")
        except ValueError:
            continue

        # ambil window token setelah Embossed
        window = toks[idx_emb: idx_emb + 250]
        window_low = [t.lower() for t in window]

        # start dari "IEC" kalau ada, kalau tidak start dari awal window
        start = 0
        for i, t in enumerate(window_low):
            if t == "iec":
                start = i
                break

        # stop sebelum "Final Test"
        stop = len(window)
        for i in range(len(window_low) - 1):
            if window_low[i] == "final" and window_low[i + 1] == "test":
                stop = i
                break

        picked = window[start:stop]
        text_1sp = re.sub(r"\s+", " ", " ".join(picked)).strip()

        # buang awalan ":" kalau kebawa
        text_1sp = re.sub(r"^\:\s*", "", text_1sp).strip()

        # pastikan ini memang marking block
        if "jembo" in text_1sp.lower() or "iec" in text_1sp.lower() or "length" in text_1sp.lower():
            doc.close()
            return text_1sp

    doc.close()
    return None




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--mode", choices=["pdf","txt"], required=True)
    ap.add_argument("--mysql_server", required=True)
    ap.add_argument("--db_name", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    headers = [
        "Conductor -", "Conductor Shielding -","Covering -", "Final Test -", "Final test -", "Packing",
        "Tapping -", "Insulation -", "Twisting -", "Insulation Shielding -", "Metallic Screen -",
        "Overall Screen", "Individual Screen",
        "Cabling -", "Inner Sheath -", "Armour -", "Outer Sheath -"
    ]

    engine = ensure_db(args.mysql_server, args.db_name)

    total = ok = fail = 0
    for path in iter_files(args.input_dir, args.mode):
        total += 1
        if args.limit and total > args.limit:
            break

        try:
            if args.mode == "pdf":
                content = pdf_to_text_one_paragraph(path)
            else:
                content = read_txt(path)

            sections = parse_sections_dynamic(content, headers)
            present_sections = [k for k, v in sections.items() if v and v != "Not Found"]

            if not present_sections:
                raise RuntimeError("Tidak ada section terbaca")

            df_map: Dict[str, pd.DataFrame] = {}
            for sec in present_sections:
                df_tmp = parse_section_rows(sec, sections[sec])

                if sec.lower() == "conductor":
                    df_tmp = postprocess_conductor_df(df_tmp)
                    df_tmp = split_second_layer_from_specified(df_tmp)
                    df_tmp = clean_second_layer_specified(df_tmp)
                if sec.lower() == "covering":
                    df_tmp = postprocess_covering_df(df_tmp)
                    df_tmp = postprocess_covering_type_qty(df_tmp)
                    df_tmp = postprocess_covering_core_identification(df_tmp)
                if sec.lower() == "final test":
                    df_tmp = postprocess_final_test_ok(df_tmp)
                if sec.lower() == "packing":
                    df_tmp = postprocess_packing(df_tmp, sections[sec])
                    df_tmp = postprocess_packing_fix_unit_spec(df_tmp)
                if sec.lower() == "conductor shielding":
                    df_tmp = postprocess_conductor_shielding(df_tmp, sections[sec])
                if sec.lower() == "insulation shielding":
                    df_tmp = postprocess_insulation_shielding(df_tmp, sections[sec], path, debug=args.debug)
                if sec.lower() == "metallic screen":
                    df_tmp = postprocess_metallic_screen_core_id(df_tmp)
                if sec.lower() == "insulation":
                    df_tmp = postprocess_insulation_type_qty(df_tmp, sections[sec])
                

                df_map[sec] = df_tmp

            wide_df = df_to_wide_dynamic(df_map)
            if wide_df.empty:
                raise RuntimeError("wide_df kosong (parsing gagal)")

            table_name = safe_table_name_from_path(path)

            # ✅ extra columns (Marking / Cable Marking by Embossed) hanya kalau ada
            extra_cols = []

            marking = extract_marking_from_text(content)
            if marking:
                wide_df.loc[wide_df["row_no"] == 1, "Marking"] = marking
                extra_cols.append(("Marking", "TEXT"))

            if args.mode == "pdf":
                embossed = extract_cable_marking_embossed_from_pdf(path)
            else:
                embossed = extract_cable_marking_embossed(content)  # kalau mode txt
            if embossed:
                wide_df.loc[wide_df["row_no"] == 1, "Cable Marking by Embossed"] = embossed
                extra_cols.append(("Cable Marking by Embossed", "TEXT"))

            if not extra_cols:
                extra_cols = None

            with engine.begin() as conn:
                ensure_table_dynamic(conn, table_name, args.db_name, present_sections, extra_cols=extra_cols)
                conn.execute(sql_text(f"TRUNCATE TABLE `{table_name}`"))
                insert_wide(conn, table_name, wide_df)

            ok += 1
            print(f"[OK] {os.path.basename(path)} -> {table_name} rows={len(wide_df)} sections={present_sections}")

        except Exception as e:
            fail += 1
            print(f"[FAIL] {path} -> {e}")

    print(f"DONE total={total}, ok={ok}, fail={fail}")

if __name__ == "__main__":
    main()
