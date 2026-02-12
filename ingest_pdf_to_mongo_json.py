import os
import re
import hashlib
import argparse
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import fitz  # PyMuPDF
from pymongo import MongoClient, ASCENDING
import base64
import os, re, hashlib, argparse


PARSER_VERSION = "mongo-json-v4-v2"

UNITS = [
    "Ohm.mm²/km", "Ohm/km", "M.Ohm.km", "kV/min.", "kg/mm²", "kg/km",
    "n/mm", "pcs/dtex", "mm/-", "%/-", "mm", "m", "kg", "%", "-", "n/denier", "pC","m²", "m2", "m^2"
]


# ----------------------------
# Utils
# ----------------------------

def cleanup_cabling_ts(rows):
    import re

    def norm(s):
        return re.sub(r"\s+", " ", (s or "")).strip()

    # normalize dulu
    rws = []
    for r in rows or []:
        rws.append({
            "description": norm(r.get("description")),
            "unit": norm(r.get("unit")),
            "specified": norm(r.get("specified")),
        })

    out = []
    i = 0
    while i < len(rws):
        cur = dict(rws[i])
        nxt = rws[i + 1] if i + 1 < len(rws) else None

        desc = cur["description"]
        unit = cur["unit"]
        spec = cur["specified"]

        # ------------------------------------------------------------
        # A) FIX: "Filler n/- 3 / (...)" nyasar ke description
        #    ambil pattern: <name> n/- <rest>
        # ------------------------------------------------------------
        m = re.search(r"(?i)^(.*?)(?:\s+)?(n\s*/\s*-)\s+(.*)$", desc)
        if m:
            left = norm(m.group(1))
            rest = norm(m.group(3))

            # kalau ini kasus filler/centre filler, unit harusnya n/-
            # (parser sering bikin unit jadi % atau -)
            if unit in {"", "-", "%", "%/-"}:
                unit = "n/-"

            desc = left
            # rest itu harusnya masuk specified (digabung dengan specified existing)
            spec = norm((rest + " " + spec).strip())

        # ------------------------------------------------------------
        # B) FIX: token "( 60" nyasar di akhir baris (biasanya milik baris berikutnya "tape dimension")
        #    kalau desc berakhir "( 60" atau "(60", pindahkan ke specified baris berikutnya (kalau relevan)
        # ------------------------------------------------------------
        tail = re.search(r"(?i)\(\s*\d+\s*$", desc)
        if tail and nxt:
            nxt_desc_l = (nxt.get("description") or "").lower()
            if ("tape" in nxt_desc_l) or ("dimension" in nxt_desc_l):
                moved = tail.group(0)  # contoh: "( 60"
                desc = norm(re.sub(r"(?i)\(\s*\d+\s*$", "", desc).strip())

                # prepend ke specified next
                nxt_spec = norm(nxt.get("specified"))
                rws[i + 1]["specified"] = norm((moved + " " + nxt_spec).strip())

        # ------------------------------------------------------------
        # C) FIX: specified cuma ")" (orphan)
        #    - kalau prev punya kurung belum ketutup -> tempel ke prev
        #    - else kalau next punya "(" -> tempel ke next
        # ------------------------------------------------------------
        if spec == ")":
            if out:
                prev_spec = norm(out[-1].get("specified"))
                if prev_spec.count("(") > prev_spec.count(")"):
                    out[-1]["specified"] = norm(prev_spec + " )")
                    i += 1
                    continue

            if nxt:
                nxt_spec = norm(nxt.get("specified"))
                if "(" in nxt_spec and nxt_spec.count("(") > nxt_spec.count(")"):
                    rws[i + 1]["specified"] = norm(nxt_spec + " )")
                    i += 1
                    continue

            # fallback: buang saja kalau benar-benar orphan
            spec = "-"

        # ------------------------------------------------------------
        # D) Clean kecil: unit "n/-" kadang nyangkut jadi "n/" / unit kosong
        # ------------------------------------------------------------
        if unit == "n/":
            unit = "n/-"

        out.append({
            "description": desc or "-",
            "unit": unit or "-",
            "specified": spec or "-",
        })
        i += 1

    return out




def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def norm(s: str) -> str:
    return norm_space(s)

def find_unit(text: str) -> Optional[str]:
    t = text or ""
    for u in sorted(UNITS, key=len, reverse=True):
        if re.search(rf"(?<!\w){re.escape(u)}(?!\w)", t):
            return u
    return None

def pdf_to_text_one_paragraph(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    parts = []
    for page in doc:
        parts.append(page.get_text("text") or "")
    doc.close()
    return " ".join(" ".join(parts).split())

def extract_section_text_words(pdf_path: str, start_pat: re.Pattern, stop_pat: re.Pattern) -> str:
    """
    Ambil teks section dari PDF pakai get_text("words") berdasarkan anchor start/stop.
    Output dibuat mirip bullet-list: " - ... - ...", biar bisa diparse split_items_safe().
    """
    def ns(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return ""

    try:
        if doc.page_count == 0:
            return ""

        # biasanya tabel ada di page 0
        page = doc.load_page(0)
        words = page.get_text("words") or []
        if not words:
            return ""

        # group by y -> lines
        lines = {}
        for x0, y0, x1, y1, w, *_ in words:
            ww = ns(str(w))
            if not ww:
                continue
            yk = round(float(y0), 1)
            lines.setdefault(yk, []).append((float(x0), ww))

        ykeys = sorted(lines.keys())

        line_texts = []
        for yk in ykeys:
            toks = [t[1] for t in sorted(lines[yk], key=lambda z: z[0])]
            txt = ns(" ".join(toks))
            if txt:
                line_texts.append((yk, txt))

        # cari start_y
        start_y = None
        for yk, txt in line_texts:
            if start_pat.search(txt):
                start_y = yk
                break
        if start_y is None:
            return ""

        # cari stop_y setelah start
        stop_y = None
        for yk, txt in line_texts:
            if yk <= start_y:
                continue
            if stop_pat.search(txt):
                stop_y = yk
                break

        # ambil baris di antara start..stop
        body = []
        for yk, txt in line_texts:
            if yk <= start_y:
                continue
            if stop_y is not None and yk >= stop_y:
                break

            # skip baris kosong / noise parah
            if not txt:
                continue

            body.append(txt)

        if not body:
            return ""

        # bentuk jadi " - item - item - item" supaya split_items_safe jalan
        return " - " + " - ".join(body)

    finally:
        doc.close()


def iter_pdf_paths(input_path: str):
    if os.path.isfile(input_path):
        if input_path.lower().endswith(".pdf"):
            yield input_path
        return
    for root, _, files in os.walk(input_path):
        for fn in files:
            if fn.lower().endswith(".pdf"):
                yield os.path.join(root, fn)


# ----------------------------
# Date helpers (V2: ISO)
# ----------------------------
MONTH_MAP = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05", "JUN": "06",
    "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12"
}

def dd_mmm_yy_to_iso(s: str) -> Optional[str]:
    """
    "17-Apr-02" -> "2002-04-17"
    "23-Nov-10" -> "2010-11-23"
    """
    s = (s or "").strip()
    m = re.fullmatch(r"(\d{1,2})[-–—]([A-Za-z]{3})[-–—](\d{2})", s)
    if not m:
        return None
    dd = m.group(1).zfill(2)
    mon = MONTH_MAP.get(m.group(2).upper())
    if not mon:
        return None
    yy = int(m.group(3))
    yyyy = 2000 + yy if yy <= 50 else 1900 + yy
    return f"{yyyy}-{mon}-{dd}"

def normalize_iso_or_keep(s: str) -> str:
    """
    Kalau bisa convert dd-mmm-yy -> ISO, pakai ISO.
    Kalau sudah ISO / format lain, balikin string asli yang sudah dirapihin.
    """
    s = norm(s)
    if not s:
        return "-"
    iso = dd_mmm_yy_to_iso(s)
    return iso if iso else s


# ----------------------------
# Marking (V2 object)
# ----------------------------
def extract_marking_triplet_from_pdf(pdf_path: str) -> Dict[str, str]:
    """
    Ambil 2-3 baris marking dari PDF:
      baris 1 -> type_marking
      baris 2 -> kalimat_marking
      baris 3 -> length_marking (HANYA jika mengandung kata 'Length')
    """
    def ns(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    def clean_type_line(s: str) -> str:
        s = ns(s)
        s = re.sub(r"^\-\s*", "", s)
        s = re.sub(r"\s*:\s*$", "", s)
        return s

    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return {}

    try:
        for page in doc:
            words = page.get_text("words") or []
            if not words:
                continue

            lines = {}
            for x0, y0, x1, y1, w, *_ in words:
                w = str(w).strip()
                if not w:
                    continue
                yk = round(float(y0), 1)
                lines.setdefault(yk, []).append((float(x0), w))

            line_texts = []
            for yk in sorted(lines.keys()):
                toks = sorted(lines[yk], key=lambda t: t[0])
                text = ns(" ".join(t for _, t in toks))
                if text:
                    line_texts.append(text)

            start_idx = None
            for i, lt in enumerate(line_texts):
                low = lt.lower()
                if "marking of cable by roll printing" in low:
                    start_idx = i
                    break
                if "cable marking by embossed" in low:
                    start_idx = i
                    break
                if "cable marking by ink jet printing" in low:
                    start_idx = i
                    break
                if "cable marking by ink-jet printing" in low:
                    start_idx = i
                    break

            if start_idx is None:
                continue

            type_line = line_texts[start_idx]
            type_marking = clean_type_line(type_line)

            kalimat_marking = None
            if ":" in type_line:
                tail = ns(type_line.split(":", 1)[1])
                if tail:
                    kalimat_marking = tail

            def next_lines(from_i: int, max_take: int = 6) -> List[str]:
                out = []
                for j in range(from_i + 1, min(len(line_texts), from_i + 1 + max_take)):
                    t = ns(line_texts[j])
                    if t:
                        out.append(t)
                return out

            after = next_lines(start_idx, max_take=6)

            if not kalimat_marking and after:
                kalimat_marking = after[0]

            length_marking = None
            if after:
                for cand in after[1:] if kalimat_marking else after:
                    if re.search(r"(?i)\blength\b", cand):
                        length_marking = cand
                        break

            # proteksi
            if kalimat_marking and ("od/lp/dl" in kalimat_marking.lower()) and not re.search(r"(?i)\blength\b", kalimat_marking):
                kalimat_marking = None
            if length_marking and ("od/lp/dl" in length_marking.lower()) and not re.search(r"(?i)\blength\b", length_marking):
                length_marking = None

            out = {}
            if type_marking:
                out["tipe_marking"] = type_marking
            if kalimat_marking:
                out["kalimat_marking"] = kalimat_marking
            if length_marking:
                out["length_marking"] = length_marking

            if out:
                return out

        return {}
    finally:
        doc.close()


# -----------------------
# Metadata extractor (V2)
# -----------------------
def extract_metadata(pdf_path: str) -> dict:
    """
    Output V2:
      {
        "ref_doc": "...",
        "type": "...",
        "size": "...",
        "rate_voltage": "...",
        "ref_spec": ["...", "..."],
        "number": "...",
        "date": {
          "established": "YYYY-MM-DD",
          "revised": "YYYY-MM-DD",
          "revision_number": 5
        }
      }
    """

    def ns(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    def join_tokens(tokens: List[str]) -> str:
        s = ns(" ".join(tokens))
        s = re.sub(r"\s*/\s*", "/", s)
        s = re.sub(r"\s*x\s*", " x ", s, flags=re.I)
        return ns(s)

    def clean_size(val: str) -> str:
        v = ns(val)
        v = re.split(r"(?i)\bTECHNICAL\s+DATA\s+SHEET\b", v)[0].strip()
        v = re.split(r"(?i)\bTECHNICAL\s+DATA\b", v)[0].strip()
        v = re.split(r"(?i)\bTDS\b", v)[0].strip()
        v = re.sub(r"\s*x\s*", " x ", v, flags=re.I)
        v = re.sub(r"\s+", " ", v).strip()
        return v

    LABELS_LEFT = ["Reff. Doc.", "Ref. Doc.", "Type", "Size", "Rate voltage", "Ref. Spec."]
    LABELS_RIGHT = ["Number", "Established", "Revised", "Revision number"]

    KEYMAP = {
        "Reff. Doc.": "ref_doc",
        "Ref. Doc.": "ref_doc",
        "Type": "type",
        "Size": "size",
        "Rate voltage": "rate_voltage",
        "Ref. Spec.": "ref_spec",
        "Number": "number",
        "Established": "established",
        "Revised": "revised",
        "Revision number": "revision_number",
    }

    doc = fitz.open(pdf_path)
    try:
        if doc.page_count == 0:
            return {
                "ref_doc": "-",
                "type": "-",
                "size": "-",
                "rate_voltage": "-",
                "ref_spec": [],
                "number": "-",
                "date": {"established": "-", "revised": "-", "revision_number": 0}
            }

        page = doc.load_page(0)
        page_w = float(page.rect.width)
        words = page.get_text("words") or []
    finally:
        doc.close()

    if not words:
        return {
            "ref_doc": "-",
            "type": "-",
            "size": "-",
            "rate_voltage": "-",
            "ref_spec": [],
            "number": "-",
            "date": {"established": "-", "revised": "-", "revision_number": 0}
        }

    # group by y
    lines: Dict[float, List[Tuple[float, float, str, float, float]]] = {}
    for (x0, y0, x1, y1, w, *_) in words:
        w = str(w).strip()
        if not w:
            continue
        yk = round(float(y0), 1)
        lines.setdefault(yk, []).append((float(x0), float(x1), w, float(y0), float(y1)))

    line_list = []
    for yk in sorted(lines.keys()):
        toks = sorted(lines[yk], key=lambda t: t[0])
        line_list.append({
            "yk": yk,
            "toks": toks,  # (x0,x1,word,y0,y1)
            "text": " ".join(t[2] for t in toks),
        })

    mid_x = page_w * 0.52

    # -------------------------
    # helpers for bbox lookup
    # -------------------------
    def norm_tok(s: str) -> str:
        # buang semua non-alnum: "Doc:" -> "doc", "Reff.Doc." -> "reffdoc"
        return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

    def find_label_bbox(label: str, side: str):
        """
        Cari bbox label pada sisi left/right.
        Support label yang kebaca:
        - terpisah: ["reff", "doc"]
        - gabungan: ["reffdoc"]
        """
        label_tokens = [norm_tok(t) for t in label.split()]
        label_tokens = [t for t in label_tokens if t]

        best = None  # (y0, x0, x1, y1)

        for ln in line_list:
            toks = ln["toks"]  # (x0, x1, word, y0, y1)
            words_low = [norm_tok(t[2]) for t in toks]  # align dengan toks

            idxs = [i for i, w in enumerate(words_low) if w]
            w_clean = [words_low[i] for i in idxs]

            # --- case A: sliding window match (multi-token) ---
            for i in range(0, len(w_clean) - len(label_tokens) + 1):
                if w_clean[i:i + len(label_tokens)] == label_tokens:
                    start_i = idxs[i]
                    end_i = idxs[i + len(label_tokens) - 1]

                    x0 = toks[start_i][0]
                    x1 = toks[end_i][1]
                    y0 = min(t[3] for t in toks[start_i:end_i + 1])
                    y1 = max(t[4] for t in toks[start_i:end_i + 1])

                    if side == "left" and x0 > mid_x:
                        continue
                    if side == "right" and x0 < mid_x:
                        continue

                    cand = (y0, x0, x1, y1)
                    if best is None or cand[0] < best[0]:
                        best = cand

            # --- case B: label kebaca jadi 1 token gabungan (mis: "reffdoc") ---
            if len(label_tokens) >= 2:
                joined = "".join(label_tokens)
                for j, w in enumerate(words_low):
                    if not w:
                        continue
                    if w == joined:
                        x0 = toks[j][0]
                        x1 = toks[j][1]
                        y0 = toks[j][3]
                        y1 = toks[j][4]

                        if side == "left" and x0 > mid_x:
                            continue
                        if side == "right" and x0 < mid_x:
                            continue

                        cand = (y0, x0, x1, y1)
                        if best is None or cand[0] < best[0]:
                            best = cand

        return best

    def find_next_label_y(cur_y0: float, side: str) -> Optional[float]:
        next_y = None
        labels = LABELS_LEFT if side == "left" else LABELS_RIGHT

        for lab in labels:
            bb = find_label_bbox(lab, side)
            if not bb:
                continue
            y0 = bb[0]
            if y0 > cur_y0 and (next_y is None or y0 < next_y):
                next_y = y0

        return next_y


    def extract_cell_value(label_bbox, side: str, max_height: Optional[float] = None, same_row: bool = False,  x_pad: float = 6) -> Optional[str]:
        y0, x0, x1, y1 = label_bbox

        x_start = x1 + x_pad
        x_end = (mid_x - 6) if side == "left" else (page_w - 6)

        y_start = y0 - 2

        if same_row:
            # ✅ hanya ambil 1 baris (y-range sempit)
            y_end = y1 + 2
        else:
            next_y = find_next_label_y(y0, side)
            if max_height is not None:
                y_end = y0 + max_height
            else:
                y_end = (next_y - 2) if next_y else (y1 + 60)

        picked = []
        for (wx0, wy0, wx1, wy1, w, *_) in words:
            wx0 = float(wx0); wx1 = float(wx1)
            wy0 = float(wy0); wy1 = float(wy1)
            ww = str(w).strip()
            if not ww:
                continue

            if wx0 < x_start or wx0 > x_end:
                continue
            if wy1 < y_start or wy0 > y_end:
                continue

            picked.append((wy0, wx0, ww))

        picked.sort(key=lambda t: (round(t[0], 1), t[1]))
        val = join_tokens([p[2] for p in picked])
        return val or None





    # -------------------------
    # extract raw
    # -------------------------
    raw: Dict[str, object] = {
        "ref_doc": "-",
        "type": "-",
        "size": "-",
        "rate_voltage": "-",
        "ref_spec": [],
        "number": "-",
        "established": "-",
        "revised": "-",
        "revision_number": 0
    }

    # LEFT
    for lab in LABELS_LEFT:
        bbox = find_label_bbox(lab, side="left")

        # fallback khusus Ref/Reff Doc kalau gak ketemu di kiri
        if not bbox and lab in ("Ref. Doc.", "Reff. Doc."):
            bbox = find_label_bbox(lab, side="right")

        if not bbox:
            continue

        bbox_side = "left" if bbox[1] < mid_x else "right"

        if lab in ("Ref. Doc.", "Reff. Doc."):
            # ✅ ambil 1 baris aja biar gak nyeret Effective Doc. date
            val = extract_cell_value(bbox, side=bbox_side, same_row=True)
        else:
            val = extract_cell_value(bbox, side=bbox_side)

        if not val:
            continue


        if lab == "Ref. Spec.":
            # ✅ ambil area row Ref Spec lebih “nempel” ke tabel, dan jangan sampai SPLN kepotong
            val = extract_cell_value(bbox, side=bbox_side, max_height=55, x_pad=2)
            if not val:
                continue

            txt = ns(val).replace("–", "-").replace("—", "-")

            # ✅ support:
            # SPLN D3.010-1:2014
            # SPLN D3.010-1:2015 ADENDUM
            # IEC 60502-2:2014, IEC 60228, dll
            REFSPEC_RE = re.compile(
                r"(?i)\b(?:SPLN|IEC)\s+[A-Z0-9][A-Z0-9.\-/:]*\b(?:\s+(?:ADENDUM|ADDENDUM|AMENDUM|AMENDMENT))?"
            )

            refs = REFSPEC_RE.findall(txt)

            out = []
            for r in refs:
                rr = ns(r).upper().strip()
                rr = re.sub(r"[;,.]+$", "", rr)  # buang punctuation ekor
                if rr and rr not in out:
                    out.append(rr)

            raw["ref_spec"] = out

        elif lab in ("Ref. Doc.", "Reff. Doc."):
            raw["ref_doc"] = ns(val)

        elif lab == "Rate voltage":
            mkv = re.search(r"(?i)^(.*?\bkV\b)", val)
            raw["rate_voltage"] = ns(mkv.group(1)) if mkv else ns(val)

        elif lab == "Size":
            raw["size"] = clean_size(val)

        elif lab == "Type":
            raw["type"] = re.sub(r"\s+", "", str(val).upper())

        else:
            raw[KEYMAP[lab]] = ns(val)


    # RIGHT
    for lab in LABELS_RIGHT:
        bbox = find_label_bbox(lab, side="right")
        if not bbox:
            continue
        val = extract_cell_value(bbox, side="right")
        if not val:
            continue

        if lab == "Number":
            val = re.sub(r"\s+", "", val).upper()
            raw["number"] = val

        elif lab == "Revision number":
            mnum = re.search(r"\b\d+\b", val)
            try:
                raw["revision_number"] = int(mnum.group(0)) if mnum else int(val)
            except:
                raw["revision_number"] = 0

        elif lab == "Established":
            raw["established"] = normalize_iso_or_keep(val)

        elif lab == "Revised":
            raw["revised"] = normalize_iso_or_keep(val) if val.strip() else "-"

        else:
            raw[KEYMAP[lab]] = ns(val)

    # defaults
    if not raw.get("ref_spec"):
        raw["ref_spec"] = []
    if not raw.get("rate_voltage"):
        raw["rate_voltage"] = "-"

    meta_v2 = {
        "ref_doc": raw.get("ref_doc") or "-",
        "type": raw.get("type") or "-",
        "size": raw.get("size") or "-",
        "rate_voltage": raw.get("rate_voltage") or "-",
        "ref_spec": raw.get("ref_spec") or [],
        "number": raw.get("number") or "-",
        "date": {
            "established": raw.get("established") or "-",
            "revised": raw.get("revised") or "-",
            "revision_number": raw.get("revision_number") or 0
        }
    }
    return meta_v2

def extract_packing_from_pdf(pdf_path: str) -> Dict:
    """
    Extract Packing (Standard length / Net Weight / Gross weight + Wooden drum + end cap)
    menggunakan layout words + split_x otomatis (pemisah sub-tabel kiri & kanan).
    """
    def ns(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    def is_number(tok: str) -> bool:
        return re.fullmatch(r"\d+(?:[.,]\d+)?", tok or "") is not None

    def to_int(s: str):
        try:
            return int(float(s.replace(",", ".")))
        except:
            return None

    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return {}

    try:
        if doc.page_count == 0:
            return {}

        page = doc.load_page(0)
        page_w = float(page.rect.width)

        words = page.get_text("words") or []
        if not words:
            return {}

        # ---------------------------------------------------
        # 1) Tentukan split_x otomatis (pemisah kiri/kanan)
        #    ambil x paling kiri dari kolom kanan (Wooden/Drum/Heat/End/Cap...)
        # ---------------------------------------------------
        RIGHT_HINT = {"wooden", "drum", "heat", "shrink", "end", "cap"}
        right_x_candidates = []

        for x0, y0, x1, y1, w, *_ in words:
            ww = (str(w) or "").strip()
            if not ww:
                continue
            wl = ww.lower()
            if wl in RIGHT_HINT and float(x0) > page_w * 0.55:
                right_x_candidates.append(float(x0))

        # fallback kalau hint tidak ketemu
        split_x = (min(right_x_candidates) - 15) if right_x_candidates else (page_w * 0.75)

        # ---------------------------------------------------
        # 2) Helper: cari Y (baris) untuk label (standard length / net weight / gross weight)
        # ---------------------------------------------------
        tol_y = 2.5  # toleransi baris

        def find_label_y(words_list, a: str, b: str) -> float:
            a = a.lower(); b = b.lower()
            # cari pasangan kata a lalu b di baris y yang sama
            for i in range(len(words_list) - 1):
                x0, y0, x1, y1, w, *_ = words_list[i]
                ww = str(w).strip().lower()
                if ww != a:
                    continue
                y0f = float(y0)
                # cari kata b setelahnya yang y dekat
                for j in range(i + 1, min(i + 8, len(words_list))):
                    x0b, y0b, x1b, y1b, wb, *_ = words_list[j]
                    if abs(float(y0b) - y0f) <= tol_y and str(wb).strip().lower() == b:
                        return y0f
            return None

        # urutkan supaya pencarian stabil
        words_sorted = sorted(words, key=lambda z: (float(z[1]), float(z[0])))

        y_standard = find_label_y(words_sorted, "standard", "length")
        y_net      = find_label_y(words_sorted, "net", "weight") or find_label_y(words_sorted, "net.", "weight")
        y_gross    = find_label_y(words_sorted, "gross", "weight")

        # ---------------------------------------------------
        # 3) Ambil nilai dari band y (label_y) di sisi kiri (< split_x)
        # ---------------------------------------------------
        def pick_value_at_y(label_y: float, unit_word: str):
            if label_y is None:
                return None

            y_top = label_y - 2
            y_bot = label_y + 10  # cukup lebar agar unit & angka yang beda sedikit y tetap masuk

            picked = []
            for x0, y0, x1, y1, w, *_ in words_sorted:
                x0f = float(x0); y0f = float(y0); y1f = float(y1)
                ww = str(w).strip()
                if not ww:
                    continue
                if x0f >= split_x:
                    continue
                if y1f < y_top or y0f > y_bot:
                    continue
                picked.append((x0f, float(x1), ww.lower(), ww))

            if not picked:
                return None

            # cari posisi unit di band ini
            unit_x1 = None
            for x0f, x1f, wl, wraw in sorted(picked, key=lambda t: t[0]):
                if wl == unit_word.lower():
                    unit_x1 = x1f
                    break

            # ambil kandidat angka
            nums = [(x0f, wraw) for x0f, x1f, wl, wraw in picked if is_number(wraw)]
            if not nums:
                return None

            # kalau unit ketemu: ambil angka pertama di kanan unit
            if unit_x1 is not None:
                nums_right = [w for x0f, w in sorted(nums, key=lambda t: t[0]) if x0f >= unit_x1 - 1]
                if nums_right:
                    return nums_right[0]

            # fallback: ambil angka paling kanan (biasanya value di kolom specified)
            nums_sorted = sorted(nums, key=lambda t: t[0])
            return nums_sorted[-1][1]

        result: Dict = {}

        v_std = pick_value_at_y(y_standard, "m")
        v_net = pick_value_at_y(y_net, "kg")
        v_grs = pick_value_at_y(y_gross, "kg")

        if v_std is not None:
            iv = to_int(v_std)
            if iv is not None:
                result["standard_length"] = iv

        if v_net is not None:
            iv = to_int(v_net)
            if iv is not None:
                result["net_weight"] = iv

        if v_grs is not None:
            iv = to_int(v_grs)
            if iv is not None:
                result["gross_weight"] = iv

        # ---------------------------------------------------
        # 4) Extract sisi kanan (>= split_x): Wooden drum + end cap
        # ---------------------------------------------------
        # kumpulkan token kanan jadi text per baris (by y cluster sederhana)
        right_rows = {}
        for x0, y0, x1, y1, w, *_ in words_sorted:
            ww = str(w).strip()
            if not ww:
                continue
            if float(x0) < split_x:
                continue
            yk = round(float(y0), 1)
            right_rows.setdefault(yk, []).append((float(x0), ww))

        for yk in sorted(right_rows.keys()):
            toks = [t[1] for t in sorted(right_rows[yk], key=lambda t: t[0])]
            txt = ns(" ".join(toks)).lower()

            if "wooden" in txt and "drum" in txt:
                nums = [t for t in toks if is_number(t)]
                # contoh: Wooden drum 180 2
                if len(nums) >= 2:
                    result["drum"] = "Wooden Drum"
                    result["drum_type"] = str(int(float(nums[0].replace(",", "."))))
                    result["drum_quantity"] = int(float(nums[1].replace(",", ".")))

            if "end" in txt and "cap" in txt:
                nums = [t for t in toks if is_number(t)]
                if nums:
                    result["end_cap_quantity"] = int(float(nums[-1].replace(",", ".")))

        return result

    finally:
        doc.close()


def extract_cabling_right_raw_materials_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Ambil raw material untuk Cabling dari KOLOM KANAN tabel (TYPE + QUANTITY).
    Target: "PVC Sheath BK (Black) 28,4"
    Output: [{"type":"PVC Sheath BK (Black)", "quantity":28.4, "unit":"kg/km"}]
    """

    def ns(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    def to_float(s: str):
        try:
            return float(s.replace(",", "."))
        except:
            return None

    def norm_color(s: str) -> str:
        # "BK ( Black )" -> "BK (Black)"
        s = ns(s)
        s = re.sub(r"\(\s*", "(", s)
        s = re.sub(r"\s*\)", ")", s)
        s = re.sub(r"\(\s*([A-Za-z ]+?)\s*\)", lambda m: f"({ns(m.group(1))})", s)
        return s

    COLOR_RE = re.compile(r"(?i)^[A-Z]{1,3}\s*\(\s*[A-Za-z ]+\s*\)$")

    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return []

    try:
        if doc.page_count == 0:
            return []

        page = doc.load_page(0)
        page_w = float(page.rect.width)
        words = page.get_text("words") or []
        if not words:
            return []

        # ----------------------------
        # 1) cari anchor "6 Cabling"
        # ----------------------------
        cab_y = None
        for x0, y0, x1, y1, w, *_ in words:
            ww = ns(str(w))
            if re.fullmatch(r"(?i)6", ww):
                for x0b, y0b, x1b, y1b, wb, *_ in words:
                    if abs(float(y0b) - float(y0)) <= 2.5 and ns(str(wb)).lower() == "cabling":
                        cab_y = float(y0)
                        break
            if cab_y is not None:
                break
        if cab_y is None:
            return []

        # ----------------------------
        # 2) cari anchor section berikutnya "7 Bedding" untuk batas y_bot
        #    (supaya PVC Sheath 705... 1192,8 tidak kebawa ke Cabling)
        # ----------------------------
        next_y = None
        for x0, y0, x1, y1, w, *_ in words:
            ww = ns(str(w))
            if re.fullmatch(r"(?i)7", ww):
                for x0b, y0b, x1b, y1b, wb, *_ in words:
                    if abs(float(y0b) - float(y0)) <= 2.5 and ns(str(wb)).lower() == "bedding":
                        next_y = float(y0)
                        break
            if next_y is not None:
                break

        y_top = cab_y + 5
        y_bot = (next_y - 2) if next_y else (cab_y + 110)  # fallback kalau bedding gak ketemu

        # ----------------------------
        # 2.5) tentukan referensi X kolom QTY (paling kanan) khusus band Cabling
        # ----------------------------
        qty_x_candidates = []
        for x0, y0, x1, y1, w, *_ in words:
            ww = ns(str(w))
            if not ww:
                continue
            y0f = float(y0); y1f = float(y1)
            if y1f < y_top or y0f > y_bot:
                continue

            # angka murni (qty)
            if re.fullmatch(r"\d+(?:[.,]\d+)?", ww):
                x0f = float(x0)
                if x0f > page_w * 0.70:   # kandidat harus cukup kanan
                    qty_x_candidates.append(x0f)

        # pilih yang paling kanan (kolom quantity biasanya paling kanan)
        qty_x_ref = max(qty_x_candidates) if qty_x_candidates else (page_w * 0.85)


        # ----------------------------
        # 3) ambil token kolom kanan (DINAMIS dari kolom qty)
        # ----------------------------
        qty_x_candidates = []
        for x0, y0, x1, y1, w, *_ in words:
            ww = ns(str(w))
            if not ww:
                continue
            y0f = float(y0); y1f = float(y1)
            if y1f < y_top or y0f > y_bot:
                continue
            if re.fullmatch(r"\d+(?:[.,]\d+)?", ww) and float(x0) > page_w * 0.70:
                qty_x_candidates.append(float(x0))

        # batas kiri kolom kanan: mundur dari posisi angka qty
        x_right_min = (min(qty_x_candidates) - 260) if qty_x_candidates else (page_w * 0.60)

        right = []
        for x0, y0, x1, y1, w, *_ in words:
            x0f = float(x0); y0f = float(y0); y1f = float(y1)
            ww = ns(str(w))
            if not ww:
                continue
            if y1f < y_top or y0f > y_bot:
                continue

            # ✅ ambil hanya yang benar-benar masuk kolom kanan
            if x0f >= x_right_min:
                right.append((round(y0f, 1), x0f, ww))
                continue

            # ✅ warna kadang lebih kiri, tetap ambil kalau bentuknya token warna
            if x0f >= x_right_min - 80:
                if re.fullmatch(r"(?i)[A-Z]{1,3}|\(|\)|[A-Za-z]+", ww):
                    right.append((round(y0f, 1), x0f, ww))

        if not right:
            return []


        # ----------------------------
        # 4) group per baris y
        # ----------------------------
        rows = {}
        for yk, x, ww in right:
            rows.setdefault(yk, []).append((x, ww))

        out = []
        ykeys = sorted(rows.keys())
        i = 0

        while i < len(ykeys):
            yk = ykeys[i]

            # ✅ WAJIB: toks_xy berisi (x, token) biar bisa pilih angka dari kolom qty
            toks_xy = sorted(rows[yk], key=lambda z: z[0])  # rows[yk] = [(x, token), ...]
            line = ns(" ".join(t[1] for t in toks_xy))

            # ✅ SKIP baris spesifikasi (OD/LP/DL) yang nyasar ke kolom kanan
            # contoh: "mm 1056 - 1377 / Z BK ( Black )" atau "mm 45,9"
            if re.match(r"(?i)^\s*mm\b", line):
                i += 1
                continue

            # skip juga kalau ada pola range "/" yang khas OD/LP/DL
            if re.search(r"(?i)\b\d+\s*-\s*\d+\b", line) and "/" in line:
                i += 1
                continue

            # =========================
            # ✅ PILIH QTY BERDASARKAN POSISI X (kolom qty)
            # =========================
            num_hits = []
            for x, ww in toks_xy:
                if re.fullmatch(r"\d+(?:[.,]\d+)?", ww):
                    num_hits.append((x, ww))

            if not num_hits:
                i += 1
                continue

            # pilih angka yang paling dekat dengan kolom qty (qty_x_ref harus sudah dihitung di atas)
            x_qty, qty_txt = min(num_hits, key=lambda p: abs(p[0] - qty_x_ref))

            # validasi: angka qty harus cukup kanan
            if x_qty < page_w * 0.70:
                i += 1
                continue

            qty = to_float(qty_txt)
            if qty is None:
                i += 1
                continue

            # type = semua token kecuali token qty yang kepilih
            parts = []
            for x, ww in toks_xy:
                if ww == qty_txt and abs(x - x_qty) < 0.5:
                    continue
                parts.append(ww)

            ttype = ns(" ".join(parts)).strip(" :-")
            # =========================

            # buang nyangkut kolom kiri
            ttype = re.sub(r"(?i)^\s*\d+\s*/\s*(triangle|round|square)\s+", "", ttype).strip()
            ttype = re.sub(r"(?i)^\s*(triangle|round|square)\s+", "", ttype).strip()

            if not ttype or "sheath" not in ttype.lower():
                i += 1
                continue

            # --- BLOK CARI WARNA ---
            color_suffix = None
            COLOR_IN_LINE_RE = re.compile(r"(?i)\b([A-Z]{1,3})\s*\(\s*([A-Za-z ]+)\s*\)")

            for hop in (1, 2, 3):
                if i + hop >= len(ykeys):
                    break
                yk2 = ykeys[i + hop]
                if (yk2 - yk) > 40:
                    break

                toks2 = [t[1] for t in sorted(rows[yk2], key=lambda z: z[0])]
                line2 = ns(" ".join(toks2))

                m = COLOR_IN_LINE_RE.search(line2)
                if m:
                    code = m.group(1).upper()
                    name = ns(m.group(2)).title()
                    color_suffix = norm_color(f"{code} ({name})")
                    break

            if color_suffix and color_suffix.lower() not in ttype.lower():
                ttype = f"{ttype} {color_suffix}".strip()

            out.append({"type": ttype, "quantity": qty, "unit": "kg/km"})
            i += 1


        return out
    finally:
        doc.close()


def extract_outer_sheath_right_raw_materials_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Ambil raw_materials OUTER SHEATH dari KOLOM KANAN tabel (material + qty).
    Target contoh:
      - "PVC Sheath 705 FL Cat-C" 748,1
      - "Additive Anti termite" 14,96
      - lanjutan tanpa qty: "UV Resistant ( Red )" -> ditempel ke entry sebelumnya
    Output:
      [{"type":"PVC Sheath 705 FL Cat-C", "quantity":748.1, "unit":"kg/km"},
       {"type":"Additive Anti termite UV Resistant (Red)", "quantity":14.96, "unit":"kg/km"}]
    """
    import re
    import fitz

    def ns(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    def to_float(s: str):
        try:
            return float(s.replace(",", "."))
        except:
            return None

    def norm_colorish(s: str) -> str:
        # "( Red )" -> "(Red)"
        s = ns(s)
        s = re.sub(r"\(\s*", "(", s)
        s = re.sub(r"\s*\)", ")", s)
        s = re.sub(r"\(\s*([A-Za-z ]+?)\s*\)", lambda m: f"({ns(m.group(1))})", s)
        return s

    def is_junk_material_line(txt: str) -> bool:
        if txt.strip() == "ok":
            return True
        t = txt.lower()
        # buang baris tech spec / marking
        if t.startswith("mm "):  # "mm 2,8" dll
            return True
        if "thickness" in t or "diameter" in t or "overall" in t:
            return True
        if "od/lp/dl" in t:
            return True
        if "kv" in t or "year" in t or "length marking" in t:
            return True
        if re.search(r"\b\d+\s*meter\b", t):
            return True
        # ini biasanya kalimat marking panjang
        if "tcu/xlpe" in t or "jembo" in t:
            return True
        return False

    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return []

    try:
        if doc.page_count == 0:
            return []

        page = doc.load_page(0)
        page_w = float(page.rect.width)
        words = page.get_text("words") or []
        if not words:
            return []

        # 1) cari anchor "9 Outer sheath"
        os_y = None
        for x0, y0, x1, y1, w, *_ in words:
            ww = ns(str(w))
            if re.fullmatch(r"(?i)9", ww):
                for x0b, y0b, x1b, y1b, wb, *_ in words:
                    if abs(float(y0b) - float(y0)) <= 2.5 and ns(str(wb)).lower() in {"outer", "sheath", "outer sheath"}:
                        # kadang "Outer" "sheath" kepisah; cukup ambil y dari token 9
                        os_y = float(y0)
                        break
            if os_y is not None:
                break
        if os_y is None:
            return []

        # 2) batas bawah: cari section berikutnya "10 Marking" / "11 Packing" (kalau ada)
        next_y = None
        for x0, y0, x1, y1, w, *_ in words:
            ww = ns(str(w)).lower()
            if ww in {"10", "11"}:
                # cari teks "marking"/"packing" dekatnya
                for x0b, y0b, x1b, y1b, wb, *_ in words:
                    if abs(float(y0b) - float(y0)) <= 2.5:
                        wbb = ns(str(wb)).lower()
                        if wbb in {"marking", "packing"}:
                            next_y = float(y0)
                            break
            if next_y is not None:
                break

        y_top = os_y + 5
        y_bot = (next_y - 2) if next_y else (os_y + 140)
        
        # ----------------------------
        # ✅ tentukan referensi X kolom QTY (paling kanan) di band Outer sheath
        # ----------------------------
        qty_x_candidates = []
        for x0, y0, x1, y1, w, *_ in words:
            ww = ns(str(w))
            if not ww:
                continue
            y0f = float(y0); y1f = float(y1)
            if y1f < y_top or y0f > y_bot:
                continue
            if re.fullmatch(r"\d+(?:[.,]\d+)?", ww):
                x0f = float(x0)
                if x0f > page_w * 0.70:
                    qty_x_candidates.append(x0f)

        if qty_x_candidates:
            # ✅ pilih yang PALING KANAN (kolom qty itu paling kanan)
            qty_x_ref = max(qty_x_candidates)
        else:
            qty_x_ref = page_w * 0.85

         #✅ DEBUG TAROK DI SINI
        print("OUTER_SHEATH qty_x_ref:", qty_x_ref, "page_w:", page_w)


        # ----------------------------
        # 3) ambil token area kanan (material + qty) - LONGGAR
        #    nanti pemilihan qty tetap STRICT by x dekat qty_x_ref
        # ----------------------------

        # 3) ambil token area kanan (material + qty) BERDASARKAN kolom qty
        # ✅ material column biasanya mulai cukup kanan (PVC/Additive/UV ada di x~420+)

        x_material_min = max(page_w * 0.65, qty_x_ref - 180)   # ✅ ini kunci (buang 2,04 / 59,9)
        x_right_max    = qty_x_ref + 40                         # qty column ga perlu terlalu lebar kanan

        right = []
        for x0, y0, x1, y1, w, *_ in words:
            ww = ns(str(w))
            if not ww:
                continue
            x0f = float(x0); y0f = float(y0); y1f = float(y1)
            if y1f < y_top or y0f > y_bot:
                continue

            # ambil area kanan cukup longgar (jangan terlalu sempit)
            if x0f < page_w * 0.45:
                continue

            right.append((round(y0f, 1), x0f, ww))

        if not right:
            return []


        # 4) group per baris y
        rows = {}
        for yk, x, ww in right:
            rows.setdefault(yk, []).append((x, ww))

        ykeys = sorted(rows.keys())

        out: List[Dict] = []
        last_idx = None

        pending_type = None  # <-- TARUH INI sebelum loop for yk in ykeys

        for yk in ykeys:
            toks_xy = sorted(rows[yk], key=lambda z: z[0])

            # ✅ buang token spec kolom kiri, tapi tetap ambil qty token kolom paling kanan
            toks_xy2 = [(x, ww) for x, ww in toks_xy if (x >= x_material_min) or (abs(x - qty_x_ref) <= 3.0)]
            line_full = ns(" ".join(t[1] for t in toks_xy2))


            # skip baris spesifikasi nyasar
            if re.match(r"(?i)^\s*mm\b", line_full):
                continue
            if re.search(r"(?i)\b\d+\s*-\s*\d+\b", line_full) and "/" in line_full:
                continue

            # kumpulin angka di baris + posisi x
            num_hits = [(x, ww) for x, ww in toks_xy2 if re.fullmatch(r"\d+(?:[.,]\d+)?", ww)]

            # helper: valid qty token harus dekat kolom qty_x_ref
            def pick_qty_token(num_hits):
                if not num_hits:
                    return None
                x_qty, qty_tok = min(num_hits, key=lambda p: abs(p[0] - qty_x_ref))
                if abs(x_qty - qty_x_ref) > 45:
                    return None
                q = to_float(qty_tok)
                if q is None:
                    return None
                return (x_qty, qty_tok, q)

            picked = pick_qty_token(num_hits)
            print("OS_LINE:", line_full)
            print("  picked:", picked, " pending:", pending_type)


            # =========================================================
            # CASE A) BARIS TANPA QTY VALID -> bisa jadi:
            #   - material line ("PVC Sheath 705 FL Cat-C")
            #   - continuation ("UV Resistant", "(Red)")
            # =========================================================
            if not picked:
                cont = norm_colorish(line_full)
                tlc = (cont or "").lower()

                # stop kalau kosong / junk
                if not cont or is_junk_material_line(cont):
                    pending_type = None
                    continue

                # =========================
                # ✅ RULE 1: UV Resistant & warna -> SELALU continuation ke item terakhir (biasanya Additive)
                # =========================
                is_uv_line = ("uv" in tlc) or ("resistant" in tlc)
                is_color_only = bool(re.fullmatch(r"(?i)\(\s*[a-z]+\s*\)", cont)) or bool(re.fullmatch(r"(?i)[a-z]+\b", cont))

                if (is_uv_line or is_color_only) and out:
                    tprev = out[-1]["type"]
                    if cont.lower() not in tprev.lower():
                        out[-1]["type"] = ns(f"{tprev} {cont}")
                    continue

                # =========================
                # ✅ RULE 2: baris material utama -> simpan sebagai pending_type
                # (PVC Sheath 705 FL Cat-C) dan (Additive Anti termite)
                # =========================
                is_main_material = any(k in tlc for k in ["sheath", "additive", "termite"])
                if is_main_material:
                    if pending_type is None:
                        pending_type = cont
                    else:
                        if cont.lower() not in pending_type.lower():
                            pending_type = ns(f"{pending_type} {cont}")
                    continue

                # =========================
                # ✅ RULE 3: sisanya -> kalau tidak ada angka, tempel ke last item
                # =========================
                if re.search(r"\d", cont):
                    # ada digit tapi bukan main material -> ini biasanya noise/spec -> skip
                    continue

                if out:
                    tprev = out[-1]["type"]
                    if cont.lower() not in tprev.lower():
                        out[-1]["type"] = ns(f"{tprev} {cont}")
                continue


            # =========================================================
            # CASE B) BARIS PUNYA QTY VALID
            #   - bisa "qty only" (contoh: "748,1")
            #   - bisa campuran ("2,8 14,96") -> qty=14,96 yang dekat qty_x_ref
            # =========================================================
            x_qty, qty_tok, qty = picked

            # ambil type token dari area material (kiri qty) dalam window
            parts = []
            for x, ww in toks_xy2:
                # skip token qty terpilih
                if ww == qty_tok and abs(x - x_qty) < 2.0:
                    continue
                # ambil token material yang dekat kolom qty (hindari spec jauh kiri)
                if x < (x_qty - 340) or x >= (x_qty - 2):
                    continue
                parts.append(ww)

            ttype = norm_colorish(ns(" ".join(parts)).strip(" :-"))

            # kalau baris ini "qty only", ttype akan kosong -> pakai pending_type
            if not ttype and pending_type:
                ttype = pending_type
                pending_type = None

            # kalau masih kosong, skip
            if not ttype:
                continue

            # kalau ttype junk, skip
            if is_junk_material_line(ttype):
                continue

            tl = ttype.lower()
            if not any(k in tl for k in ["sheath", "additive", "termite", "uv"]):
                # kalau pending_type ada, bisa jadi ini sebenarnya qty untuk pending_type (jarang)
                if pending_type and any(k in pending_type.lower() for k in ["sheath", "additive", "termite", "uv"]):
                    ttype = pending_type
                    pending_type = None
                else:
                    continue

            out.append({"type": ttype, "quantity": qty, "unit": "kg/km"})


        # 5) dedup (type+qty)
        ded = {}
        for rm in out:
            key = (rm["type"].lower(), float(rm["quantity"]))
            ded[key] = rm

        return list(ded.values())

    finally:
        doc.close()


def extract_denier_specs_from_pdf(pdf_path: str) -> Dict[str, str]:
    """
    Ambil specified untuk:
      - Centre filler (unit n/denier)
      - Filler (unit n/denier)
    dari halaman 1 (page 0) berbasis area row & kolom (multi-line aman).
    Return:
      {"centre filler": "1 / (1 x 100,000)", "filler": "3 / (9 x 100,000 + 1 x 10,000)"}
    """
    def ns(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    def clean_expr(s: str) -> str:
        s = ns(s)
        s = re.sub(r"\s*/\s*", " / ", s)
        s = re.sub(r"\s*x\s*", " x ", s, flags=re.I)
        s = re.sub(r"\s*\+\s*", " + ", s)
        s = re.sub(r"\(\s*", "(", s)
        s = re.sub(r"\s*\)", ")", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def norm_line(s: str) -> str:
        s = ns(s).lower()
        s = re.sub(r"^[\-\u2022]+\s*", "", s)  # buang bullet "-" / "•"
        return s

    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return {}

    try:
        if doc.page_count == 0:
            return {}
        page = doc.load_page(0)
        page_w = float(page.rect.width)
        words = page.get_text("words") or []
        if not words:
            return {}

        # group by y -> lines (yk)
        lines = {}
        for x0, y0, x1, y1, w, *_ in words:
            ww = str(w).strip()
            if not ww:
                continue
            yk = round(float(y0), 1)
            lines.setdefault(yk, []).append((float(x0), float(x1), ww, float(y0), float(y1)))

        ordered_y = sorted(lines.keys())

        def line_text(yk: float) -> str:
            toks = sorted(lines[yk], key=lambda t: t[0])
            return ns(" ".join(t[2] for t in toks))

        # cari y untuk 2 row denier
        row_y = {}
        for yk in ordered_y:
            txt = norm_line(line_text(yk))
            if "n/denier" in txt:
                if "centre" in txt and "filler" in txt:
                    row_y["centre filler"] = yk
                elif txt.startswith("filler"):
                    row_y["filler"] = yk

        if not row_y:
            return {}

        # cari y batas bawah row: y berikutnya yang merupakan row baru (filler/od/no./min./outer)
        def find_next_row_y(start_yk: float) -> Optional[float]:
            for yk in ordered_y:
                if yk <= start_yk:
                    continue
                t = norm_line(line_text(yk))
                if ("n/denier" in t and ("filler" in t)) or t.startswith("od/lp/dl") or t.startswith("no.") or t.startswith("min.") or t.startswith("outer diameter"):
                    return yk
            return None

        # tentukan batas kolom raw material (kanan) secara otomatis: cari word material di band y row tsb
        MATERIAL_WORDS = {"pp", "yarn", "polyester", "tape", "copper", "binder", "xlpe", "pvc", "pe", "semiconductive"}

        results = {}
        for key in ["centre filler", "filler"]:
            if key not in row_y:
                continue

            yk0 = row_y[key]
            y_top = min(t[3] for t in lines[yk0]) - 2

            nxt = find_next_row_y(yk0)
            if nxt:
                y_bottom = min(t[3] for t in lines[nxt]) - 2
            else:
                # fallback ketat: hanya sekitar 1 row
                y_bottom = y_top + 30

            # cari posisi unit "n/denier" pada row ini
            unit_x1 = None
            for x0, x1, ww, yy0, yy1 in sorted(lines[yk0], key=lambda z: z[0]):
                if "n/denier" in ww.lower():
                    unit_x1 = x1
                    break

            x_start = (unit_x1 + 6) if unit_x1 else (page_w * 0.32)

            # cari batas kanan (awal kolom raw material) di dalam band y ini
            raw_min_x0 = None
            for x0, y0, x1, y1, w, *_ in words:
                ww = str(w).strip()
                if not ww:
                    continue
                if float(x0) <= x_start + 20:
                    continue
                if float(y1) < y_top or float(y0) > y_bottom:
                    continue
                if ns(ww).lower() in MATERIAL_WORDS:
                    raw_min_x0 = float(x0) if raw_min_x0 is None else min(raw_min_x0, float(x0))

            x_end = (raw_min_x0 - 8) if raw_min_x0 else (page_w * 0.70)

            # ambil words di kolom specified
            picked = []
            for x0, y0, x1, y1, w, *_ in words:
                ww = str(w).strip()
                if not ww:
                    continue
                x0f = float(x0); y0f = float(y0); y1f = float(y1)
                if x0f < x_start or x0f > x_end:
                    continue
                if y1f < y_top or y0f > y_bottom:
                    continue
                picked.append((y0f, x0f, ww))

            picked.sort(key=lambda t: (round(t[0], 1), t[1]))
            expr = clean_expr(" ".join(p[2] for p in picked))

            # ✅ normalisasi kalau kebalik: "(...) 1 /" -> "1 / (...)"
            m_rev = re.search(r"(\([^)]*\))\s*(\d+)\s*/", expr)
            if m_rev:
                expr = f"{m_rev.group(2)} / {m_rev.group(1)}"

            # ambil pola normal
            m = re.search(r"(\d+\s*/\s*\([^)]*\))", expr)
            if m:
                results[key] = clean_expr(m.group(1))
            else:
                # fallback: minimal "digit / ..."
                m2 = re.search(r"(\d+\s*/\s*.+)$", expr)
                results[key] = clean_expr(m2.group(1)) if m2 else expr


        return results
    finally:
        doc.close()




# ----------------------------
# Section detection (ordered)
# ----------------------------
CANON_MAP = {
    "Conductor -": "Conductor",
    "Conductor Shielding -": "Conductor Shielding",
    "Covering -": "Covering",
    "Final Test -": "Final Test",
    "Final test -": "Final Test",
    "Packing": "Packing",
    "Tapping -": "Tapping",
    "Insulation -": "Insulation",
    "Insulation Shielding -": "Insulation Shielding",
    "Metallic Screen -": "Metallic Screen",
    "Twisting -": "Twisting",
    "Overall Screen": "Overall Screen",
    "Individual Screen": "Individual Screen",
    "Cabling -": "Cabling",
    "Bedding -": "Bedding",
    "Inner Sheath -": "Inner Sheath",
    "Armouring (Braided) -": "Armouring (Braided)",
    "Armouring (Braided)": "Armouring (Braided)", 
    "Armour -": "Armour",
    "Outer Sheath -": "Outer Sheath",
}

def build_header_patterns(headers: List[str]) -> List[Tuple[str, re.Pattern]]:
    out: List[Tuple[str, re.Pattern]] = []

    for h in headers:
        base = h.strip().rstrip("-").strip()
        esc = re.escape(base).replace(r"\ ", r"\s+")
        if h.strip().endswith("-"):
            out.append((h, re.compile(rf"(?i)\b{esc}\b\s*\-", re.I)))
        else:
            out.append((h, re.compile(rf"(?i)\b{esc}\b", re.I)))

    numeric = {
        "Conductor -": r"(?i)\b1\s+Conductor\b",
        "Conductor Shielding -": r"(?i)\b2\s+Conductor\s+Shielding\b",
        "Insulation -": r"(?i)\b3\s+Insulation\b",
        "Insulation Shielding -": r"(?i)\b4\s+Insulation\s+Shielding\b",
        "Metallic Screen -": r"(?i)\b5\s+Metallic\s+Screen\b",
        "Cabling -": r"(?i)\b6\s+Cabling\b",
        "Bedding -": r"(?i)\b7\s+Bedding\b",
        "Inner Sheath -": r"(?i)\b7\s+Inner\s+Sheath\b",
        "Armouring (Braided) -": r"(?i)\b8\s+Armouring\s*\(\s*Braided\s*\)\b",
        "Armour -": r"(?i)\b8\s+Armour\b",
        "Outer Sheath -": r"(?i)\b9\s+Outer\s+Sheath\b",
        "Final Test -": r"(?i)\b10\s+Final\s+Test\b|\b3\s+Final\s+Test\b",
        "Packing": r"(?i)\b11\s+Packing\b|\b4\s+Packing\b",
        "Tapping -": r"(?i)\b2\s+Tapping\b",
        "Twisting -": r"(?i)\b3\s+Twisting\b",
        "Individual Screen": r"(?i)\b4\s+Individual\s+Screen\b",
        "Overall Screen": r"(?i)\b6\s+Overall\s+Screen\b",
        "Covering -": r"(?i)\b2\s+Covering\b",
    }

    hdr_set = set(headers)
    for h, pat in numeric.items():
        if h in hdr_set:
            out.append((h, re.compile(pat, re.I)))

    return out

def parse_sections_dynamic(content: str, headers: List[str]) -> List[Tuple[str, str]]:
    pats = build_header_patterns(headers)

    matches = []
    for header, pat in pats:
        m = pat.search(content)
        if m:
            matches.append({"header": header, "start": m.start(), "end": m.end()})

    matches.sort(key=lambda x: x["start"])
    if not matches:
        return []

    ordered: List[Tuple[str, str]] = []
    for i, m in enumerate(matches):
        start_pos = m["end"]
        end_pos = matches[i + 1]["start"] if i < len(matches) - 1 else len(content)
        chunk = content[start_pos:end_pos].strip()
        if not chunk:
            continue
        sec_name = CANON_MAP.get(m["header"], m["header"])
        ordered.append((sec_name, chunk))

    return ordered


# ----------------------------
# Parse section text (V2)
# ----------------------------
def split_items_safe(section_text: str) -> List[str]:
    t = norm(section_text)
    if not t:
        return []
    parts = re.split(r"\s-\s(?=[A-Za-z])", t)
    return [p.strip() for p in parts if p.strip()]

def to_snake(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

_RAW_MAT_BLOCKLIST = [
    "thickness", "diameter", "resistance", "test", "identification", "construction",
    "shape", "marking", "tensile", "elongation", "voltage", "weight", "length",
    "drum", "cap", "od/lp/dl", "min.", "max."
]

def _looks_like_raw_material(type_text: str) -> bool:
    tl = (type_text or "").lower()
    return not any(k in tl for k in _RAW_MAT_BLOCKLIST)

def _parse_qty_float(q: str) -> Optional[float]:
    q = (q or "").strip()
    if not q:
        return None
    try:
        return float(q.replace(",", "."))
    except:
        return None

def extract_layers_from_odlpdl(desc: str, unit: Optional[str], specified: Optional[str]) -> List[Dict]:
    """
    Dari item OD/LP/DL of outer layer ... jadi layers[] seperti V2.
    """
    d = norm(desc)
    u = norm(unit) if unit else "mm/-"
    sp = norm(specified)

    full = norm(f"{d} {u} {sp}")

    if "od/lp/dl" not in full.lower() or "layer" not in full.lower():
        return []

    # pastikan unit mm/- jadi standard
    full = re.sub(r"(?i)\bmm\s*/\s*-\b", "mm/-", full)

    # cari prefix OD/LP/DL
    prefix = "OD/LP/DL of outer layer"
    if re.search(r"(?i)\bod/lp/dl\s+of\s+outer\s+layer\b", full):
        # buang prefix biar regex gampang, tapi nanti kita tambahin lagi di description
        full_tail = re.split(r"(?i)\bod/lp/dl\s+of\s+outer\s+layer\b", full, maxsplit=1)[1].strip()
    else:
        full_tail = full

    # pola: "<nth Layer> <construction...> mm/- <specified...> ... (hingga sebelum "- <next layer>")"
    layer_pat = re.compile(
        r"(?i)(\d+(?:st|nd|rd|th)\s+Layer)\s+(.+?)\s+mm/\-\s+(.+?)(?=\s*-\s*\d+(?:st|nd|rd|th)\s+Layer|\s*$)"
    )

    out = []
    for m in layer_pat.finditer(full_tail):
        layer_name = norm(m.group(1))
        construction = norm(m.group(2))
        spec = norm(m.group(3))
        if layer_name and construction and spec:
            out.append({
                "description": f"{prefix} {layer_name}",
                "construction": construction,
                "unit": "mm/-",
                "specified": spec
            })

    return out

def parse_section_v2(
    section_text: str,
    default_rm_unit: Optional[str] = None,
    enable_layers: bool = False
) -> Dict:
    """
    Gabungan:
    - PRE-EXTRACT global raw material dari seluruh section (kg/km & Wire qty terbesar) -> bagus untuk Conductor
    - EXTRACT raw material dari tail specified "spec + type + qty" -> bagus untuk Covering
    Output:
      {
        "raw_materials": [{"type":..., "quantity":..., "unit":"kg/km"}, ...],
        "technical_specs": [{"description":..., "unit":..., "specified":...}, ...],
        "layers": [...] (optional)
      }
    """
    raw_materials: List[Dict] = []
    technical_specs: List[Dict] = []
    layers: List[Dict] = []

    items = split_items_safe(section_text)
    guessed_unit = default_rm_unit or "kg/km"

    # -----------------------------
    # helpers
    # -----------------------------

    def _next_centre_filler_ratio(lines: List[str], idx: int, max_hop: int = 4) -> Optional[str]:
        # cari pola "1 / Triangle" di baris berikutnya
        pat = re.compile(r"(?i)\b(\d+)\s*/\s*(triangle|round|square)\b")
        for k in range(idx + 1, min(len(lines), idx + 1 + max_hop)):
            s = re.sub(r"\s+", " ", (lines[k] or "")).strip()
            m = pat.search(s)
            if m:
                num = m.group(1)
                shp = m.group(2).capitalize()
                return f"{num} / {shp}"
        return None


    def inject_core_identification_from_section_text(specs: List[Dict], section_text_: str) -> List[Dict]:
        """
        Kalau ada row 'Core identification' tapi warna kepotong,
        ambil warna dari section_text (string section) langsung.
        """
        st = norm(section_text_)

        # cari blok setelah "Core identification" lalu ambil warna yang muncul
        m = re.search(r"(?i)\bcore\s+identification\b(.{0,120})", st)
        if not m:
            return specs

        tail = m.group(1)

        # ambil warna yang mungkin muncul di tail
        colors = ["Brown","Black","Grey","Gray","Red","Blue","Yellow","Green","White","Orange","Purple","Violet","Pink"]
        found = []
        for c in colors:
            if re.search(rf"(?i)\b{c}\b", tail):
                cc = "Grey" if c.lower() == "gray" else c
                if cc not in found:
                    found.append(cc)

        if not found:
            return specs

        # update row core identification yang ada
        for row in specs:
            if (row.get("description") or "").strip().lower() == "core identification":
                row["unit"] = "-"
                row["specified"] = ", ".join(found)
                break

        return specs


    def dedup_technical_specs(specs: List[Dict]) -> List[Dict]:
        """
        Dedup berdasarkan (description_normalized, unit, specified).
        Normalisasi: lowercase + hilangkan titik berlebih + spasi rapi.
        """
        def norm_desc(d: str) -> str:
            d = (d or "").strip().lower()
            d = re.sub(r"\.+", ".", d)          # rapihin titik
            d = d.replace("nom. thickness", "nom thickness")
            d = re.sub(r"\s+", " ", d).strip()
            return d

        seen = set()
        out = []
        for row in specs:
            d0 = row.get("description") or ""
            u0 = (row.get("unit") or "-").strip()
            s0 = (row.get("specified") or "-").strip()

            key = (norm_desc(d0), u0, s0)
            if key in seen:
                continue

            # sekalian rapihin output description khusus Nom thickness
            if norm_desc(d0) == "nom thickness":
                row = dict(row)
                row["description"] = "Nom thickness"

            out.append(row)
            seen.add(key)
        return out


    def _pick_nom_thickness_from_window(lines_: List[str], start_idx: int, max_hop: int = 6) -> Optional[Tuple[int, str]]:
        stop_re = re.compile(r"(?i)\b(min\.?\s*thickness|core\s+identification|outer\s+diameter)\b")
        window = []
        last_i = start_idx
        for j in range(start_idx, min(len(lines_), start_idx + max_hop)):
            t = norm(lines_[j])
            if not t:
                continue
            if j != start_idx and stop_re.search(t):
                break
            window.append(t)
            last_i = j

        blob = " ".join(window)
        m = re.search(r"(?i)\bnom\.?\s*thickness\b.*?(\d+(?:[.,]\d+)?)\b", blob)
        if m:
            return last_i, m.group(1)
        return None

    def split_multi_specs_line(r: str) -> list[str]:
        """
        Pecah baris yang berisi beberapa item spec sekaligus.
        Fokus: kasus 'Nom thickness' + 'Min. Thickness...' dalam satu baris.
        """
        s = (r or "").strip()
        if not s:
            return []

        # split khusus yang sering kejadian di PDF kamu
        # "Nom thickness ... Min. Thickness at any point ..."
        m = re.search(r"(?i)\bmin\.?\s*thickness\s+at\s+any\s+point\b", s)
        if m and re.search(r"(?i)\bnom\.?\s*thickness\b", s):
            a = s[:m.start()].strip()
            b = s[m.start():].strip()
            out = []
            if a: out.append(a)
            if b: out.append(b)
            return out

        # default: tidak perlu split
        return [s]


    def fix_tinned_annealed_copper_tape(rms: List[Dict], section_text: str) -> List[Dict]:
        """
        Kalau tabel raw material kebaca pecah:
        TYPE: "Tinned Annealed"
        baris berikutnya: "Copper Tape"
        maka gabungkan jadi "Tinned Annealed Copper Tape"
        """
        st = (section_text or "").lower()

        # hanya lakukan kalau memang ada indikasi Copper Tape di teks section
        if "copper tape" not in st:
            return rms

        for rm in rms:
            t = (rm.get("type") or "").strip()
            if not t:
                continue
            if t.lower() == "tinned annealed":
                rm["type"] = "Tinned Annealed Copper Tape"
                break

        return rms

    def attach_uv_resistant_from_section_text(rms: List[Dict], section_text: str) -> List[Dict]:
        """
        Kalau di section_text ada 'UV Resistant', tempel ke raw material terakhir yang bertipe sheath/insulation/filler.
        """
        st = norm(section_text).lower()

        # deteksi UV Resistant meskipun kepisah spasi/line
        has_uv = ("uv" in st and "resist" in st)

        if not has_uv:
            return rms

        for i in range(len(rms) - 1, -1, -1):
            t = (rms[i].get("type") or "").strip()
            tl = t.lower()

            # hanya untuk material yang wajar punya UV Resistant
            if re.search(r"(?i)\b(sheath|insulation|filler)\b", t):
                if "uv resistant" not in tl:
                    rms[i]["type"] = (t + " UV Resistant").strip()

                ## rapihin urutan: taruh UV Resistant sebelum kode warna seperti "RD (Red)"
                tcur = rms[i]["type"]

                # case: "RD (Red) UV Resistant" -> "UV Resistant RD (Red)"
                tcur = re.sub(
                    r"(?i)\b((?:RD|BK|BL|BR|WH|YL|GN)\s*\(\s*[A-Za-z ]+\s*\))\s*(UV Resistant)\b",
                    r"\2 \1",
                    tcur
                )

                # case: "UV Resistant RD (Red)" sudah benar, biarkan
                rms[i]["type"] = tcur.strip()

                break

        return rms


    def attach_parentheses_note_to_rm(rms: List[Dict], section_text: str) -> List[Dict]:
        """
        Tempel catatan dalam kurung (mis: "(Helically)") ke raw material type yang ada di section_text.
        Contoh: "Galv. Steel Tape ( Helically )" -> type jadi "Galv. Steel Tape (Helically)"
        """
        t = norm(section_text)

        for rm in rms:
            base = (rm.get("type") or "").strip()
            if not base:
                continue

            # cari "( ... )" yang muncul setelah nama type
            m = re.search(rf"(?i)\b{re.escape(base)}\b\s*\(\s*([A-Za-z][A-Za-z \-\/]*)\s*\)", t)
            if m:
                note = m.group(1).strip()
                suffix = f"({note})"
                if suffix.lower() not in base.lower():
                    rm["type"] = (base + " " + suffix).strip()

        return rms


    def merge_percent_continuations(rms: List[Dict], items: List[str]) -> List[Dict]:
        """
        Tempel persen ke RM yang tepat (filler -> RM mengandung filler).
        Ambil persen hanya dari item yang relevan, dan pilih persen terbesar (hindari 6% nyasar).
        """
        if not rms:
            return rms

        for raw in items:
            t = norm(raw)
            if not t:
                continue

            tl = t.lower()
            if "filler" not in tl:
                continue  # ✅ hanya proses baris filler

            # ambil semua persen yang ada di baris ini
            percents = re.findall(r"\(\s*(\d+(?:[.,]\d+)?)\s*%\s*\)", t)
            if not percents:
                continue

            # pilih persen terbesar (biasanya 60 bukan 6)
            vals = []
            for p in percents:
                try:
                    vals.append(float(p.replace(",", ".")))
                except:
                    pass
            if not vals:
                continue

            pct_val = max(vals)
            pct_str = str(int(pct_val)) if pct_val.is_integer() else f"{pct_val}".rstrip("0").rstrip(".")
            suffix = f"({pct_str}%)"

            # target: RM terakhir yang mengandung filler
            target_idx = None
            for i in range(len(rms) - 1, -1, -1):
                if "filler" in (rms[i].get("type") or "").lower():
                    target_idx = i
                    break
            if target_idx is None:
                target_idx = len(rms) - 1

            base = (rms[target_idx].get("type") or "").strip()
            # hapus persen lama kalau ada (biar gak dobel/nyangkut 6%)
            base = re.sub(r"\(\s*\d+(?:[.,]\d+)?\s*%\s*\)", "", base).strip()
            rms[target_idx]["type"] = (base + " " + suffix).strip()

        return rms




    def attach_percent_to_rm(rms: List[Dict], section_text: str) -> List[Dict]:
        """
        Kalau di section_text ada pattern: "<TYPE> ( 60 % )" atau "<TYPE> (60%)"
        maka tempel "(60%)" ke type raw material yang sama.
        """
        t = norm(section_text)

        for rm in rms:
            base = (rm.get("type") or "").strip()
            if not base:
                continue

            # cari persentase tepat setelah type
            m = re.search(
                rf"(?i)\b{re.escape(base)}\b\s*\(\s*(\d+(?:[.,]\d+)?)\s*%\s*\)",
                t
            )
            if m:
                pct = m.group(1).replace(",", ".").rstrip("0").rstrip(".")
                suffix = f"({pct}%)"
                if suffix.lower() not in base.lower():
                    rm["type"] = (base + " " + suffix).strip()

        return rms


    def merge_core_identification_colors(specs: List[Dict]) -> List[Dict]:
        colors = {
            "brown","black","grey","gray","red","blue","yellow","green","white",
            "orange","purple","violet","pink"
        }

        def norm_color_word(w: str) -> Optional[str]:
            if not w:
                return None
            wl = w.strip().lower().strip(" ,.;:")
            if wl == "gray":
                wl = "grey"
            if wl in colors:
                # output pakai kapital awal, Grey khusus
                return "Grey" if wl == "grey" else wl.capitalize()
            return None

        def extract_colors_from_text(txt: str) -> List[str]:
            t = (txt or "").strip()
            if not t:
                return []
            # pecah pakai koma / spasi
            parts = re.split(r"[,\s]+", t)
            out = []
            for p in parts:
                c = norm_color_word(p)
                if c and c not in out:
                    out.append(c)
            return out

        def is_stop_row(txt: str) -> bool:
            t = (txt or "").strip().lower()
            return bool(re.search(r"\b(outer\s+diameter|min\.?\s*|nom\.?\s*thickness|od/lp/dl|no\./|no\.|approx)\b", t))

        out = []
        i = 0
        while i < len(specs):
            cur = dict(specs[i])
            desc = (cur.get("description") or "").strip()

            if desc.lower() == "core identification":
                collected = []

                # 1) ambil dari specified kalau sudah ada
                collected += extract_colors_from_text(cur.get("specified") or "")

                # 2) serap beberapa baris setelahnya sampai ketemu row lain
                j = i + 1
                while j < len(specs):
                    nxt = specs[j] or {}
                    nd = (nxt.get("description") or "").strip()
                    nu = (nxt.get("unit") or "-").strip()
                    nspec = (nxt.get("specified") or "-").strip()

                    # stop kalau sudah ketemu baris row lain yang jelas
                    if is_stop_row(nd):
                        break

                    # kumpulkan warna dari description & specified baris itu
                    got = []
                    got += extract_colors_from_text(nd)
                    if nspec not in {"-", "", None}:
                        got += extract_colors_from_text(nspec)

                    # kalau baris itu tidak mengandung warna sama sekali,
                    # tapi unit/spec juga kosong -> lanjut 1 baris lagi (kadang pecah)
                    if not got:
                        # kalau ini baris "kosong" (unit '-' dan specified '-') kita boleh skip
                        if (nu == "-" and nspec in {"-", "", None}):
                            j += 1
                            continue
                        # kalau bukan kosong tapi tidak ada warna, stop supaya tidak nyedot row lain
                        break

                    for c in got:
                        if c not in collected:
                            collected.append(c)

                    j += 1

                # rapihin hasil
                if collected:
                    cur["specified"] = ", ".join(collected)
                else:
                    cur["specified"] = (cur.get("specified") or "-").strip().rstrip(" ,") or "-"

                cur["unit"] = "-"
                out.append(cur)

                # skip baris-baris yang sudah diserap kalau memang itu “lanjutan warna”
                i = j
                continue

            out.append(cur)
            i += 1

        return out

    def _parse_qty_float(q: str) -> Optional[float]:
        q = (q or "").strip()
        if not q:
            return None
        try:
            return float(q.replace(",", "."))
        except:
            return None

    def _looks_like_raw_material(type_text: str) -> bool:
        tl = (type_text or "").lower()
        block = [
            "thickness", "diameter", "resistance", "test", "identification", "construction",
            "shape", "marking", "tensile", "elongation", "voltage", "weight", "length",
            "drum", "cap", "od/lp/dl", "min.", "max."
        ]
        return bool(type_text) and not any(k in tl for k in block)

    def add_rm(ttype: str, qty: float, unit: Optional[str] = None):
        if not ttype or qty is None:
            return

        t_clean = norm(ttype)

        # ✅ normalisasi khusus Binder Tape (warna jangan masuk ke type)
        if re.search(r"(?i)\bbinder\s+tape\b", t_clean):
            t_clean = "Binder Tape"

        if not _looks_like_raw_material(t_clean):
            return

        raw_materials.append({
            "type": t_clean,
            "quantity": float(qty),
            "unit": unit or guessed_unit
        })


    def dedup_rm(rms: List[Dict]) -> List[Dict]:
        ded = {}
        for rm in rms:
            k = (rm["type"].lower(), float(rm["quantity"]))
            ded[k] = rm
        return list(ded.values())

    def _qty_regex(qty: float) -> str:
        # contoh 51.5 -> 51[.,]5(0*) ; 207.3 -> 207[.,]3(0*)
        s = f"{qty:.6f}".rstrip("0").rstrip(".")
        if "." in s:
            a, b = s.split(".")
            return rf"{re.escape(a)}[.,]{re.escape(b)}0*"
        return re.escape(s)

    def strip_rm_from_text(s: Optional[str]) -> Optional[str]:
        """Hapus 'TYPE + QTY' yang kebawa di specified (support koma/titik)."""
        if not s:
            return s
        out = s

        for rm in raw_materials:
            t = re.escape(rm["type"])
            qre = _qty_regex(float(rm["quantity"]))

            out = re.sub(rf"(?i)\b{t}\b\s+{qre}\b", "", out).strip()
            out = re.sub(r"(?i)\bkg\s*/\s*km\b|\bkg/km\b", "", out).strip()

        return out if out else None

    def merge_orphan_spec_rows(specs: List[Dict]) -> List[Dict]:
        """
        Kalau ada row pendek yang harusnya jadi specified baris sebelumnya.
        Contoh:
          Conductor shape (-,-)
          Round Stranded (-,-)
        => Conductor shape specified = "Round Stranded"
        """
        out = []
        i = 0
        while i < len(specs):
            cur = dict(specs[i])

            if i + 1 < len(specs):
                nxt = specs[i + 1] or {}
                nxt_desc = (nxt.get("description") or "").strip()
                nxt_unit = (nxt.get("unit") or "-").strip()
                nxt_spec = (nxt.get("specified") or "-").strip()

                cur_unit = (cur.get("unit") or "-").strip()
                cur_spec = (cur.get("specified") or "-").strip()

                looks_like_value_line = (
                    nxt_desc
                    and nxt_unit == "-"
                    and nxt_spec == "-"
                    and (len(nxt_desc.split()) <= 5 or re.fullmatch(r"[A-Z0-9 \/\.\-\(\)<>]+", nxt_desc) is not None)
                )

                can_absorb = (cur_unit == "-" and cur_spec in ("-", "", None))

                if looks_like_value_line and can_absorb:
                    cur["specified"] = nxt_desc
                    out.append(cur)
                    i += 2
                    continue

            out.append(cur)
            i += 1

        return out
    
    def merge_rm_type_continuations(rms: List[Dict], items: List[str]) -> List[Dict]:
        """
        Gabungkan baris lanjutan TYPE ke raw_material terakhir.
        Aman: hanya merge baris yang memang continuation raw material (contoh: ada 'PEROXIDE').
        """
        if not rms:
            return rms

        for raw in items:
            t = norm(raw)
            if not t:
                continue

            # baris continuation biasanya:
            # - tidak ada unit
            # - tidak ada angka
            

            # ✅ kalau baris lanjutan bentuknya persen dalam kurung, tempel ke type terakhir
            if re.fullmatch(r"\(\s*\d+(?:[.,]\d+)?\s*%\s*\)", t):
                last = rms[-1]
                last_type = (last.get("type") or "").strip()
                if t not in last_type:
                    last["type"] = (last_type + " " + t).strip()
                continue
            if find_unit(t):
                continue
            tl = t.lower()

            # ✅ kunci utama: hanya yang mengandung "peroxide"
            if "peroxide" not in tl:
                continue

            last = rms[-1]
            last_type = (last.get("type") or "").strip()

            # hindari dobel tempel
            if tl in last_type.lower():
                continue

            last["type"] = (last_type + " " + t).strip()


        return rms
    
    def _fmt_qty(q) -> str:
        # 2.0 -> "2", 2.64 -> "2.64"
        try:
            f = float(q)
            if f.is_integer():
                return str(int(f))
            s = f"{f:.6f}".rstrip("0").rstrip(".")
            return s
        except:
            return str(q)
    
    def merge_peroxide_rm(rms: List[Dict]) -> List[Dict]:
        """
        Gabungkan item raw_material bertipe PEROXIDE ke item sebelumnya,
        tapi angka yang kebaca sebagai quantity PEROXIDE dipindah jadi bagian TYPE.
        Contoh:
        {"type":"XLPE","quantity":503.6} + {"type":"PEROXIDE","quantity":2}
            => {"type":"XLPE PEROXIDE 2","quantity":503.6}
        """
        if not rms:
            return rms

        out: List[Dict] = []
        for rm in rms:
            t = (rm.get("type") or "").strip()
            tl = t.lower().strip()
            qty = rm.get("quantity", None)

            # deteksi item "peroxide" yang berdiri sendiri
            is_peroxide_only = tl in {"peroxide", "is peroxide", "os peroxide"} or tl.endswith(" peroxide")

            if is_peroxide_only and out:
                prev = out[-1]
                prev_type = (prev.get("type") or "").strip()

                addon = t  # "PEROXIDE" / "IS PEROXIDE"
                # kalau peroxide kebaca punya quantity (mis: 2), angka itu harus ikut TYPE
                if qty is not None and not re.search(r"\d", addon):
                    addon = f"{addon} {_fmt_qty(qty)}"

                # tempel (pakai spasi, bukan newline)
                if addon.lower() not in prev_type.lower():
                    prev["type"] = (prev_type + " " + addon).strip()

                # buang item peroxide-nya
                continue

            out.append(rm)

        return out


    # -----------------------------
    # 1) PRE-EXTRACT raw materials (CONDUCTOR friendly)
    # -----------------------------
    rm_unit_pat = re.compile(
        r"(?i)\b([A-Za-z][A-Za-z0-9\.\(\)/% \-]{2,120}?)\s+(\d+(?:[.,]\d+)?)\s*(kg\s*/\s*km|kg/km)\b"
    )
    for m in rm_unit_pat.finditer(section_text):
        ttype = norm(m.group(1))
        qty = _parse_qty_float(m.group(2))
        if ttype and qty is not None and _looks_like_raw_material(ttype):
            add_rm(ttype, qty, "kg/km")

    wire_pat = re.compile(r"(?i)\b([A-Za-z][A-Za-z0-9\.\(\)/% \-]{2,120}?\bWire)\s+(\d+(?:[.,]\d+)?)\b")
    best_wire = None
    best_qty = None
    best_val = -1.0
    for m in wire_pat.finditer(section_text):
        ttype = norm(m.group(1))
        qty = _parse_qty_float(m.group(2))
        if not ttype or qty is None:
            continue
        if qty > best_val and _looks_like_raw_material(ttype):
            best_val = qty
            best_wire = ttype
            best_qty = qty
    if best_wire and best_qty is not None:
        add_rm(best_wire, best_qty, guessed_unit)


        # ✅ raw material dengan unit area (m²) untuk WBT
    rm_area_pat = re.compile(
        r"(?i)\b([A-Za-z][A-Za-z0-9\.\(\)/% \-]{2,160}?)\s+(\d+(?:[.,]\d+)?)\s*(m²|m2|m\^2)\b"
    )
    for m in rm_area_pat.finditer(section_text):
        ttype = norm(m.group(1))
        qty = _parse_qty_float(m.group(2))
        if qty is None or not ttype:
            continue

        # normalisasi unit ke m²
        add_rm(ttype, qty, "m²")


    # ✅ fallback: tangkap "TYPE 128.27" meskipun tanpa unit kg/km
    # contoh: "Copper Tape 195.70" / "PVC Sheath 339" / "Steel Tape 12.5"
    rm_noun_pat = re.compile(
        r"(?i)\b([A-Za-z][A-Za-z0-9\.\(\)/% \-]{2,120}?)\s+(\d+(?:[.,]\d+)?)\b"
    )

    # keyword supaya tidak ketangkep angka spek
    MATERIAL_HINTS = [
        "pvc", "pe", "xlpe", "epr", "rubber", "compound", "insulation", "sheath", "screen",
        "shield", "shielding", "conductor", "wire", "tape", "copper", "aluminium", "aluminum",
        "steel", "armour", "armor", "yarn", "binder", "bedding", "polyester", "mylar",
        "carbon black", "sylane", "semiconductive", "semicon", "mica","wbt", "non-conductive", "nonconductive"
    ]

    # unit/spek yang harus di-skip
    SPEC_UNITS_HINT = ["mm", "kv", "ohm", "n/mm", "kg/mm", "kv/min", "ohm/km", "ohm.mm"]

    for m in rm_noun_pat.finditer(section_text):
        ttype = norm(m.group(1))
        qty = _parse_qty_float(m.group(2))
        if not ttype or qty is None:
            continue

        tl = ttype.lower()

        # skip kalau kelihatan ini baris spec (bukan material)
        if any(u in tl for u in SPEC_UNITS_HINT):
            continue
        if not _looks_like_raw_material(ttype):
            continue

        # ✅ kalau qty-nya cuma kode 2-3 digit (contoh 098) dan type mengandung filler,
        # jangan dianggap quantity (itu bagian type)
        qty_raw = m.group(2)
        if re.fullmatch(r"\d{2,3}", qty_raw) and "filler" in tl:
            continue

        qty_raw = m.group(2)

        # ✅ kalau angka 2-3 digit itu kemungkinan kode type (704, 098, dst), jangan jadi quantity palsu
        if re.fullmatch(r"\d{2,3}", qty_raw) and re.search(r"(?i)\b(filler|sheath|insulation)\b", ttype):
            continue

        # wajib ada hint material biar gak nangkep "Nom thickness 3.0"
        if any(h in tl for h in MATERIAL_HINTS):
            add_rm(ttype, qty, guessed_unit)


    raw_materials = dedup_rm(raw_materials)
   


    # -----------------------------
    # 2) RM from specified tail (COVERING friendly)
    # -----------------------------
    RM_FROM_SPEC = re.compile(
        r"""(?ix)
        ^\s*
        (?P<spec>\d+(?:[.,]\d+)?)\s+
        (?P<type>[A-Za-z][A-Za-z0-9\.\(\)/% \-]{2,120}?)\s+
        (?P<qty>\d+(?:[.,]\d+)?)
        \s*(?P<unit>kg\s*/\s*km|kg/km)?\s*$
        """
    )

    # -----------------------------
    # 3) Parse items -> technical_specs (+ layers)
    # -----------------------------
        # ✅ FIX: raw material kolom kanan sering nempel di ujung baris spec (tanpa "kg/km")
        # contoh: "... mm 35 / 2,14 ± 0,01  Tinned Annealed Copper Tape 3799,1"
    TRAIL_RM_RE = re.compile(
            r"""(?ix)
            \b(
                [A-Za-z][A-Za-z0-9\.\(\)/% \-]{2,160}?
                (?:tape|wire|yarn|compound|screen|sheath|insulation)
            )\s+
            (\d+(?:[.,]\d+)?)
            \s*$
            """
        )
        
    MATERIAL_HINTS2 = [
            "pvc","pe","xlpe","epr","rubber","compound","insulation","sheath","screen",
            "shield","shielding","conductor","wire","tape","copper","aluminium","aluminum",
            "steel","armour","armor","yarn","binder","bedding","polyester","mylar",
            "carbon black","sylane","semiconductive","semicon","mica",
            "tinned","annealed","wbt","non-conductive","nonconductive" # ✅ tambah ini
        ]
    
    pending_rm_suffix = None  # ✅ nyimpen "Copper Tape" kalau kebaca duluan

    has_copper_tape = (
        re.search(r"(?i)\bcopper\b", section_text) is not None
        and re.search(r"(?i)\btape\b", section_text) is not None
    )

    for raw in items:
        r = norm(raw)
        if not r:
            continue

        # A) tangkap Copper Tape (atau pecah) - HARUS ketat (jangan makan technical spec)
        if re.fullmatch(r"(?i)copper\W*tape", r.strip()):
            pending_rm_suffix = "Copper Tape"
            continue
        if re.fullmatch(r"(?i)copper", r.strip()):
            pending_rm_suffix = "Copper"
            continue
        if pending_rm_suffix == "Copper" and re.fullmatch(r"(?i)tape", r.strip()):
            pending_rm_suffix = "Copper Tape"
            continue

        # ✅ B) FALLBACK: "Tinned Annealed 3799,1" (tanpa Copper Tape)
        m_ta = re.search(r"(?i)\b(tinned\s+annealed)\s+(\d+(?:[.,]\d+)?)\s*$", r)
        if m_ta:
            qty = _parse_qty_float(m_ta.group(2))
            if qty is not None:
                rm_type = "Tinned Annealed"

                if pending_rm_suffix:
                    rm_type = f"{rm_type} {pending_rm_suffix}".strip()
                    pending_rm_suffix = None
                elif has_copper_tape:
                    rm_type = f"{rm_type} Copper Tape"

                add_rm(rm_type, qty, guessed_unit)
                r = norm(r[:m_ta.start()].strip())


        # ✅ C) trailing RM general (kalau ada)
        mtr = TRAIL_RM_RE.search(r)
        if mtr:
            rm_type = norm(mtr.group(1))
            rm_qty = _parse_qty_float(mtr.group(2))
            if rm_type and rm_qty is not None and any(h in rm_type.lower() for h in MATERIAL_HINTS2):
                if pending_rm_suffix and pending_rm_suffix.lower() not in rm_type.lower():
                    rm_type = f"{rm_type} {pending_rm_suffix}".strip()
                    pending_rm_suffix = None

                add_rm(rm_type, rm_qty, guessed_unit)
                r = norm(r[:mtr.start()].strip())

        
        # ✅ continuation raw material: baris cuma "(...)" seperti "(Helically)" -> tempel ke raw_material terakhir
        if raw_materials:
            if re.fullmatch(r"\(\s*[A-Za-z][A-Za-z \-\/]*\s*\)", r):
                last_type = (raw_materials[-1].get("type") or "").strip()
                if r.lower() not in last_type.lower():
                    raw_materials[-1]["type"] = (last_type + " " + r).strip()
                continue

        # ✅ lanjutan TYPE untuk sheath/insulation/filler: UV Resistant (berbagai variasi)
        if raw_materials:
            last_type = (raw_materials[-1].get("type") or "").strip()
            if re.search(r"(?i)\b(filler|sheath|insulation)\b", last_type):

                r_low = r.lower()

                # variasi: "UV Resistant" / "UV resistant" / "UV" "Resistant"
                if (("uv" in r_low and "resist" in r_low) or re.fullmatch(r"(?i)uv", r) or re.fullmatch(r"(?i)resistant", r)):
                    # jangan tempel kalau ada angka/unit (biar gak salah)
                    if not re.search(r"\d", r) and not find_unit(r):
                        if "uv resistant" not in last_type.lower():
                            raw_materials[-1]["type"] = (last_type + " UV Resistant").strip()
                            # ✅ TAROK DI SINI (langsung setelah nempel UV Resistant)
                            tcur = raw_materials[-1]["type"]
                            raw_materials[-1]["type"] = re.sub(
                                r"(?i)\b(RD\s*\(\s*Red\s*\))\s*(UV Resistant)\b",
                                r"\2 \1",
                                tcur
                            ).strip()
                        continue

        # ✅ Jangan masukin baris Marking ke technical_specs section mana pun
        if re.search(r"(?i)\bmarking\b", r) and (
            "marking of cable" in r.lower()
            or "cable marking" in r.lower()
        ):
            continue
        
        # ✅ lanjutan TYPE raw material untuk sheath/filler/insulation (UV Resistant / RD (Red) / BK (Black), dll)
        if raw_materials:
            last_type = (raw_materials[-1].get("type") or "").strip()

            # hanya tempel kalau raw material terakhir memang kategori sheath/filler/insulation
            if re.search(r"(?i)\b(filler|sheath|insulation)\b", last_type):

                # (1) UV Resistant
                if re.fullmatch(r"(?i)uv\s+resistant", r):
                    if "uv resistant" not in last_type.lower():
                        raw_materials[-1]["type"] = (last_type + " " + r).strip()
                    continue

                # (2) Kode warna seperti RD (Red), BK (Black), dll (jika baris berdiri sendiri)
                if re.fullmatch(r"(?i)\b[A-Z]{1,3}\s*\(\s*[A-Za-z ]+\s*\)\b", r):
                    if r.lower() not in last_type.lower():
                        raw_materials[-1]["type"] = (last_type + " " + r).strip()
                    continue

        # ✅ TAROK DI SINI (global untuk semua section)
        if raw_materials:
            color_line = r.strip()
            if re.fullmatch(r"(?i)\b[A-Z]{1,3}\s*\(\s*[A-Za-z]+\s*\)\b", color_line):
                last_type = (raw_materials[-1].get("type") or "").strip()
                if re.search(r"(?i)\b(filler|sheath|insulation)\b", last_type):
                    if color_line.lower() not in last_type.lower():
                        raw_materials[-1]["type"] = (last_type + " " + color_line).strip()
                    continue



        # ✅ SPECIAL: n/denier (jangan ketipu persen %)
        if re.search(r"(?i)\bn\s*/\s*denier\b", r):
            # buang persen dalam kurung kalau ada: "( 60 % )"
            r_clean = re.sub(r"\(\s*\d+(?:[.,]\d+)?\s*%\s*\)", "", r).strip()

            # target format: "<desc> n/denier <specified...>"
            m = re.search(r"(?i)^(.*?\bn\s*/\s*denier\b)\s+(.+)$", r_clean)
            if m:
                left = norm(m.group(1))   # contoh: "Centre filler n/denier" atau "Filler n/denier"
                right = norm(m.group(2))  # contoh: "1 / (1 x 100,000)" atau "3 / (9 x ...)"

                  # ✅(buang PP Yarn Filler 314,0 dari specified)
                right = re.sub(r"(?i)\bPP\s+Yarn\s+Filler\b\s+\d+(?:[.,]\d+)?\b", "", right).strip()
                right = right.strip(" :-/")

                 # kalau right cuma angka, coba ambil kurung pertama "( ... )" dari baris ini (kalau ada)
                if re.fullmatch(r"\d+(?:[.,]\d+)?", right or ""):
                    mparen = re.search(r"(\(.+?\))", r_clean)
                    if mparen:
                        right = f"{right} / {mparen.group(1)}"
                
                # pisahin desc dan unit
                desc_part = re.sub(r"(?i)\bn\s*/\s*denier\b", "", left).strip(" :-/") or left
                technical_specs.append({
                    "description": desc_part,
                    "unit": "n/denier",
                    "specified": right if right else "-"
                })
                continue


        unit = find_unit(r)
        desc = r
        specified = None

        if unit:
            mm = re.search(rf"(?<!\w){re.escape(unit)}(?!\w)", r)
            if mm:
                desc = r[:mm.start()].strip()
                specified = r[mm.end():].strip() or None
        else:
            mnum = re.search(r"\b\d", r)
            if mnum:
                left = r[:mnum.start()].strip()
                right = r[mnum.start():].strip()
                if left:
                    desc = left
                    specified = right

        desc = desc.strip(" :-")
        unit_out = unit if unit else "-"

        # ✅ Nom thickness: angka kadang nyasar / di baris lain
        if re.search(r"(?i)\bnom\.?\s*thickness\b", desc) and (specified is None or specified.strip() in {"", "-", "—"}):
            # coba cari angka murni di baris berikutnya
            nxt = _next_numeric_spec(lines, idx, max_hop=4)
            if nxt:
                k, val = nxt
                specified = val
                skip_until = max(skip_until, k)
            else:
                # fallback: cari angka dari gabungan beberapa baris tapi stop sebelum row lain
                win = _pick_nom_thickness_from_window(lines, idx, max_hop=6)
                if win:
                    k2, val2 = win
                    specified = val2
                    skip_until = max(skip_until, k2)

            technical_specs.append({
                "description": "Nom thickness",
                "unit": "mm" if unit_out == "-" else unit_out,
                "specified": specified if specified is not None else "-"
            })
            continue

        # ✅ CABLING: Centre filler biasanya "1 / Triangle" ada di baris berikutnya
        if re.search(r"(?i)\bcentre\s+filler\b", desc) and (specified is None or specified.strip() in {"", "-", "—"}):
            ratio = _next_centre_filler_ratio(lines, idx, max_hop=4)
            if ratio:
                specified = ratio
            unit_out = "n/-"  # sesuai tabel PDF

            technical_specs.append({
                "description": "Centre filler",
                "unit": unit_out,
                "specified": specified if specified is not None else "-"
            })
            continue

        if specified is None and re.search(r"(?i)\bok\b$", r):
            specified = "Ok"
            desc = re.sub(r"(?i)\bok\b$", "", r).strip(" :-")
            unit_out = "-"

        # extract RM dari specified (covering case)
        if specified:
            mrm = RM_FROM_SPEC.match(specified)
            if mrm:
                spec_num = mrm.group("spec")
                ttype = mrm.group("type")
                qty = _parse_qty_float(mrm.group("qty"))
                u = mrm.group("unit")
                add_rm(ttype, qty, "kg/km" if u else guessed_unit)
                raw_materials = dedup_rm(raw_materials)
                specified = spec_num.replace(",", ".")

        # strip RM yang nyangkut (conductor case)
        specified = strip_rm_from_text(specified)

        # ✅ pindahin kode warna seperti "BK ( Black )" dari specified ke raw_material terakhir (mis. PVC Filler 098)
        # berlaku semua section
        if specified and raw_materials:
            mcol = re.search(r"(?i)\b([A-Z]{1,3})\s*\(\s*([A-Za-z ]+)\s*\)", specified)
            if mcol:
                color_txt = f"{mcol.group(1)} ({mcol.group(2).strip()})"  # contoh: "BK (Black)"
                last_type = (raw_materials[-1].get("type") or "").strip()

                # hanya tempel kalau raw material terakhir memang tipe filler/sheath/insulation
                if re.search(r"(?i)\b(filler|sheath|insulation)\b", last_type):
                    if color_txt.lower() not in last_type.lower():
                        raw_materials[-1]["type"] = (last_type + " " + color_txt).strip()

                    # hapus dari specified (biar thickness jadi murni angka)
                    specified = re.sub(r"(?i)\b[A-Z]{1,3}\s*\(\s*[A-Za-z ]+\s*\)", "", specified).strip()
                    specified = specified.strip(" :-/")

        

        # ✅ pindahin "IS/OS PEROXIDE" yang nyangkut di specified ke raw_material terakhir
        # contoh: specified = "0,5 IS PEROXIDE"  -> specified jadi "0,5", dan RM type jadi "Semiconductive\nIS PEROXIDE"
        if specified and re.search(r"(?i)\bperoxide\b", specified):
            m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s+(.+?\bperoxide\b.*)\s*$", specified, flags=re.I)
            if m:
                only_num = m.group(1)      # "0,5"
                tail = m.group(2).strip()  # "IS PEROXIDE"

                if raw_materials:
                    last_type = (raw_materials[-1].get("type") or "").strip()
                    if tail.lower() not in last_type.lower():
                        raw_materials[-1]["type"] = (last_type + " " + tail).strip()


                specified = only_num  # tinggal angkanya saja


        # ✅ jangan simpan baris "Semiconductive 47,0" ke technical_specs (itu raw material)
        # ini berlaku untuk semua section (bukan cuma core identification)
        if desc and specified:
            if re.fullmatch(r"\d+(?:[.,]\d+)?", specified.strip()):
                desc_lw = desc.strip().lower()
                for rm in raw_materials:
                    rm_type_lw = (rm.get("type") or "").strip().lower()
                    if rm_type_lw.startswith(desc_lw):   # ✅ Semiconductive match Semiconductive IS PEROXIDE
                        continue_flag = True
                        break
                else:
                    continue_flag = False

                if continue_flag:
                    continue

        if enable_layers:
            ly = extract_layers_from_odlpdl(desc, unit_out, specified or "")
            if ly:
                layers.extend(ly)
                continue

        # ✅ COVERING special: tangkap "Sylane 1.32" walau formatnya nempel di desc atau specified
        full_line = f"{desc} {specified or ''}".strip()
        if re.search(r"(?i)\bsylane\b", full_line):
            mqty = re.search(r"(\d+(?:[.,]\d+)?)\s*$", full_line)
            if mqty:
                q = _parse_qty_float(mqty.group(1))
                if q is not None:
                    add_rm("Sylane", q, guessed_unit)
                    raw_materials = dedup_rm(raw_materials)


        # ✅ Normalisasi "Core identification" agar sesuai tabel PDF
        desc_l = (desc or "").lower()
        if "core identification" in desc_l:
            # cari warna yang mungkin muncul di desc (contoh: Black)
            colors = ["black", "red", "blue", "yellow", "green", "white", "brown", "grey", "gray", "orange", "purple", "violet", "pink"]
            found_color = None
            for c in colors:
                if re.search(rf"(?i)\b{re.escape(c)}\b", desc or ""):
                    found_color = "Gray" if c == "gray" else c.capitalize()
                    break

            # kalau specified sekarang angka (mis 1.32), itu bukan specified core-id, itu qty Sylane
            if specified and re.fullmatch(r"\d+(?:[.,]\d+)?", specified.strip()):
                if found_color:
                    specified = found_color
                # rapihin description jadi hanya "Core identification"
                desc = "Core identification"
                unit_out = "-"  # biasanya "-" di tabel

            # kalau specified kosong tapi ada warna di desc, isi juga
            elif (not specified or specified in ("-", "")) and found_color:
                specified = found_color
                desc = "Core identification"
                unit_out = "-"

        # ✅ kalau baris ini nyebut Binder Tape dan ada angka -> anggap raw material, jangan jadi technical_specs
        if re.search(r"(?i)\bbinder\s+tape\b", desc) and specified and re.search(r"\d", specified):
            continue

        # ✅ khusus thickness: ambil angka pertama saja, buang BK(Black) dll
        if specified and unit_out == "mm" and re.search(r"(?i)\bthickness\b", desc):
            mnum = re.match(r"^\s*(\d+(?:[.,]\d+)?)", specified.strip())
            if mnum:
                specified = mnum.group(1)  # hasil: "1,09"

        # ✅ Bersihkan "m²" yang nyangkut di specified untuk unit n/mm (No./size)
        if specified and unit_out and unit_out.strip().lower() == "n/mm":
            specified = re.sub(r"(?i)\b(m²|m2|m\^2)\b", "", specified)
            specified = re.sub(r"\s+", " ", specified).strip()
            specified = specified.strip(" ,;")

        # ✅ Bersihkan specified untuk diameter: ambil angka pertama saja (buang '7 Bedding' dll)
        if specified and unit_out == "mm" and re.search(r"(?i)\bdiameter\b", desc):
            mnum = re.match(r"^\s*(\d+(?:[.,]\d+)?)", specified.strip())
            if mnum:
                specified = mnum.group(1)

        if desc:
            technical_specs.append({
                "description": desc,
                "unit": unit_out if unit_out else "-",
                "specified": specified if specified is not None else "-"
            })

        def inject_nom_thickness_if_missing(specs: List[Dict], section_text_: str):
            # kalau sudah ada, stop
            if any((s.get("description") or "").strip().lower() == "nom thickness" for s in specs):
                return specs

            st = norm(section_text_)

            # cari: "Nom thickness ... mm ... 3.0"
            m = re.search(r"(?i)\bnom\.?\s*thickness\b.*?\bmm\b\s*([0-9]+(?:[.,][0-9]+)?)\b", st)
            if not m:
                # fallback: "Nom thickness 3.0 mm"
                m = re.search(r"(?i)\bnom\.?\s*thickness\b.*?([0-9]+(?:[.,][0-9]+)?)\s*\bmm\b", st)

            if m:
                val = m.group(1).replace(",", ".")
                row = {"description": "Nom thickness", "unit": "mm", "specified": val}
                # taruh di awal supaya urutan sesuai tabel
                return [row] + specs

            return specs



    # ✅ tarok di sini (setelah loop)
    technical_specs = merge_orphan_spec_rows(technical_specs)
    technical_specs = dedup_technical_specs(technical_specs)
    technical_specs = inject_core_identification_from_section_text(technical_specs, section_text)
    technical_specs = merge_core_identification_colors(technical_specs)  # ✅ TAMBAH INI
    raw_materials = merge_percent_continuations(raw_materials, items)   # ✅ TAMBAH INI
    raw_materials = attach_percent_to_rm(raw_materials, section_text)   # ✅ TAMBAH INI
    technical_specs = inject_nom_thickness_if_missing(technical_specs, section_text)
    raw_materials = attach_uv_resistant_from_section_text(raw_materials, section_text)
    raw_materials = attach_parentheses_note_to_rm(raw_materials, section_text)  # ✅ TAMBAH INI
    raw_materials = merge_rm_type_continuations(raw_materials, items)
    raw_materials = merge_peroxide_rm(raw_materials)     # ✅ TAMBAH DI SINI
    raw_materials = dedup_rm(raw_materials)
    raw_materials = fix_tinned_annealed_copper_tape(raw_materials, section_text)

        # ✅ FIX unit WBT: kalau type mengandung WBT, unit harus m² (bukan kg/km)
    for rm in raw_materials:
        t = (rm.get("type") or "").lower()
        if "wbt" in t:
            rm["unit"] = "m²"

    out = {"raw_materials": raw_materials, "technical_specs": technical_specs}
    if enable_layers:
        out["layers"] = layers
    return out


def cleanup_technical_specs_rows(rows):
    import re

    def norm(s):
        return re.sub(r"\s+", " ", (s or "")).strip()

    UNIT_TOKENS = [
        "n/-", "%/-", "mm/-", "pcs/dtex", "Ohm.mm²/km", "Ohm/km", "M.Ohm.km",
        "kV/min.", "kg/mm²", "kg/km", "mm", "m", "kg", "%", "-"
    ]

    cleaned = []
    i = 0
    while i < len(rows):
        r = rows[i] or {}
        desc = norm(r.get("description"))
        unit = norm(r.get("unit"))
        spec = norm(r.get("specified"))

        # 1) kalau unit '-' tapi di desc ada 'n/-' atau '%/-' dll, pindahin ke unit
        if unit == "-" and desc:
            m = re.search(r"(?i)\b(n\s*/\s*-\b|%\s*/\s*-\b|mm\s*/\s*-\b|pcs\s*/\s*dtex\b)\s*$", desc)
            if m:
                unit = re.sub(r"\s+", "", m.group(1)).lower()  # n/- atau %/-
                desc = norm(desc[:m.start()])

        # 2) kalau unit '%' tapi desc mengandung "n/- ... ( ...", pisahin: description tetap, unit tetap, specified ambil sisanya
        #    contoh: "Filler n/- 3 / ( ... ) ( 60" unit="%" -> "Filler", unit="%", specified="3 / ( ... ) ( 60"
        if unit in {"%", "%/-"} and desc:
            # ambil pola "xxx n/- <rest>"
            m = re.match(r"(?is)^(.*?)(?:\bn\s*/\s*-\b)\s+(.*)$", desc)
            if m:
                desc = norm(m.group(1))
                rest = norm(m.group(2))
                # gabung ke specified (spec kadang cuma ")")
                spec = norm((rest + " " + spec).strip())

        # 3) kalau specified cuma ")" dan baris sebelumnya specified masih buka "(" atau berakhir "(" / "( 60"
        if spec == ")" and cleaned:
            prev = cleaned[-1]
            prev_spec = norm(prev.get("specified"))
            if prev_spec.count("(") > prev_spec.count(")") or prev_spec.endswith("(") or prev_spec.endswith("( 60"):
                prev["specified"] = norm(prev_spec + " )")
                i += 1
                continue

        # 4) rapihin token yang kepotong "n/" jadi "n/-" kalau unit "-"
        if unit == "-" and desc.endswith(" n/"):
            desc = norm(desc[:-2])  # buang "n/"
            unit = "n/-"

        cleaned.append({"description": desc, "unit": unit or "-", "specified": spec or "-"})
        i += 1

    return cleaned



def parse_final_test_v2(section_text: str) -> List[Dict]:
    """
    Parse Final Test dan STOP kalau sudah kebaca masuk section berikutnya.
    Fix:
    - merge orphan 'Ok' ke baris sebelumnya
    - potong text jika ketemu marker section lain (Packing/Marking/Conductor/etc)
    """
    def ns(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    # ✅ STOP markers: begitu ketemu ini, artinya sudah keluar dari Final Test
    STOP_PAT = re.compile(
        r"(?i)\b("
        r"packing|marking|cable\s+marking|marking\s+of\s+cable|"
        r"conductor\s*-\b|conductor\s+shielding|insulation\s*-\b|"
        r"inner\s+sheath|outer\s+sheath|armour|armour\s*-\b|"
        r"covering\s*-\b|tapping\s*-\b|twisting\s*-\b|"
        r"metallic\s+screen|overall\s+screen|individual\s+screen|cabling\s*-\b"
        r")\b"
    )

    # ✅ potong section_text jika ada stop marker
    mstop = STOP_PAT.search(section_text or "")
    if mstop:
        section_text = (section_text or "")[:mstop.start()].strip()

    out: List[Dict] = []
    items = split_items_safe(section_text)

    for raw in items:
        r = ns(raw)
        if not r or r in {"-", "–", "—"}:
            continue

        # ✅ kalau item kebaca jadi nomor section (contoh "1.", "2.", dst) -> stop
        if re.match(r"^\s*\d+\s*[\.\)]\s+", r):
            break

        unit = find_unit(r)
        test_name = r
        specified = None

        if unit:
            mm = re.search(rf"(?<!\w){re.escape(unit)}(?!\w)", r)
            if mm:
                test_name = r[:mm.start()].strip(" :-")
                specified = r[mm.end():].strip() or None
        else:
            # "Visual all layer Ok", "Dimension Ok", atau "Ok" sendiri
            if re.search(r"(?i)\bok\b$", r):
                specified = "Ok"
                test_name = re.sub(r"(?i)\bok\b$", "", r).strip(" :-")

        test_name = test_name.strip() if test_name else "-"
        unit_out = unit if unit else "-"
        specified_out = specified if specified else "-"

        # ✅ merge orphan "Ok" yang jadi baris sendiri
        if out:
            is_orphan = (test_name in {"-", ""}) and (unit_out in {"-", ""}) and (specified_out not in {"-", "", None})
            if is_orphan and out[-1].get("specified") in {"-", "", None}:
                out[-1]["specified"] = specified_out
                continue

        out.append({
            "test_name": test_name,
            "unit": unit_out,
            "specified": specified_out
        })

    return out



def parse_packing_as_object_v2(section_text: str) -> Dict:
    t = norm(section_text)

    def grab_value(label_pat: str, unit_pat: str):
        # support:
        # 1) "Label unit 300"
        # 2) "Label 300 unit"
        pats = [
            rf"(?i)(?:-\s*)?{label_pat}\s+{unit_pat}\s+([0-9]+(?:[.,][0-9]+)?)\b",
            rf"(?i)(?:-\s*)?{label_pat}\s+([0-9]+(?:[.,][0-9]+)?)\s+{unit_pat}\b",
        ]
        for p in pats:
            m = re.search(p, t)
            if m:
                return m.group(1)
        return None

    def to_int(x: str):
        return int(float(x.replace(",", ".")))

    obj: Dict = {}

    # ✅ bisa kebaca "Standard length m 300" atau "Standard length 300 m"
    standard = grab_value(r"Standard\s+length", r"m")
    netw     = grab_value(r"Net\.?\s*Weight", r"kg")
    gross    = grab_value(r"Gross\s+weight", r"kg")

    if standard:
        obj["standard_length"] = to_int(standard)
    if netw:
        obj["net_weight"] = to_int(netw)
    if gross:
        obj["gross_weight"] = to_int(gross)

    # tetap: drum & end cap (aku sekalian bikin endcap lebih fleksibel)
    drum_type = None
    m = re.search(r"(?i)\bWooden\s+Drum\s+(\d+)\b", t)
    if m:
        drum_type = m.group(1)

    drum_qty = None
    m = re.search(r"(?i)\bWooden\s+Drum\s+\d+\s+(\d+)\b", t)
    if m:
        drum_qty = m.group(1)

    endcap_qty = None
    # support: "Heat shrink end cap 4" / "End cap 4"
    m = re.search(r"(?i)\b(?:Heat\s+shrink\s+)?End\s+cap\s+(\d+)\b", t)
    if m:
        endcap_qty = m.group(1)

    if drum_type:
        obj["drum"] = "Wooden Drum"
        obj["drum_type"] = str(int(drum_type))
    if drum_qty:
        obj["drum_quantity"] = int(float(drum_qty.replace(",", ".")))
    if endcap_qty:
        obj["end_cap_quantity"] = int(float(endcap_qty.replace(",", ".")))

    return obj



# ----------------------------
# Revision History (V2)
# ----------------------------
DASH = r"[\-–—-]"
DATE_RE = rf"\d{{1,2}}{DASH}[A-Za-z]{{3}}{DASH}\d{{2}}"

ROW_FULL = re.compile(rf"^\s*(\d{{1,2}})\s+({DATE_RE})\s+(.+?)\s*$")
ROW_HEAD = re.compile(rf"^\s*(\d{{1,2}})\s+({DATE_RE})\s*$")

def build_revision_rows(rowlines: List[str]) -> List[Dict]:
    rows: List[Dict] = []
    last_desc = None
    current = None

    for raw in rowlines:
        line = (raw or "").strip()
        if not line:
            continue

        low = line.lower()
        if ("revision" in low and "level" in low and "date" in low and "description" in low):
            continue

        m = ROW_FULL.match(line)
        if m:
            level, date, desc = m.group(1), m.group(2), m.group(3).strip()
            iso = dd_mmm_yy_to_iso(date.replace("–","-").replace("—","-")) or date
            rows.append({
                "rev_level": level.zfill(2),
                "date": iso,
                "description": desc
            })
            current = rows[-1]
            last_desc = None
            continue

        m = ROW_HEAD.match(line)
        if m:
            level, date = m.group(1), m.group(2)
            desc = (last_desc or "").strip()
            iso = dd_mmm_yy_to_iso(date.replace("–","-").replace("—","-")) or date
            rows.append({
                "rev_level": level.zfill(2),
                "date": iso,
                "description": desc
            })
            current = rows[-1]
            last_desc = None
            continue

        if current is not None and (current.get("description") in ("", None)):
            current["description"] = line
        elif current is not None and current.get("description"):
            current["description"] = (current["description"] + " " + line).strip()
        else:
            last_desc = line

    # default description "-"
    for r in rows:
        if not r.get("description"):
            r["description"] = "-"
        if not r.get("date"):
            r["date"] = "-"
    return rows


def extract_cabling_centre_filler_from_pdf(pdf_path: str) -> str:
    """
    Ambil nilai Centre filler untuk Cabling dari layout words PDF.
    Target output: "1 / Triangle" (atau Round/Square)
    Return "" kalau tidak ketemu.
    """
    def ns(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return ""

    try:
        if doc.page_count == 0:
            return ""

        page = doc.load_page(0)
        page_w = float(page.rect.width)
        words = page.get_text("words") or []
        if not words:
            return ""

        # anchor "6 Cabling"
        cab_y = None
        for x0, y0, x1, y1, w, *_ in words:
            ww = ns(str(w))
            if re.fullmatch(r"(?i)6", ww):
                for x0b, y0b, x1b, y1b, wb, *_ in words:
                    if abs(float(y0b) - float(y0)) <= 2.5 and ns(str(wb)).lower() == "cabling":
                        cab_y = float(y0)
                        break
            if cab_y is not None:
                break
        if cab_y is None:
            return ""

        # anchor "7 Bedding" biar nggak nyasar ke section bawah
        next_y = None
        for x0, y0, x1, y1, w, *_ in words:
            ww = ns(str(w))
            if re.fullmatch(r"(?i)7", ww):
                for x0b, y0b, x1b, y1b, wb, *_ in words:
                    if abs(float(y0b) - float(y0)) <= 2.5 and ns(str(wb)).lower() == "bedding":
                        next_y = float(y0)
                        break
            if next_y is not None:
                break

        y_top = cab_y + 2
        y_bot = (next_y - 2) if next_y else (cab_y + 120)

        # group words jadi lines berdasarkan y
        lines = {}
        for x0, y0, x1, y1, w, *_ in words:
            ww = ns(str(w))
            if not ww:
                continue
            y0f = float(y0); y1f = float(y1)
            if y1f < y_top or y0f > y_bot:
                continue
            yk = round(y0f, 1)
            lines.setdefault(yk, []).append((float(x0), ww))

        if not lines:
            return ""

        ykeys = sorted(lines.keys())

        def line_text(yk: float) -> str:
            toks = [t[1] for t in sorted(lines[yk], key=lambda z: z[0])]
            return ns(" ".join(toks))

        # cari y baris yang mengandung "Centre filler"
        cf_y = None
        for yk in ykeys:
            t = line_text(yk).lower()
            if "centre" in t and "filler" in t:
                cf_y = yk
                break
        if cf_y is None:
            return ""

        # cari pola "1 / Triangle" di baris yang sama / baris setelahnya (kolom tengah)
        pat = re.compile(r"(?i)\b(\d+)\s*/\s*(triangle|round|square)\b")
        for yk in ykeys:
            if yk < cf_y - 2:
                continue
            if yk > cf_y + 40:  # cukup dekat
                break

            txt = line_text(yk)

            # batasi ke area kolom tengah biar nggak ke kanan (raw material) atau kiri (desc)
            # ini longgar aja: ambil kata dari x 0.32w..0.70w
            toks_mid = [ww for x, ww in sorted(lines[yk], key=lambda z: z[0])
                        if (page_w * 0.32) <= x <= (page_w * 0.70)]
            mid = ns(" ".join(toks_mid))

            m = pat.search(mid) or pat.search(txt)
            if m:
                return f"{m.group(1)} / {m.group(2).capitalize()}"

        return ""
    finally:
        doc.close()



def extract_revision_history_from_pdf(pdf_path: str, debug: bool = False) -> List[Dict]:
    def ns(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    def is_header_line(s: str) -> bool:
        t = (s or "").lower()
        return ("revision" in t and "level" in t and "date" in t and "description" in t)

    DATE_PAT = re.compile(r"^\d{1,2}-[A-Za-z]{3}-\d{2}$")
    LEVEL_PAT = re.compile(r"^\d{1,2}$")

    MONTH_MAP = {
        "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05", "JUN": "06",
        "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12"
    }

    def dd_mmm_yy_to_iso(s: str) -> str:
        s = ns(s).replace("–", "-").replace("—", "-")
        m = re.fullmatch(r"(\d{1,2})-([A-Za-z]{3})-(\d{2})", s)
        if not m:
            return s
        dd = m.group(1).zfill(2)
        mon = MONTH_MAP.get(m.group(2).upper())
        yy = int(m.group(3))
        yyyy = 2000 + yy if yy <= 50 else 1900 + yy
        return f"{yyyy}-{mon}-{dd}" if mon else s

    STOP_RE = re.compile(r"(?i)\b(prepared|checked|approved|design\s+change\s+notice|raw\s+material)\b")

    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return []

    try:
        for page_idx in range(doc.page_count):
            page = doc.load_page(page_idx)
            words = page.get_text("words") or []
            if not words:
                continue

            # group by y (line)
            lines: Dict[float, List[Tuple[float, str]]] = {}
            for x0, y0, x1, y1, w, *_ in words:
                w = str(w).strip()
                if not w:
                    continue
                yk = round(float(y0), 1)
                lines.setdefault(yk, []).append((float(x0), w))

            ordered = []
            for yk in sorted(lines.keys()):
                toks = sorted(lines[yk], key=lambda t: t[0])
                txt = ns(" ".join(t for _, t in toks))
                ordered.append((yk, toks, txt))

            # find header
            header_idx = None
            for i, (_, _, txt) in enumerate(ordered):
                if is_header_line(txt):
                    header_idx = i
                    break
            if header_idx is None:
                continue

            out: List[Dict] = []
            pending = None      # row yg butuh desc dari baris berikutnya (normal order)
            carry_desc = None   # desc yg muncul duluan sebelum row level+date (reverse order)
            empty_streak = 0

            for (yk, toks, txt) in ordered[header_idx + 1:]:
                if not txt:
                    empty_streak += 1
                    if empty_streak >= 10:
                        break
                    continue
                empty_streak = 0

                if STOP_RE.search(txt):
                    break

                tokens = [ns(t) for _, t in toks]
                tokens = [t for t in tokens if t]

                # cari token tanggal
                date_idx = None
                for i, tk in enumerate(tokens):
                    if DATE_PAT.match(tk):
                        date_idx = i
                        break

                if date_idx is not None:
                    # row baru (level+date)
                    level = None
                    for tk in tokens[:date_idx]:
                        if LEVEL_PAT.match(tk):
                            level = tk
                            break
                    if not level:
                        continue

                    date_raw = tokens[date_idx]
                    desc_inline = ns(" ".join(tokens[date_idx + 1:]))

                    row = {
                        "rev_level": level.zfill(2),
                        "date": dd_mmm_yy_to_iso(date_raw),
                        "description": desc_inline if desc_inline else "-"
                    }

                    # ✅ kalau inline desc kosong dan ada carry_desc (desc muncul duluan), pakai carry_desc
                    if row["description"] in ("-", "", None) and carry_desc:
                        row["description"] = carry_desc
                        carry_desc = None
                        pending = None
                    else:
                        # kalau masih kosong, pending untuk baris berikutnya
                        pending = row if row["description"] in ("-", "", None) else None

                    out.append(row)
                    continue

                # baris TANPA tanggal:
                if pending is not None:
                    # ✅ normal order: isi desc untuk row pending
                    pending["description"] = txt
                    pending = None
                else:
                    # ✅ reverse order: simpan buat row berikutnya
                    if carry_desc:
                        carry_desc = ns(carry_desc + " " + txt)
                    else:
                        carry_desc = txt

            # cleanup
            for r in out:
                if not r.get("description"):
                    r["description"] = "-"

            if out:
                return out

        return []
    finally:
        doc.close()

def attach_armour_helically_from_pdf(pdf_path: str, base_type: str) -> Optional[str]:
    """
    Khusus Armour: cari '(Helically)' dari PDF words (layout).
    Strategi:
    - cari baris yang memuat base_type (mis. 'Galv. Steel Tape')
    - ambil rentang x kolom TYPE dari token base_type
    - scan beberapa baris di bawahnya yang y dekat, ambil teks di rentang x itu
    - kalau ketemu kata 'Helically' -> return '(Helically)'
    """
    def ns(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return None

    try:
        if doc.page_count == 0:
            return None

        base_low = (base_type or "").lower()
        base_parts = [p for p in re.split(r"\s+", base_low) if p]  # ['galv.', 'steel', 'tape']

        for pidx in range(min(3, doc.page_count)):
            page = doc.load_page(pidx)
            words = page.get_text("words") or []
            if not words:
                continue

            # group words into lines by y
            lines: Dict[float, List[Tuple[float, float, str]]] = {}
            for x0, y0, x1, y1, w, *_ in words:
                w = str(w).strip()
                if not w:
                    continue
                yk = round(float(y0), 1)
                lines.setdefault(yk, []).append((float(x0), float(x1), w))

            y_keys = sorted(lines.keys())

            # helper: line text + tokens
            def get_line(yk: float):
                toks = sorted(lines[yk], key=lambda t: t[0])
                txt = ns(" ".join(t[2] for t in toks))
                return toks, txt

            for i, yk in enumerate(y_keys):
                toks, txt = get_line(yk)
                low = txt.lower()
                if not low:
                    continue

                # cari baris yang memuat base_type
                if base_low in low:
                    # tentukan x-range kolom type dari token yg match bagian base_type
                    # ambil token yang kira-kira match "galv", "steel", "tape"
                    x_min = None
                    x_max = None
                    for x0, x1, w in toks:
                        wl = w.lower()
                        if any(bp.replace(".", "") in wl.replace(".", "") for bp in base_parts):
                            x_min = x0 if x_min is None else min(x_min, x0)
                            x_max = x1 if x_max is None else max(x_max, x1)

                    if x_min is None or x_max is None:
                        continue

                    # scan beberapa baris setelahnya (y dekat) untuk cari "Helically"
                    for j in range(i + 1, min(i + 8, len(y_keys))):
                        yk2 = y_keys[j]
                        # stop kalau jaraknya terlalu jauh (keluar dari row material)
                        if (yk2 - yk) > 35:
                            break

                        toks2, txt2 = get_line(yk2)
                        if not txt2:
                            continue

                        # ambil teks di kolom type (x dekat dengan x_min..x_max)
                        band = []
                        for x0, x1, w in toks2:
                            if x0 >= (x_min - 10) and x0 <= (x_max + 120):
                                band.append((x0, w))
                        band_txt = ns(" ".join(w for _, w in sorted(band, key=lambda t: t[0])))

                        if "helically" in band_txt.lower():
                            return "(Helically)"

        return None
    finally:
        doc.close()


def extract_first_raw_material_from_pdf(pdf_path: str) -> Optional[Dict]:
    """
    Ambil entry RAW MATERIAL pertama (TYPE + QUANTITY) dari tabel di page 0.
    Return: {"type": "...", "quantity": 3799.1, "unit": "kg/km"} atau None.

    Strategy:
    - Temukan header RAW MATERIAL + kolom QUANTITY
    - Cari angka qty pertama (kolom QUANTITY) -> dapat row_y
    - Ambil kata-kata material di sekitar row_y dengan keyword filter (tinned/annealed/copper/tape/etc)
      supaya "Tinned Annealed" + "Copper Tape" kebaca walau posisi kata geser.
    """
    def ns(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    def is_num(s: str) -> bool:
        return re.fullmatch(r"\d+(?:[.,]\d+)?", s or "") is not None

    def to_float(s: str) -> Optional[float]:
        try:
            return float(s.replace(",", "."))
        except:
            return None

    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return None

    try:
        if doc.page_count == 0:
            return None

        page = doc.load_page(0)
        page_w = float(page.rect.width)
        words = page.get_text("words") or []
        if not words:
            return None

        # cari bbox kata target (exact token)
        def find_word_bbox(target: str, y_min=None, y_max=None):
            tgt = target.lower()
            best = None
            for x0, y0, x1, y1, w, *_ in words:
                ww = str(w).strip().lower()
                if ww != tgt:
                    continue
                y0f = float(y0); y1f = float(y1)
                if y_min is not None and y1f < y_min:
                    continue
                if y_max is not None and y0f > y_max:
                    continue
                cand = (y0f, float(x0), float(x1), y1f)
                if best is None or cand[0] < best[0]:
                    best = cand
            return best

        # 1) header RAW MATERIAL
        bb_raw = find_word_bbox("raw")
        bb_mat = find_word_bbox("material", y_min=bb_raw[0]-5, y_max=bb_raw[3]+40) if bb_raw else None
        if not bb_raw or not bb_mat:
            return None

        header_y1 = max(bb_raw[3], bb_mat[3])

        # 2) header TYPE & QUANTITY
        bb_type = find_word_bbox("type", y_min=header_y1, y_max=header_y1 + 140)
        bb_qty  = find_word_bbox("quantity", y_min=header_y1, y_max=header_y1 + 140)
        if not bb_type or not bb_qty:
            return None

        y_start = max(bb_type[3], bb_qty[3]) + 2

        # kolom qty kira-kira dari x header qty sampai kanan halaman
        x_qty0 = bb_qty[1] - 10
        x_qty1 = page_w - 5

        # 3) cari qty pertama (angka pertama di kolom QUANTITY setelah header)
        qty_tokens = []
        for x0, y0, x1, y1, w, *_ in words:
            x0f = float(x0); y0f = float(y0); y1f = float(y1)
            ww = str(w).strip()
            if not ww:
                continue
            if x0f < x_qty0 or x0f > x_qty1:
                continue
            if y1f < y_start or y0f > (y_start + 280):
                continue
            if is_num(ww):
                qty_tokens.append((y0f, x0f, ww))

        if not qty_tokens:
            return None

        qty_tokens.sort(key=lambda t: (t[0], t[1]))
        row_y = qty_tokens[0][0]
        qty_val = to_float(qty_tokens[0][2])
        if qty_val is None:
            return None

        # 4) ambil kandidat TYPE di sekitar row_y (lebih lebar), tapi hanya keyword material
        # x band: mulai dari sekitar kolom TYPE sampai sebelum kolom QUANTITY
        x_left  = bb_type[1] - 120
        x_right = bb_qty[1] - 5

        y_top = max(y_start, row_y - 70)
        y_bot = row_y + 70

        KEY = [
            "tinned", "annealed", "copper", "aluminium", "aluminum", "steel", "galv",
            "pvc", "pe", "xlpe", "epr", "rubber", "pp", "polyester", "yarn",
            "semiconductive", "semicon", "compound",
            "tape", "wire", "screen", "sheath", "insulation", "armour", "armor"
        ]
        DROP = {"raw", "material", "type", "quantity", "kg", "km", "kg/km"}

        cand = []
        for x0, y0, x1, y1, w, *_ in words:
            x0f = float(x0); y0f = float(y0); y1f = float(y1)
            ww_raw = str(w).strip()
            if not ww_raw:
                continue
            ww = ww_raw.lower()

            if ww in DROP:
                continue
            if is_num(ww_raw):
                continue
            if x0f < x_left or x0f > x_right:
                continue
            if y1f < y_top or y0f > y_bot:
                continue

            # hanya token yang mengandung keyword material
            if not any(k in ww for k in KEY):
                continue

            cand.append((y0f, x0f, ww_raw))

        if not cand:
            return None

        cand.sort(key=lambda t: (round(t[0], 1), t[1]))
        # join token + dedup token berurutan
        toks = []
        prev = None
        for _, _, w in cand:
            w2 = ns(w)
            if not w2:
                continue
            if prev and prev.lower() == w2.lower():
                continue
            toks.append(w2)
            prev = w2

        low_all = " ".join(t.lower() for t in toks)

        # normalisasi minimal: paksa gabungan yang kamu butuh
        if "tinned" in low_all and "annealed" in low_all:
            left = "Tinned Annealed"
        elif "annealed" in low_all:
            left = "Annealed"
        else:
            left = None

        if "copper" in low_all and "tape" in low_all:
            right = "Copper Tape"
        elif "tape" in low_all:
            right = "Tape"
        else:
            right = None

        # kalau bisa, gabungkan pakai format umum
        if left and right:
            type_text = f"{left} {right}"
        else:
            # fallback: join toks apa adanya
            type_text = ns(" ".join(toks))

        # guard: jangan kebawa kata aneh
        if re.search(r"(?i)\b(peroxide|test|dimension|od/lp/dl)\b", type_text):
            return None

        return {"type": type_text, "quantity": qty_val, "unit": "kg/km"}

    finally:
        doc.close()



# ----------------------------
# Mongo write
# ----------------------------
def upsert_document(col, doc: Dict):
    _id = doc["_id"]
    now = datetime.now(timezone.utc)
    col.update_one(
        {"_id": _id},
        {"$set": doc, "$setOnInsert": {"created_at": now}},
        upsert=True
    )

# ----------------------------
def split_cable_blocks(raw_text: str) -> List[Tuple[str, str]]:
    """
    Split raw_text jadi blok-blok berdasarkan header cable seperti:
      "6 Pilot Cable, 70 mm² Insulated"
    Return: [(title, block_text), ...]
    Kalau tidak ketemu header -> [("main", raw_text)]
    """
    t = norm(raw_text)

    # pola header cable (bisa kamu tambah keyword lain selain Pilot Cable nanti)
    pat = re.compile(r"(?i)\b(\d+)\s+(pilot\s+cable[^0-9]{0,80}?(?:insulated)?)\b")

    matches = []
    for m in pat.finditer(t):
        title = norm(m.group(0))  # ✅ pakai norm, bukan ns
        matches.append((m.start(), title))

    if not matches:
        return [("main", t)]

    blocks = []
    for i, (start, title) in enumerate(matches):
        end = matches[i + 1][0] if i < len(matches) - 1 else len(t)
        body = t[start:end].strip()
        blocks.append((title, body))

    return blocks


def cleanup_cabling_ts(rows):
    def norm(s):
        import re
        return re.sub(r"\s+", " ", (s or "")).strip()

    import re

    # jangan pakai \b di belakang "-" (itu yang bikin nggak ke-detect)
    NPER_RE = re.compile(r"(?i)\bn\s*/\s*-")

    out = []
    for r in rows:
        desc = norm(r.get("description"))
        unit = norm(r.get("unit"))
        spec = norm(r.get("specified"))

        # --- FIX A: kalau desc mengandung "n/-" nyasar -> pindahkan ke unit & specified ---
        if NPER_RE.search(desc):
            parts = NPER_RE.split(desc, maxsplit=1)  # split di "n/-"
            left = norm(parts[0])
            right = norm(parts[1]) if len(parts) > 1 else ""

            # buang ekor yang nyasar kayak "( 60" di akhir (biasanya potongan baris berikutnya)
            right = re.sub(r"\(\s*\d+\s*$", "", right).strip()

            # kalau unit % atau %/-: ini salah parsing -> unit seharusnya n/-
            if unit in {"%", "%/-"}:
                desc = left
                unit = "n/-"
                spec = norm((right + " " + spec).strip())

            # kalau unit "-" : jadikan unit n/-
            elif unit == "-":
                desc = left
                unit = "n/-"
                spec = norm((right + " " + spec).strip())

        # --- FIX B: kalau specified cuma ")" dan sebelumnya masih kebuka "(" -> tempel ke prev ---
        if spec == ")" and out:
            prev = out[-1]
            prev_spec = norm(prev.get("specified"))
            if prev_spec.count("(") > prev_spec.count(")"):
                prev["specified"] = norm(prev_spec + " )")
                continue
        
        # ✅ tambahin ini
        spec = re.sub(r"\s*,\s*0+\s*$", "", spec).strip()

        out.append({
            "description": desc or "-",
            "unit": unit or "-",
            "specified": spec or "-"
        })

    return out





# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", required=True, help="Folder atau file PDF")
    ap.add_argument("--mongo_uri", required=True, help='contoh: "mongodb://user:pass@host:27017/?authSource=admin"')
    ap.add_argument("--db_name", default="tds_db")
    ap.add_argument("--collection", default="tds_documents")
    ap.add_argument("--limit", type=int, default=0, help="0=semua")
    ap.add_argument("--progress_every", type=int, default=100)
    args = ap.parse_args()

    headers = [
        "Conductor -",
        "Conductor Shielding -",
        "Covering -",
        "Final Test -",
        "Final test -",
        "Packing",
        "Tapping -",
        "Insulation -",
        "Insulation Shielding -",
        "Metallic Screen -",
        "Twisting -",
        "Overall Screen",
        "Individual Screen",
        "Cabling -",
        "Bedding -",
        "Inner Sheath -",
        "Armouring (Braided)",
        "Armoring (Braided)",      # variasi US spelling
        "Armouring Braided",
        "Armoring Braided",
        "Armouring (Braided)", 
        "Armouring (Braided) -",
        "Armour",
        "Armouring",
        "Armoring",
        "Armour -",
        "Outer Sheath -",
    ]

    client = MongoClient(args.mongo_uri)
    db = client[args.db_name]
    col = db[args.collection]

    total = ok = fail = 0

    for pdf_path in iter_pdf_paths(args.pdf_dir):
        total += 1
        if args.limit and total > args.limit:
            break

        try:
            sha = sha256_file(pdf_path)
            raw_text = pdf_to_text_one_paragraph(pdf_path)
            print("HAS Armouring Braided?", bool(re.search(r"(?i)armou?r.*braided", raw_text)))


            # metadata
            metadata = extract_metadata(pdf_path)

            # marking
            mk = extract_marking_triplet_from_pdf(pdf_path)
            marking = {
                "tipe_marking": mk.get("tipe_marking") or "-",
                "kalimat_marking": mk.get("kalimat_marking") or "-",
                "length_marking": mk.get("length_marking") or "-",
            }

            # split cable blocks (kalau ada)
            cable_blocks = split_cable_blocks(raw_text)

            # denier (layout-based) sekali per PDF
            denier_specs = extract_denier_specs_from_pdf(pdf_path)

            section_obj: Dict = {}
            multi = len(cable_blocks) > 1
            if multi:
                section_obj["cables"] = []

            # parse per cable block
            for cable_title, cable_text in cable_blocks:
                ordered_sections = parse_sections_dynamic(cable_text, headers)
                print("FOUND SECTIONS:", [name for name, _ in ordered_sections])  # ✅ TAROK DI SINI
                if not ordered_sections:
                    continue

                cable_section: Dict = {"title": cable_title, "section": {}}

                for sec_name, sec_body in ordered_sections:
                    if (not sec_body) or (sec_body == "Not Found"):
                        continue

                    s_key = to_snake(sec_name)
                     # paksa nama key supaya konsisten
                    if re.search(r"(?i)armou?r(?:ing)?\s*\(\s*braided\s*\)|armou?r(?:ing)?\s+braided", sec_name):
                        s_key = "armouring_braided"


                    # -------------------------
                    # FINAL TEST
                    # -------------------------
                    if sec_name == "Final Test":
                        # trim biar tidak kebawa ke Packing/section lain
                        sec_body = re.split(r"(?i)\b11\s+Packing\b|\bPacking\b", sec_body, maxsplit=1)[0].strip()
                        cable_section["section"]["final_test"] = parse_final_test_v2(sec_body)
                        continue

                    # -------------------------
                    # PACKING
                    # -------------------------
                    if sec_name == "Packing":
                        p = extract_packing_from_pdf(pdf_path)
                        if not p:
                            p = parse_packing_as_object_v2(sec_body)  # fallback
                        cable_section["section"]["packing"] = p
                        continue

                    # -------------------------
                    # CONDUCTOR
                    # -------------------------
                    if sec_name == "Conductor":
                        cable_section["section"]["conductor"] = parse_section_v2(
                            sec_body,
                            default_rm_unit="kg/km",
                            enable_layers=True
                        )

                        # PATCH RAW MATERIAL dari layout PDF (kalau hasil parser masih kepotong)
                        try:
                            rms = cable_section["section"]["conductor"].get("raw_materials", [])
                            if rms:
                                base_type = (rms[0].get("type") or "").strip().lower()
                                if base_type in {"tinned annealed", "annealed", "tinned"}:
                                    rm_pdf = extract_first_raw_material_from_pdf(pdf_path)
                                    if rm_pdf and rm_pdf.get("type"):
                                        low = rm_pdf["type"].lower()
                                        if "peroxide" not in low and "type" not in low and "quantity" not in low:
                                            rms[0]["type"] = rm_pdf["type"]
                        except Exception:
                            pass

                        continue

                                        
                    # DEFAULT: section lain
                    # -------------------------
                    cable_section["section"][s_key] = parse_section_v2(
                        sec_body,
                        default_rm_unit="kg/km",
                        enable_layers=False
                    )

                    # ==========================================================
                    # ✅ ARMOURING (BRAIDED): force jadi armouring_braided
                    # ==========================================================
                    if s_key == "armouring":
                        # deteksi braided dari body / raw_materials
                        _body = (sec_body or "").lower()
                        _arm_tmp = cable_section["section"].get("armouring") or {}
                        _rms_tmp = _arm_tmp.get("raw_materials") or []
                        _has_braided = ("braided" in _body) or ("braiding" in _body) or any(
                            re.search(r"(?i)\b(braid|braided|braiding)\b", (rm.get("type") or ""))
                            for rm in _rms_tmp
                        )

                        if _has_braided:
                            arm = cable_section["section"].pop("armouring", None) or {}
                            s_key = "armouring_braided"

                            # --- raw_materials: pilih satu yang paling spesifik ---
                            rms = arm.get("raw_materials") or []
                            if rms:
                                def _score_type(t: str) -> float:
                                    tl = (t or "").lower()
                                    sc = 0.0
                                    if "braid" in tl: sc += 10
                                    if "(" in tl and ")" in tl: sc += 5
                                    sc += min(len(t or ""), 200) / 100.0
                                    return sc

                                best = sorted(
                                    rms,
                                    key=lambda rm: (_score_type(rm.get("type")), rm.get("quantity") or 0),
                                    reverse=True
                                )[0]

                                t = (best.get("type") or "").strip()
                                t = re.sub(r"(?i)\(\s*braided\s*\)", "", t).strip()
                                if not re.search(r"(?i)\(\s*braiding\s*\)", t):
                                    t = (t + " ( Braiding )").strip()

                                t = re.sub(r"\(\s*", "( ", t)
                                t = re.sub(r"\s*\)", " )", t)
                                t = re.sub(r"\s+", " ", t).strip()

                                best["type"] = t
                                arm["raw_materials"] = [best]

                            # --- buang "Z ( Braiding )" kalau nyasar ke technical_specs ---
                            ts = arm.get("technical_specs") or []
                            if ts:
                                arm["technical_specs"] = [
                                    r for r in ts
                                    if not re.search(r"(?i)^\s*Z\s*\(\s*Braiding\s*\)\s*$",
                                                    (r.get("description") or "").strip())
                                ]

                            cable_section["section"]["armouring_braided"] = arm



                    obj = cable_section["section"].get(s_key) or {}
                    ts = obj.get("technical_specs") or []
                    if ts:
                        obj["technical_specs"] = cleanup_technical_specs_rows(ts)
                        cable_section["section"][s_key] = obj

                   # ✅ ARMOURING / ARMOURING_BRAIDED: dedup raw_materials
                    if s_key in {"armouring", "armouring_braided"}:
                        arm = cable_section["section"].get(s_key) or {}
                        rms = arm.get("raw_materials") or []

                        if rms:
                            grouped = {}
                            for rm in rms:
                                qty = rm.get("quantity")
                                unit = (rm.get("unit") or "").strip()
                                t = (rm.get("type") or "").strip()

                                key = (qty, unit.lower())

                                score = 0.0
                                if re.search(r"(?i)\b(braided|braid|braiding)\b", t):
                                    score += 10.0
                                if "(" in t and ")" in t:
                                    score += 5.0
                                score += min(len(t), 200) / 100.0

                                cur = grouped.get(key)
                                if (cur is None) or (score > cur["score"]):
                                    grouped[key] = {"score": score, "rm": {"type": t, "quantity": qty, "unit": unit}}

                            arm["raw_materials"] = [v["rm"] for v in grouped.values()]
                            cable_section["section"][s_key] = arm

                    # ✅ INNER_SHEATH: dedup raw_materials (pilih yang paling spesifik per qty+unit)
                    if s_key == "inner_sheath":
                        inn = cable_section["section"].get("inner_sheath") or {}
                        rms = inn.get("raw_materials") or []

                        if rms:
                            grouped = {}
                            for rm in rms:
                                qty = rm.get("quantity")
                                unit = (rm.get("unit") or "").strip()
                                t = (rm.get("type") or "").strip()

                                # key: sama quantity & unit dianggap duplikat
                                key = (qty, unit.lower())

                                # skor: lebih spesifik kalau ada "(...)" / kode warna / lebih panjang
                                score = 0.0
                                if "(" in t and ")" in t:
                                    score += 5.0
                                if re.search(r"(?i)\b(BK|Black|RD|Red|BL|Blue|GR|Green|GY|Grey|Gray|YE|Yellow|WH|White)\b", t):
                                    score += 3.0
                                score += min(len(t), 200) / 100.0

                                cur = grouped.get(key)
                                if (cur is None) or (score > cur["score"]):
                                    grouped[key] = {
                                        "score": score,
                                        "rm": {"type": t, "quantity": qty, "unit": unit}
                                    }

                            inn["raw_materials"] = [v["rm"] for v in grouped.values()]
                            cable_section["section"]["inner_sheath"] = inn
                    

                    # ✅ OUTER_SHEATH: dedup raw_materials (pilih yang paling spesifik per qty+unit)
                    if s_key == "outer_sheath":
                        out_ = cable_section["section"].get("outer_sheath") or {}
                        rms = out_.get("raw_materials") or []

                        if rms:
                            grouped = {}
                            for rm in rms:
                                qty = rm.get("quantity")
                                unit = (rm.get("unit") or "").strip()
                                t = (rm.get("type") or "").strip()

                                key = (qty, unit.lower())

                                # skor: lebih spesifik kalau ada UV Resistant / kode warna / kurung / lebih panjang
                                score = 0.0
                                if re.search(r"(?i)\buv\s*resistant\b", t):
                                    score += 6.0
                                if "(" in t and ")" in t:
                                    score += 5.0
                                if re.search(r"(?i)\b(BK|Black|RD|Red|BL|Blue|GR|Green|GY|Grey|Gray|YE|Yellow|WH|White)\b", t):
                                    score += 3.0
                                score += min(len(t), 200) / 100.0

                                cur = grouped.get(key)
                                if (cur is None) or (score > cur["score"]):
                                    grouped[key] = {
                                        "score": score,
                                        "rm": {"type": t, "quantity": qty, "unit": unit}
                                    }

                            out_["raw_materials"] = [v["rm"] for v in grouped.values()]
                            cable_section["section"]["outer_sheath"] = out_


                    # -------------------------
                    # CABLING: cleanup + fallback raw_materials + centre filler + rapihin technical_specs
                    # -------------------------
                    if s_key == "cabling":
                        cab = cable_section["section"].get("cabling") or {}

                        # --- raw_materials cleanup + dedup ---
                        rms = cab.get("raw_materials") or []
                        cleaned = []
                        for rm in rms:
                            t = (rm.get("type") or "").strip()
                            tl = t.lower()

                            if "centre filler" in tl or "center filler" in tl:
                                continue
                            if re.search(r"(?i)\bn\s*/\s*-\b", t):
                                continue
                            if re.search(r"(?i)\b\d+\s*/\s*(triangle|round|square)\b", t):
                                continue
                            if re.fullmatch(r"(?i)triangle\s+pvc\s+sheath", t):
                                continue
                            if re.search(r"(?i)^triangle\s+.*\bsheath\b", t):
                                t = re.sub(r"(?i)^triangle\s+", "", t).strip()

                            rm["type"] = t
                            cleaned.append(rm)

                        ded = {}
                        for rm in cleaned:
                            key = (rm.get("type", "").lower(), float(rm.get("quantity", 0.0)))
                            ded[key] = rm

                        cab["raw_materials"] = list(ded.values())

                        # --- fallback raw_materials kalau kosong ---
                        if not cab["raw_materials"]:
                            cab["raw_materials"] = extract_cabling_right_raw_materials_from_pdf(pdf_path)

                        # --- technical_specs: ambil dulu ---
                        ts = cab.get("technical_specs") or []

                        # ✅ DEBUG PRINT (TARUH DI SINI)
                        print("TS BEFORE CLEAN:", ts[:5])

                        # --- insert centre filler kalau belum ada ---
                        has_cf = any("centre filler" in (r.get("description", "").lower()) for r in ts)
                        if not has_cf:
                            cf_val = extract_cabling_centre_filler_from_pdf(pdf_path)
                            if cf_val:
                                ts.insert(0, {"description": "Centre filler", "unit": "n/-", "specified": cf_val})

                        # --- FINAL cleanup technical_specs ---
                        cab["technical_specs"] = cleanup_cabling_ts(ts)

                        # ✅ DEBUG PRINT (SETELAH CLEAN)
                        print("TS AFTER  CLEAN:", cab["technical_specs"][:5])

                        # --- simpan balik ---
                        cable_section["section"]["cabling"] = cab


                    # ✅✅ TARUH FALLBACK OUTER_SHEATH DI SINI (MASIH DI DALAM LOOP)
                    if s_key == "outer_sheath":
                        os_ = cable_section["section"].get("outer_sheath") or {}

                        rms = os_.get("raw_materials") or []
                        bad = False
                        if not rms:
                            bad = True
                        else:
                            t0 = (rms[0].get("type") or "").lower()
                            q0 = rms[0].get("quantity")
                            if ("jembo" in t0 and "cable" in t0) or (isinstance(q0, (int, float)) and q0 in (3, 9, 10)):
                                bad = True

                        if bad:
                            os_["raw_materials"] = extract_outer_sheath_right_raw_materials_from_pdf(pdf_path)

                        cable_section["section"]["outer_sheath"] = os_

                    # ✅ CLEANUP technical_specs OUTER_SHEATH: buang qty-only yang nyasar (contoh "748,1")
                    if s_key == "outer_sheath":
                        os_ = cable_section["section"].get("outer_sheath") or {}
                        ts = os_.get("technical_specs") or []

                        cleaned_ts = []
                        for row in ts:
                            desc = (row.get("description") or "").strip()
                            unit = (row.get("unit") or "").strip()
                            spec = (row.get("specified") or "").strip()

                            # 1) buang kalau description cuma angka (748,1 / 14,96 / 59,9 / 2,04 dll)
                            if re.fullmatch(r"\d+(?:[.,]\d+)?", desc):
                                continue

                            # 2) buang kalau row kosong / dash semua
                            if (not desc) or (desc in {"-", "—"} and spec in {"-", "—", ""}):
                                continue

                            cleaned_ts.append(row)

                        os_["technical_specs"] = cleaned_ts
                        cable_section["section"]["outer_sheath"] = os_


                    # patch denier hanya untuk cabling
                    if s_key == "cabling" and denier_specs:
                        cab = cable_section["section"].get("cabling")
                        if cab and "technical_specs" in cab:
                            for row in cab["technical_specs"]:
                                d = (row.get("description") or "").strip().lower()
                                u = (row.get("unit") or "").strip().lower()
                                if u == "n/denier" and d in denier_specs:
                                    row["specified"] = denier_specs[d]

                    # patch armour helically
                    if s_key == "armour":
                        rms = cable_section["section"]["armour"].get("raw_materials", [])
                        if rms:
                            base_type = rms[0].get("type", "")
                            cont = attach_armour_helically_from_pdf(pdf_path, base_type)
                            if cont and cont.lower() not in base_type.lower():
                                rms[0]["type"] = (base_type + " " + cont).strip()

                # simpan hasil per cable block
                if multi:
                    section_obj["cables"].append(cable_section)
                else:
                    section_obj = cable_section["section"]

            # revision history
            revision_history = extract_revision_history_from_pdf(pdf_path, debug=False)

            # doc id
            doc_id_b64 = base64.urlsafe_b64encode(bytes.fromhex(sha)).decode("utf-8").rstrip("=")

            doc = {
                "_id": doc_id_b64,
                "id": metadata.get("number") or "-",
                "metadata": metadata,
                "marking": marking,
                "section": section_obj,
                "revision_history": revision_history,
                "updated_at": datetime.now(timezone.utc),
            }

            upsert_document(col, doc)
            ok += 1

            if ok % args.progress_every == 0 or ok == 1:
                print(f"[OK] {ok} docs inserted/updated. last={os.path.basename(pdf_path)} sections={len(section_obj)}")

        except Exception as e:
            fail += 1
            print(f"[FAIL] {os.path.basename(pdf_path)} -> {e}")

    print(f"[DONE] total_scanned={total} ok={ok} fail={fail} db={args.db_name} collection={args.collection}")


if __name__ == "__main__":
    main()

