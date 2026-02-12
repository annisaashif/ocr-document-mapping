import re
import json
import argparse
from typing import List, Dict, Tuple, Optional
import fitz  # PyMuPDF
import os
from pymongo import MongoClient, ASCENDING

# ----------------------------
# TABLE (5 COLS) PARSER
# ----------------------------
COLS = ["DESCRIPTION", "UNIT", "SPECIFIED", "TYPE", "QUANTITY"]
FOOTER_RE = re.compile(r"(?i)\bpage\s+\d+\s*/\s*\d+")
SECTION_RE = re.compile(r"^\s*(\d{1,2})\s+([A-Za-z][A-Za-z0-9 \-/\(\)&]+)\s*$")
MONTH_RE = re.compile(r"(?i)\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b")
BAD_SECTION_WORDS_RE = re.compile(r"(?i)\b(revision|item\s*no|established|revised|number)\b")
DATEISH_RE = re.compile(r"(?i)^\s*\d{1,2}[-/ ](?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[-/ ]\d{2,4}\b")



def slugify(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "unknown"

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def is_blank(s: str) -> bool:
    s = norm_space(s)
    return (s == "" or s == "-" or s == "–" or s == "—")

def merge_same_y(lines, tol=1.0):
    merged = []
    for r in lines:
        if not merged:
            merged.append(r.copy())
            continue
        if abs(r["_y"] - merged[-1]["_y"]) <= tol:
            for c in COLS:
                if r[c]:
                    merged[-1][c] = (merged[-1][c] + " " + r[c]).strip()
        else:
            merged.append(r.copy())
    return merged

def detect_header_bboxes(page):
    d = page.get_text("dict")
    hits = {}
    targets = set(COLS)
    for b in d.get("blocks", []):
        for l in b.get("lines", []):
            for s in l.get("spans", []):
                t = (s.get("text") or "").strip()
                if t in targets and t not in hits:
                    hits[t] = s["bbox"]  # (x0,y0,x1,y1)
    return hits

def build_boundaries_from_header(hdr):
    """
    Boundary khusus supaya value SPECIFIED yang agak ke kiri tetap masuk SPECIFIED.
    """
    if not all(k in hdr for k in COLS):
        return [250, 275, 420, 520]

    b_desc_unit = (hdr["DESCRIPTION"][2] + hdr["UNIT"][0]) / 2
    b_unit_spec = hdr["UNIT"][2] + 10  # geser kanan
    b_spec_type = (hdr["SPECIFIED"][2] + hdr["TYPE"][0]) / 2
    b_type_qty = (hdr["TYPE"][2] + hdr["QUANTITY"][0]) / 2
    return [b_desc_unit, b_unit_spec, b_spec_type, b_type_qty]

def assign_col_by_x0(x0, bnds):
    if x0 < bnds[0]:
        return "DESCRIPTION"
    if x0 < bnds[1]:
        return "UNIT"
    if x0 < bnds[2]:
        return "SPECIFIED"
    if x0 < bnds[3]:
        return "TYPE"
    return "QUANTITY"

def parse_lines(page, bnds, y_start=0):
    d = page.get_text("dict")
    lines = []
    for b in d.get("blocks", []):
        for l in b.get("lines", []):
            lb = l["bbox"]
            if lb[3] < y_start:
                continue
            rec = {c: "" for c in COLS}
            for s in l.get("spans", []):
                txt = (s.get("text") or "").strip()
                if not txt:
                    continue
                x0 = s["bbox"][0]
                c = assign_col_by_x0(x0, bnds)
                rec[c] = (rec[c] + " " + txt).strip()
            if any(rec[c] for c in COLS):
                rec["_y"] = lb[1]
                lines.append(rec)

    lines.sort(key=lambda r: r["_y"])
    return merge_same_y(lines, tol=1.0)

def find_table_header_y(page):
    hdr = detect_header_bboxes(page)
    if "DESCRIPTION" in hdr:
        return hdr["DESCRIPTION"][3] + 2
    return 160  # ✅ naikkan dari 110


# ----------------------------
# METADATA HEADER (TOP TABLE)
# ----------------------------
META_LABELS_LEFT = ["Reff. Doc.", "Ref. Doc.", "Type", "Size", "Rate voltage", "Ref. Spec."]
META_LABELS_RIGHT = ["Number", "Established", "Revised", "Revision number", "Page"]

def _group_words_to_lines(words, y_tol=1.6):
    rows = []
    for x0, y0, x1, y1, t, *_ in (words or []):
        t = str(t).strip()
        if not t:
            continue
        rows.append((float(y0), float(x0), float(x1), t))

    rows.sort(key=lambda r: (r[0], r[1]))
    lines = []
    for y0, x0, x1, t in rows:
        if not lines:
            lines.append({"y": y0, "items": [(x0, x1, t)]})
            continue
        if abs(y0 - lines[-1]["y"]) <= y_tol:
            lines[-1]["items"].append((x0, x1, t))
        else:
            lines.append({"y": y0, "items": [(x0, x1, t)]})

    out = []
    for ln in lines:
        items = sorted(ln["items"], key=lambda a: a[0])
        text = norm_space(" ".join(a[2] for a in items))
        out.append({"y": ln["y"], "items": items, "text": text})
    return out

def _find_label_in_line(items: List[Tuple[float,float,str]], label: str):
    label_parts = label.split()
    toks = [w for *_ , w in items]
    low = [w.lower() for w in toks]
    target = [p.lower() for p in label_parts]

    for i in range(0, len(low) - len(target) + 1):
        if low[i:i+len(target)] == target:
            x1_end = items[i+len(target)-1][1]
            return i, i+len(target), x1_end
    return None

def _collect_value_right(items: List[Tuple[float,float,str]], x_min: float):
    parts = []
    for x0, x1, t in items:
        if x0 >= x_min:
            parts.append(t)
    return norm_space(" ".join(parts))

def _clean_size(v: str) -> str:
    v = norm_space(v)
    # buang noise yang sering nyangkut
    v = re.split(r"(?i)\bTECHNICAL\s+DATA\s+SHEET\b", v)[0].strip()
    v = re.split(r"(?i)\bTECHNICAL\s+DATA\b", v)[0].strip()
    v = re.split(r"(?i)\bTDS\b", v)[0].strip()
    return v or "-"

def _clean_rate_voltage(v: str) -> str:
    v = norm_space(v)
    # ambil sampai "kV" saja (umumnya bentuk: 3,6/6 (7.2) kV)
    m = re.search(r"(?i)(.+?\bkV\b)", v)
    if m:
        v = m.group(1)
    return v.strip() or "-"

def _clean_type(v: str) -> str:
    v = norm_space(v)
    # beberapa file suka nyangkut kata proses
    v = re.sub(r"(?i)\bLINE\s+PROCESS\b", "", v).strip()
    return v or "-"

def extract_metadata_header(pdf_path: str) -> dict:
    out = {
        "ref_doc": "JCC-PE-PS-003",
        "type": "-",
        "size": "-",
        "rate_voltage": "-",
        "ref_spec": [],
        "number": "-",
        "date": {"established": "-", "revised": "-", "revision_number": 0},
        "page": "-"
    }

    doc = fitz.open(pdf_path)
    try:
        if doc.page_count == 0:
            return out

        page = doc[0]
        page_w = float(page.rect.width)
        mid_x = page_w * 0.52
        words = page.get_text("words") or []
        if not words:
            return out

        lines = _group_words_to_lines(words, y_tol=1.6)

        # fokus area header saja
        header_lines = [ln for ln in lines if ln["y"] <= 140]

        def scan_labels(labels: List[str], region: str) -> Dict[str, str]:
            got = {}
            for label in labels:
                for ln in header_lines:
                    if region == "left":
                        items = [it for it in ln["items"] if it[0] < mid_x]
                    else:
                        items = [it for it in ln["items"] if it[0] >= mid_x - 5]

                    hit = _find_label_in_line(items, label)
                    if not hit:
                        continue
                    _, _, x1_end = hit
                    val = _collect_value_right(items, x1_end + 2)
                    if val:
                        got[label] = val
                        break
            return got

        left = scan_labels(META_LABELS_LEFT, "left")
        right = scan_labels(META_LABELS_RIGHT, "right")

        if "Ref. Doc." in left:
            out["ref_doc"] = left["Ref. Doc."]
        if "Reff. Doc." in left:
            out["ref_doc"] = left["Reff. Doc."]

        if "Type" in left:
            out["type"] = _clean_type(left["Type"])

        if "Size" in left:
            out["size"] = _clean_size(left["Size"])

        if "Rate voltage" in left:
            out["rate_voltage"] = _clean_rate_voltage(left["Rate voltage"])

        # --- ref_spec: ambil jadi LIST, lengkap persis termasuk /A, /C, -22, dll ---
        ref_blob = ""
        if "Ref. Spec." in left:
            ref_blob += " " + left["Ref. Spec."]

        # tambah juga baris-baris kiri lain yang merupakan lanjutan Ref. Spec.
        collecting = False
        for ln in header_lines:
            items_left = [it for it in ln["items"] if it[0] < mid_x]
            text_left = norm_space(" ".join(it[2] for it in items_left))

            if not collecting:
                if re.search(r"(?i)\bRef\.?\s*Spec\.?\b", text_left):
                    collecting = True
                continue

            # stop kalau masuk label header kiri lain
            if re.search(r"(?i)\b(Reff\.?\s*Doc\.?|Ref\.?\s*Doc\.?|Type|Size|Rate\s*voltage)\b", text_left):
                break

            if text_left and text_left != "-":
                ref_blob += " " + text_left

        ref_blob = norm_space(ref_blob)

        # Regex token: tangkap IEC/ISO/BS/JIS/SPLN + kode lengkap (boleh -, /, ., (), dll)
        SPEC_TOKEN_RE = re.compile(
            r"(?i)\b("
            r"(?:SNI\s+)?IEC|ISO|BS|JIS|SPLN"
            r")\s+"
            r"([0-9][0-9A-Za-z\-\./\(\)]*)"
        )

        found = []
        for m in SPEC_TOKEN_RE.finditer(ref_blob):
            org = m.group(1).upper()
            code = m.group(2).strip().rstrip(",;")

            # rapihin spasi "SNI IEC" kalau ada
            org = re.sub(r"(?i)^SNI\s+IEC$", "SNI IEC", org)

            found.append(f"{org} {code}")

        # unique preserve order
        uniq = []
        seen = set()
        for x in found:
            if x not in seen:
                seen.add(x)
                uniq.append(x)

        out["ref_spec"] = uniq


        if "Number" in right:
            out["number"] = right["Number"]

        if "Established" in right:
            out["date"]["established"] = right["Established"]

        if "Revised" in right:
            out["date"]["revised"] = right["Revised"]

        if "Revision number" in right:
            rn = right["Revision number"].strip()
            out["date"]["revision_number"] = int(rn) if rn.isdigit() else rn

        if "Page" in right:
            # prefer format x/y only
            m = re.search(r"(\d+\s*/\s*\d+)", right["Page"])
            out["page"] = m.group(1).replace(" ", "") if m else right["Page"]

        return out
    finally:
        doc.close()


def fix_specified_order_everywhere(out: dict) -> dict:
    """
    Rapihin 'specified' yang kebaca kebalik:
      - "± 1.0 18,1" -> "18,1 ± 1.0"
      - "± 2.0 25,3" -> "25,3 ± 2.0"
      - "± 10% 34 / 1,60" -> "34 / 1,60 ± 10%"
      - "1 1500 /" -> "1 / 1500"
    Berlaku untuk semua section.*.technical_specs
    """
    sec = out.get("section") or {}
    if not isinstance(sec, dict):
        return out

    def ns(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    # pola angka (koma/titik)
    NUM = r"\d+(?:[.,]\d+)?"

    for sname, sobj in sec.items():
        if not isinstance(sobj, dict):
            continue
        ts = sobj.get("technical_specs")
        if not isinstance(ts, list):
            continue

        for row in ts:
            if not isinstance(row, dict):
                continue

            desc = ns(row.get("description", ""))
            if desc.startswith("*"):  # jangan ganggu marker pilot
                continue

            spec = ns(row.get("specified", ""))
            if not spec or spec == "-":
                continue

            # 1) "± X Y"  -> "Y ± X" (X bisa 1.0 / 2.0 / 10% / 0,01)
            m = re.match(rf"^±\s*({NUM}%?)\s*({NUM})$", spec)
            if m:
                row["specified"] = f"{m.group(2)} ± {m.group(1)}"
                continue

            # 2) "± X Y / Z" -> "Y / Z ± X"  (untuk "34 / 1,60 ± 10%")
            m = re.match(rf"^±\s*({NUM}%?)\s*({NUM}\s*/\s*{NUM})$", spec)
            if m:
                row["specified"] = f"{m.group(2)} ± {m.group(1)}"
                continue

            # 3) "± X Y" tapi ada teks lain, contoh: "± 1.0 18,1" (udah ke-cover #1)
            #    kalau kebaca: "± 1.0 18,1" tanpa spasi rapih:
            m = re.match(rf"^±\s*({NUM}%?)\s+({NUM})$", spec)
            if m:
                row["specified"] = f"{m.group(2)} ± {m.group(1)}"
                continue

            # 4) "1 1500 /" -> "1 / 1500"
            m = re.match(rf"^({NUM})\s+({NUM})\s*/\s*$", spec)
            if m:
                row["specified"] = f"{m.group(1)} / {m.group(2)}"
                continue

            # 5) "1 1500 / ..." (kalau ada ekor) -> "1 / 1500 ..."
            m = re.match(rf"^({NUM})\s+({NUM})\s*/\s*(.+)$", spec)
            if m:
                row["specified"] = f"{m.group(1)} / {m.group(2)} {m.group(3)}".strip()
                continue

            # 6) "± X Y" dengan Y mengandung koma (sudah termasuk NUM), aman

        sobj["technical_specs"] = ts
        sec[sname] = sobj

    out["section"] = sec
    return out


# ----------------------------
# MARKING TRIPLET
# ----------------------------
def extract_marking_triplet_from_pdf(pdf_path: str) -> Dict[str, str]:
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
                    start_idx = i; break
                if "cable marking by embossed" in low:
                    start_idx = i; break
                if "cable marking by ink jet printing" in low:
                    start_idx = i; break
                if "cable marking by ink-jet printing" in low:
                    start_idx = i; break

            if start_idx is None:
                continue

            tipe_marking = clean_type_line(line_texts[start_idx])

            def next_lines(from_i: int, max_take: int = 8) -> List[str]:
                out = []
                for j in range(from_i + 1, min(len(line_texts), from_i + 1 + max_take)):
                    t = ns(line_texts[j])
                    if t:
                        out.append(t)
                return out

            after = next_lines(start_idx, 8)
            kalimat_marking = after[0] if after else None

            length_marking = None
            for cand in after[1:]:
                if re.search(r"(?i)\blength\b", cand):
                    length_marking = cand
                    break

            out = {}
            if tipe_marking:
                out["tipe_marking"] = tipe_marking
            if kalimat_marking:
                out["kalimat_marking"] = kalimat_marking
            if length_marking:
                out["length_marking"] = length_marking
            if out:
                return out

        return {}
    finally:
        doc.close()

def is_marking_line(desc: str, unit: str, spec: str) -> bool:
    """Supaya baris marking tidak masuk ke section technical_specs."""
    text = norm_space(" ".join([desc, unit, spec])).lower()
    if not text:
        return False

    if "cable marking by" in text:
        return True
    if text.startswith("length marking"):
        return True

    # biasanya kalimat marking mengandung IEC dan JEMBO CABLE + ada "Year"/"Length Marking"
    if ("jembo" in text and "iec" in text and ("\"year\"" in text or "length marking" in text)):
        return True

    # ✅ TAMBAHKAN INI (lebih kebal, walau tanpa IEC / tanpa tanda kutip)
    if ("jembo" in text and "year" in text and "length marking" in text):
        return True

    return False

DDATE_ONLY_RE = re.compile(r"^\d{1,2}-[A-Za-z]{3}-\d{2}$")               # 23-Nov-10
DATE_NUMBERED_LINE_RE = re.compile(r"^\d+\s+(\d{1,2}-[A-Za-z]{3}-\d{2})$")  # 01 23-Nov-10
REV_HDR_RE = re.compile(r"(?i)\brevision\s+level\b.*\brevision\s+date\b")
# ----------------------------
# SECTIONS (5 COL TABLE)
# ----------------------------
def extract_sections(pdf_path: str) -> dict:
    doc = fitz.open(pdf_path)
    try:
        out = {"section": {}}

        for pi in range(len(doc)):
            page = doc[pi]
            hdr = detect_header_bboxes(page)
            bnds = build_boundaries_from_header(hdr)

            y_start = find_table_header_y(page)
            lines = parse_lines(page, bnds, y_start=y_start)

            current_section = "unknown"
            raw_unit = None

            def ensure(sec):
                if sec not in out["section"]:
                    out["section"][sec] = {"technical_specs": [], "raw_materials": []}

            ensure(current_section)

            for ln in lines:
                desc = norm_space(ln["DESCRIPTION"])
                unit = norm_space(ln["UNIT"])
                spec = norm_space(ln["SPECIFIED"])
                typ  = norm_space(ln["TYPE"])
                qty  = norm_space(ln["QUANTITY"])

                joined = " ".join([desc, unit, spec, typ, qty]).strip()
                if not joined:
                    continue

                if FOOTER_RE.search(joined):
                    continue
                if "JEMBO CABLE COMPANY" in joined.upper():
                    continue
                if desc == "DESCRIPTION" and unit == "UNIT":
                    continue
                if typ.upper() == "RAW MATERIAL":
                    continue

                # treat "-" as empty for logic
                desc_e = "" if is_blank(desc) else desc
                unit_e = "" if is_blank(unit) else unit
                spec_e = "" if is_blank(spec) else spec
                typ_e  = "" if is_blank(typ) else typ
                qty_e  = "" if is_blank(qty) else qty

              
                # SECTION TITLE: allow unit/spec be "-" (treated as empty)
                if desc_e and not unit_e and not spec_e and not typ_e and not qty_e:
                    low_desc = desc_e.lower()

                    # ✅ stop: revision header / revision lines jangan dianggap section
                    if REV_HDR_RE.search(desc_e) or low_desc.startswith("revision"):
                        continue

                    # ✅ stop: tanggal murni (23-Nov-10)
                    if DATE_ONLY_RE.fullmatch(desc_e):
                        continue

                    # ✅ stop: tanggal yang punya nomor depan (01 23-Nov-10)
                    if DATE_NUMBERED_LINE_RE.fullmatch(desc_e):
                        continue

                    m = SECTION_RE.match(desc_e)
                    if m:
                        # ✅ extra safety: kalau judul section hasilnya tanggal, skip
                        maybe_title = m.group(2).strip()
                        if DATE_ONLY_RE.fullmatch(maybe_title):
                            continue

                        current_section = slugify(maybe_title)
                        ensure(current_section)
                        raw_unit = None
                        continue


                # raw material unit line
                if qty_e and not typ_e and not desc_e and not unit_e and not spec_e:
                    raw_unit = qty_e.strip().lower()
                    continue

                # SKIP marking lines from specs (they belong to marking object)
                if is_marking_line(desc_e, unit_e, spec_e) and not typ_e and not qty_e:
                    continue

                # ✅ HARD SKIP: jangan masukkan header revision ke technical_specs
                if REV_HDR_RE.search(desc_e) or REV_HDR_RE.search(joined):
                    continue
                
                # STREAM 1: technical_specs
                if desc_e or unit_e or spec_e:
                    ts_list = out["section"][current_section]["technical_specs"]

                    # if only specified continuation
                    if not desc_e and spec_e and ts_list:
                        ts_list[-1]["specified"] = norm_space(ts_list[-1]["specified"] + " " + spec_e)
                    elif desc_e:
                        ts_list.append({
                            "description": desc_e,
                            "unit": unit_e if unit_e else "-",
                            "specified": spec_e if spec_e else "-"
                        })

                # STREAM 2: raw_materials
                if typ_e or qty_e:
                    rm_list = out["section"][current_section]["raw_materials"]

                    if typ_e and not qty_e and rm_list:
                        rm_list[-1]["type"] = norm_space(rm_list[-1]["type"] + " " + typ_e)
                    elif typ_e and qty_e:
                        rm_list.append({
                            "type": typ_e,
                            "quantity": qty_e,
                            "unit": raw_unit if raw_unit else "kg/km"
                        })
                    elif qty_e and not typ_e and rm_list:
                        rm_list[-1]["quantity"] = norm_space(rm_list[-1]["quantity"] + " " + qty_e)

        if "unknown" in out["section"]:
            u = out["section"]["unknown"]
            if len(u["technical_specs"]) == 0 and len(u["raw_materials"]) == 0:
                out["section"].pop("unknown", None)

        # cleanup: buang section kosong
        for k in list(out["section"].keys()):
            v = out["section"][k]
            if isinstance(v, dict) and not v.get("technical_specs") and not v.get("raw_materials"):
                out["section"].pop(k, None)

        return out
    finally:
        doc.close()

PILOT_HDR_RE = re.compile(r"(?i)^\s*\d+\s+pilot\s+cable\b.*\binsulated\b")

def relocate_pilot_block_from_other_sections(out: dict, pilot_key: str = "pilot_cable_70_mm_insulated") -> dict:
    """
    Kalau blok pilot (judul: '6 Pilot Cable ... Insulated') nyangkut di section lain,
    pindahkan slice ts dari header pilot sampai akhir ke section[pilot_key].

    Aman: kalau tidak ketemu, tidak ngubah apa-apa.
    """
    sec = out.get("section")
    if not isinstance(sec, dict):
        return out

    def ns(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    # siapkan pilot section jika perlu
    if pilot_key not in sec or not isinstance(sec.get(pilot_key), dict):
        sec[pilot_key] = {"technical_specs": [], "raw_materials": []}

    for sname, sobj in list(sec.items()):
        if sname == pilot_key:
            continue
        if not isinstance(sobj, dict):
            continue

        ts = sobj.get("technical_specs")
        if not isinstance(ts, list) or not ts:
            continue

        hit_i = None
        for i, r in enumerate(ts):
            if not isinstance(r, dict):
                continue
            d = ns(r.get("description", ""))
            u = ns(r.get("unit", ""))
            sp = ns(r.get("specified", ""))
            joined = ns(" ".join([d, u, sp]))
            if PILOT_HDR_RE.match(d) or PILOT_HDR_RE.match(joined):
                hit_i = i
                break

        if hit_i is None:
            continue

        # pindahkan blok
        moved = ts[hit_i:]
        remain = ts[:hit_i]

        # drop baris header "6 Pilot Cable..." biar gak masuk specs
        if moved and isinstance(moved[0], dict):
            d0 = ns(moved[0].get("description", ""))
            u0 = ns(moved[0].get("unit", ""))
            sp0 = ns(moved[0].get("specified", ""))
            if PILOT_HDR_RE.match(d0) or PILOT_HDR_RE.match(ns(" ".join([d0, u0, sp0]))):
                moved = moved[1:]

        # append ke pilot
        sec[pilot_key]["technical_specs"].extend(moved)

        # rapihin source section
        sobj["technical_specs"] = remain
        sec[sname] = sobj

        # biasanya cuma muncul sekali, tapi kalau mau lanjut cari lagi, hapus break
        break

    out["section"] = sec
    return out


def _build_layers_from_ts(ts: list) -> tuple[list, list]:
    """
    Input: list technical_specs (dict: description/unit/specified)
    Output:
      - new_ts: technical_specs yang sudah dibuang baris layer detail
      - layers: list layer objects (kalau ketemu)
    """

    def ns(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    def is_od_header(desc: str) -> bool:
        d = ns(desc).lower()
        return ("od/lp/dl" in d) and ("layer" in d)   

    def parse_construction_only(desc: str):
        d = ns(desc)
        # contoh: "1 + 6 wires" / "1 + 6 Wire" / "11 wires"
        if re.fullmatch(r"(?i)\d+\s*(?:\+\s*\d+)*\s*wires?", d):
            return d
        return None


    def parse_layer_desc(desc: str):
        # "1st Layer 1 + 6 Wire" -> ("1st Layer", "1 + 6 Wire")
        d = ns(desc)
        m = re.match(r"(?i)^(?P<title>\d+(?:st|nd|rd|th)\s+Layer)\s+(?P<rest>.+)$", d)
        if not m:
            return None, None
        return ns(m.group("title")), ns(m.group("rest"))

    def is_outer_diameter(desc: str) -> bool:
        return "outer diameter" in (desc or "").lower()

    def clean_spec_noise(s: str) -> str:
        x = ns(s)
        x = re.sub(r"\s+\d+\s*\*?\s*$", "", x).strip()
        return x or "-"

    new_ts = []
    layers = []

    od_header_seen = False
    last_layer_idx = None

    for r in (ts or []):
        desc = ns(r.get("description"))
        unit = ns(r.get("unit") or "-") or "-"
        spec = ns(r.get("specified") or "-") or "-"

        if desc and is_od_header(desc):
            od_header_seen = True
            continue

        if od_header_seen:
            title, construction = parse_layer_desc(desc)
            if title and construction:
                layers.append({
                    "description": f"OD/LP/DL of outer layer {title}",
                    "construction": construction,
                    "unit": unit if unit else "-",
                    "specified": clean_spec_noise(spec),
                })
                last_layer_idx = len(layers) - 1
                continue

            # ✅ STOP LAYER MODE:
            # ketemu row lain (bukan layer & bukan outer diameter) => layer block selesai
            od_header_seen = False
            last_layer_idx = None
            # jangan continue; biar row ini diproses sebagai technical_specs normal

        # default masuk technical_specs biasa
        new_ts.append({
            "description": desc or "-",
            "unit": unit or "-",
            "specified": spec or "-",
        })

    return new_ts, layers

def nest_pilot_cable_conductor_insulation(out: dict, pilot_key: str = "pilot_cable_70_mm_insulated") -> dict:
    """
    Mengubah:
      section[pilot_key] = {technical_specs: [...], raw_materials: [...]}
    Menjadi:
      section[pilot_key] = {
        conductor: {technical_specs:[...], raw_materials:[...], layers:[...]?},
        insulation:{technical_specs:[...], raw_materials:[...]}
      }

    Marker "* Conductor" dan "* Insulation" dipakai sebagai pembatas.
    """

    sec = out.get("section") or {}
    pilot = sec.get(pilot_key)
    if not isinstance(pilot, dict):
        return out

    ts = pilot.get("technical_specs") or []
    rms = pilot.get("raw_materials") or []

    def ns(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    # --- cari index marker ---
    idx_cond = None
    idx_ins = None
    for i, r in enumerate(ts):
        d = ns((r.get("description") or ""))
        low = d.lower()
        if low.startswith("* conductor"):
            idx_cond = i
        if low.startswith("* insulation"):
            idx_ins = i

    # kalau marker gak lengkap, biarkan (biar kamu tau kalau parsing-nya belum bener)
    if idx_cond is None or idx_ins is None or idx_ins <= idx_cond:
        return out

    # --- split technical specs ---
    cond_ts = ts[idx_cond + 1: idx_ins]    # setelah "* Conductor" sampai sebelum "* Insulation"
    ins_ts  = ts[idx_ins + 1:]            # setelah "* Insulation" sampai akhir

    # --- split raw materials (heuristic sederhana yang aman) ---
    def is_ins_rm(rm: dict) -> bool:
        t = ns(rm.get("type", "")).lower()
        return ("siloxen" in t) or ("master batch" in t) or ("masterbatch" in t)

    ins_rm = [rm for rm in rms if is_ins_rm(rm)]
    cond_rm = [rm for rm in rms if not is_ins_rm(rm)]

    # --- apply layers rule ke pilot conductor ---
    cond_new_ts, cond_layers = _build_layers_from_ts(cond_ts)

    pilot_obj = {
        "conductor": {
            "technical_specs": cond_new_ts,
            "raw_materials": cond_rm,
        },
        "insulation": {
            "technical_specs": ins_ts,
            "raw_materials": ins_rm,
        }
    }
    if cond_layers:
        pilot_obj["conductor"]["layers"] = cond_layers

    sec[pilot_key] = pilot_obj
    out["section"] = sec
    return out


def move_packing_from_unknown(out: dict) -> dict:
    """
    Kalau packing kebaca di section.unknown (ada row "13 Packing"),
    pindahkan ke section.packing.
    """
    sec = out.get("section") or {}
    unk = sec.get("unknown")
    if not unk:
        return out

    ts = unk.get("technical_specs") or []
    rm = unk.get("raw_materials") or []

    # indikator kuat bahwa ini packing
    is_packing = any("packing" in (r.get("description","").lower()) for r in ts)

    if not is_packing:
        return out

    # pastiin ada section.packing
    if "packing" not in sec:
        sec["packing"] = {"technical_specs": [], "raw_materials": []}

    # pindahin semua isi unknown ke packing
    sec["packing"]["technical_specs"].extend(ts)
    sec["packing"]["raw_materials"].extend(rm)

    # hapus unknown supaya bersih
    sec.pop("unknown", None)

    out["section"] = sec
    return out


NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")

def _to_int_safe(s: str):
    s = (s or "").strip()
    if not s or s == "-":
        return None
    m = re.search(r"\d+", s)
    return int(m.group(0)) if m else None

def _first_number(s: str):
    s = (s or "").strip()
    m = re.search(r"\d+(?:[.,]\d+)?", s)
    if not m:
        return None
    # angka di packing biasanya integer; kalau ada koma/titik, buang desimal
    return _to_int_safe(m.group(0))

def _extract_two_numbers(s: str):
    s = (s or "").strip()
    nums = re.findall(r"\d+(?:[.,]\d+)?", s)
    if not nums:
        return []
    return nums

def build_packing_object_from_section(out: dict) -> dict:
    sec = out.get("section") or {}
    psec = sec.get("packing")
    if not psec:
        return out

    packing = {
        "standard_length": 0,
        "net_weight": 0,
        "gross_weight": 0,
        "drum": "-",
        "drum_type": "-",
        "drum_quantity": 0,
        "end_cap_quantity": 0,
    }

    def to_int_safe(s: str):
        s = (s or "").strip()
        if not s or s == "-":
            return None
        m = re.search(r"\d+", s)
        return int(m.group(0)) if m else None

    def extract_numbers(s: str):
        return re.findall(r"\d+(?:[.,]\d+)?", (s or ""))

    ts = psec.get("technical_specs") or []
    rms = psec.get("raw_materials") or []

    # ambil standard_length / net_weight / gross_weight
    for r in ts:
        d = (r.get("description") or "").strip().lower()
        val = (r.get("specified") or "").strip()

        if d in ("- standard length", "standard length"):
            nums = extract_numbers(val)   # contoh: ["500","3601"]
            if len(nums) >= 1:
                packing["standard_length"] = to_int_safe(nums[0]) or 0
            if len(nums) >= 2:
                packing["net_weight"] = to_int_safe(nums[1]) or packing["net_weight"]

        elif d in ("- net. weight", "net. weight", "net weight"):
            n = to_int_safe(val)
            if n is not None:
                packing["net_weight"] = n

        elif d in ("- gross weight", "gross weight"):
            n = to_int_safe(val)
            if n is not None:
                packing["gross_weight"] = n

    # ambil drum & end cap dari raw_materials
    for rm in rms:
        t = (rm.get("type") or "").strip()
        q = (rm.get("quantity") or "").strip()
        low = t.lower()

        if "drum" in low:
            packing["drum"] = "Wooden Drum"
            m = re.search(r"\b(\d{2,4})\b", t)  # 210
            if m:
                packing["drum_type"] = m.group(1)
            packing["drum_quantity"] = to_int_safe(q) or packing["drum_quantity"]

        if "end cap" in low:
            packing["end_cap_quantity"] = to_int_safe(q) or packing["end_cap_quantity"]

    # ✅ simpan object packing ke section (replace struktur lama)
    sec["packing"] = packing
    out["section"] = sec

    # ✅ kalau sebelumnya sempat kebentuk root packing, buang
    out.pop("packing", None)

    return out

def build_layers_in_conductor_obj(conductor_obj: dict) -> dict:
    """
    RULE (sama dengan global):
    - layers hanya dibuat kalau ada baris "1st Layer ...", "2nd Layer ...", dst.
    - Kalau hanya header "OD/LP/DL of outer layer" lalu baris "1 + 6 wires" (tanpa 1st/2nd),
      maka: gabungkan jadi technical_specs biasa, TANPA layers.
    - Outer diameter tidak ditempel ke layers.
    """
    if not isinstance(conductor_obj, dict):
        return conductor_obj

    ts = conductor_obj.get("technical_specs") or []
    if not isinstance(ts, list) or not ts:
        return conductor_obj

    def ns(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    def is_od_header(desc: str) -> bool:
        d = ns(desc).lower()
        return ("od/lp/dl" in d) and ("layer" in d)

    def parse_nth_layer(desc: str):
        d = ns(desc)
        m = re.match(r"(?i)^(?P<title>\d+(?:st|nd|rd|th)\s+Layer)\s+(?P<rest>.+)$", d)
        if not m:
            return None, None
        return ns(m.group("title")), ns(m.group("rest"))

    def is_construction_only(desc: str) -> bool:
        d = ns(desc)
        return re.fullmatch(r"(?i)\d+\s*(?:\+\s*\d+)*\s*wires?", d) is not None

    def clean_spec_noise(s: str) -> str:
        x = ns(s)
        # "/ 99 - 104 / Z 5.8" -> "5.8 / 99 - 104 / Z"
        m = re.match(r"^\s*/\s*(.+?)\s+(\d+(?:[.,]\d+)?)\s*$", x)
        if m:
            x = f"{m.group(2)} / {m.group(1)}"
        x = re.sub(r"\s+\d+\s*\*?\s*$", "", x).strip()
        return x or "-"

    new_ts = []
    layers = []

    od_mode = False
    pending_od_header = None

    saw_nth_layer = False
    pending_construction_only = None
    pending_unit = None
    pending_spec = None

    for r in ts:
        desc = ns(r.get("description"))
        unit = ns(r.get("unit") or "-") or "-"
        spec = ns(r.get("specified") or "-") or "-"

        # header OD/LP/DL
        if desc and is_od_header(desc):
            od_mode = True
            pending_od_header = desc
            saw_nth_layer = False
            pending_construction_only = None
            pending_unit = None
            pending_spec = None
            continue

        if od_mode:
            title, construction = parse_nth_layer(desc)

            # CASE A: Nth Layer -> buat layers
            if title and construction:
                saw_nth_layer = True
                pending_construction_only = None  # batalin construction-only

                spec_norm = ns(spec)
                od_unit = "-"
                od_spec = None

                # split outer diameter kalau nyatu
                if re.search(r"(?i)\bouter[-\s]*diameter\b", spec_norm):
                    left, right = re.split(r"(?i)\bouter[-\s]*diameter\b", spec_norm, maxsplit=1)
                    left = ns(left)
                    right = ns(right)
                    layer_spec = clean_spec_noise(left)

                    m = re.match(r"(?i)^(?:approx\.?\s*)?(mm|cm|m)?\s*(.+)$", right)
                    if m:
                        if m.group(1):
                            od_unit = m.group(1)
                        od_spec = ns(m.group(2))
                    else:
                        od_spec = right
                else:
                    layer_spec = clean_spec_noise(spec_norm)

                layers.append({
                    "description": f"OD/LP/DL of outer layer {title}",
                    "construction": construction,
                    "unit": unit if unit else "-",
                    "specified": layer_spec if layer_spec else "-",
                })

                # outer diameter yang nyatu -> masuk technical_specs
                if od_spec:
                    new_ts.append({
                        "description": "Outer diameter",
                        "unit": od_unit if od_unit else "-",
                        "specified": od_spec if od_spec else "-",
                    })

                continue

            # CASE B: construction-only tanpa Nth Layer -> simpan dulu (jangan output)
            if pending_od_header and is_construction_only(desc) and not saw_nth_layer:
                pending_construction_only = desc
                pending_unit = unit
                pending_spec = spec
                continue

            # CASE C: keluar dari OD mode (ketemu baris lain)
            # flush construction-only jika tidak ada Nth Layer sama sekali
            if not saw_nth_layer and pending_od_header and pending_construction_only:
                new_ts.append({
                    "description": f"{pending_od_header} {pending_construction_only}",
                    "unit": pending_unit if pending_unit else "-",
                    "specified": pending_spec if pending_spec else "-",
                })

            # reset OD mode, lalu baris ini diproses normal
            od_mode = False
            pending_od_header = None
            pending_construction_only = None
            pending_unit = None
            pending_spec = None
            saw_nth_layer = False
            # jatuh ke default append

        # default technical_specs
        new_ts.append({
            "description": desc if desc else "-",
            "unit": unit if unit else "-",
            "specified": spec if spec else "-",
        })

    # flush kalau EOF masih di od_mode
    if od_mode and (not saw_nth_layer) and pending_od_header and pending_construction_only:
        new_ts.append({
            "description": f"{pending_od_header} {pending_construction_only}",
            "unit": pending_unit if pending_unit else "-",
            "specified": pending_spec if pending_spec else "-",
        })

    conductor_obj["technical_specs"] = new_ts
    if layers:
        conductor_obj["layers"] = layers
    else:
        conductor_obj.pop("layers", None)

    return conductor_obj


def move_packing_from_final_test(out: dict) -> dict:
    """
    Kalau di section.final_test ada baris "10 Packing" dst,
    potong final_test sampai sebelum "Packing",
    sisanya pindah ke section.packing (technical_specs).
    """
    sec = out.get("section") or {}
    ft = sec.get("final_test")
    if not isinstance(ft, list):
        return out

    PACK_HDR_RE = re.compile(r"(?i)^\s*\d+\s+packing\b")
    pack_start = None
    for i, r in enumerate(ft):
        name = (r.get("test_name") or "").strip()
        if PACK_HDR_RE.match(name):
            pack_start = i
            break

    if pack_start is None:
        return out

    # pastikan packing section ada
    if "packing" not in sec:
        sec["packing"] = {"technical_specs": [], "raw_materials": []}

    # ambil packing rows (buang header "10 Packing")
    packing_rows = ft[pack_start+1:]
    # ubah format dari final_test item -> technical_specs row
    for r in packing_rows:
        sec["packing"]["technical_specs"].append({
            "description": (r.get("test_name") or "-").strip(),
            "unit": (r.get("unit") or "-").strip() or "-",
            "specified": (r.get("specified") or "-").strip() or "-",
        })

    # final_test tinggal sebelum packing
    sec["final_test"] = ft[:pack_start]

    out["section"] = sec
    return out

def fix_layer_specified_order(out: dict) -> dict:
    """
    Kalau specified layer kebentuk jadi: "/ 99 - 104 / Z 5.8"
    ubah jadi: "5.8 / 99 - 104 / Z"
    """
    sec = out.get("section") or {}
    c = sec.get("conductor")
    if not isinstance(c, dict):
        return out

    layers = c.get("layers")
    if not isinstance(layers, list):
        return out

    for ly in layers:
        s = (ly.get("specified") or "").strip()
        # pola: diawali "/" lalu angka di belakang
        m = re.match(r"^/\s*(.+?)\s+(\d+(?:[.,]\d+)?)\s*$", s)
        if m:
            mid = m.group(1).strip()
            num = m.group(2).strip()
            ly["specified"] = f"{num} / {mid}".strip()

    return out

def move_revision_unknown_to_root(out: dict) -> dict:
    """
    Aman untuk semua file:
    - Kalau unknown bukan revision history -> tidak ngapa-ngapain.
    - Kalau unknown adalah revision history:
        * jika root revision_history sudah ada -> hapus unknown (hindari double)
        * jika root kosong -> parse unknown -> isi root (date ISO) -> hapus unknown
    """
    sec = out.get("section") or {}
    unk = sec.get("unknown")
    if not isinstance(unk, dict):
        return out

    ts = unk.get("technical_specs") or []
    if not isinstance(ts, list) or not ts:
        return out

    def ns(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    # --- cek apakah unknown ini revision-history atau bukan (ketat) ---
    first3 = " ".join(ns(r.get("description","")) for r in ts[:3] if isinstance(r, dict)).lower()
    header_like = (
        ("revision" in first3 and "level" in first3 and "date" in first3 and "description" in first3)
        or ("revision level" in first3 and "revision date" in first3 and "description" in first3)
    )

    # harus ada minimal 1 baris yang mengandung pola level + date
    DATE_RE = re.compile(r"\b(\d{1,2}-[A-Za-z]{3}-\d{2})\b")
    LEVEL_RE = re.compile(r"\b(\d{1,2})\b")

    def row_has_level_date(line: str) -> bool:
        line = ns(line)
        mdate = DATE_RE.search(line)
        if not mdate:
            return False
        pre = line[:mdate.start()]
        # level 1-2 digit di bagian sebelum date
        return any(LEVEL_RE.fullmatch(x) for x in re.findall(r"\b\d{1,2}\b", pre))

    level_date_hits = 0
    for r in ts:
        if not isinstance(r, dict):
            continue
        if row_has_level_date(r.get("description","")):
            level_date_hits += 1

    # kalau tidak memenuhi indikator revision -> biarkan unknown (ini kunci “tidak semua file punya”)
    if not header_like or level_date_hits == 0:
        return out

    # --- kalau root revision_history sudah ada, cukup buang unknown supaya tidak double ---
    rh_root = out.get("revision_history")
    if isinstance(rh_root, list) and len(rh_root) > 0:
        sec.pop("unknown", None)
        out["section"] = sec
        return out

    # --- helper: convert dd-Mmm-yy ke ISO ---
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

    # --- parse rows dari unknown ---
    def parse_row(line: str):
        line = ns(line)
        low = line.lower()

        # skip header row
        if ("revision" in low and "level" in low and "date" in low and "description" in low):
            return None

        mdate = DATE_RE.search(line)
        if not mdate:
            return None
        date_raw = mdate.group(1)

        pre = line[:mdate.start()]
        # ambil level digit terakhir sebelum date
        mlev = None
        for mm in LEVEL_RE.finditer(pre):
            mlev = mm
        if not mlev:
            return None
        level = mlev.group(1).zfill(2)

        before_level = ns(pre[:mlev.start()].strip(" :-"))
        after_date = ns(line[mdate.end():].strip(" :-"))

        desc = ns(" ".join([before_level, after_date]).strip()) or "-"

        return {
            "rev_level": level,
            "date": dd_mmm_yy_to_iso(date_raw),
            "description": desc
        }

    parsed = []
    for r in ts:
        if not isinstance(r, dict):
            continue
        pr = parse_row(r.get("description",""))
        if pr:
            parsed.append(pr)

    if not parsed:
        return out

    # dedupe parsed (just in case)
    def k(x: dict):
        return (ns(x.get("rev_level","")).lower(), ns(x.get("date","")).lower(), ns(x.get("description","")).lower())

    seen = set()
    dedup = []
    for x in parsed:
        kk = k(x)
        if kk in seen:
            continue
        seen.add(kk)
        dedup.append(x)

    # ✅ kalau tidak ada hasil, jangan bikin key revision_history sama sekali
    if not dedup:
        # kalau unknown memang revision header doang, boleh bersihin unknown (opsional)
        # sec.pop("unknown", None)
        # out["section"] = sec
        return out

    out["revision_history"] = dedup

    # hapus unknown karena ini revision history
    sec.pop("unknown", None)
    out["section"] = sec
    return out

def fix_swapped_specified_patterns_everywhere(out: dict) -> dict:
    """
    Perbaiki kasus specified kebalik:
      "( ... ) 1 /"  -> "1 / ( ... )"
      "( ... ) 3 /"  -> "3 / ( ... )"
    Berlaku untuk semua section.*.technical_specs
    """
    sec = out.get("section")
    if not isinstance(sec, dict):
        return out

    def ns(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    # ( ... ) 1 /  ️atau  ( ... ) 3 /
    SWAP_PAREN_NUM_SLASH_RE = re.compile(r"^\(\s*(.+?)\s*\)\s*(\d+)\s*/\s*$")

    for sname, sobj in sec.items():
        if not isinstance(sobj, dict):
            continue

        ts = sobj.get("technical_specs")
        if not isinstance(ts, list):
            continue

        for row in ts:
            if not isinstance(row, dict):
                continue
            spec = ns(row.get("specified", ""))
            if not spec or spec == "-":
                continue

            m = SWAP_PAREN_NUM_SLASH_RE.match(spec)
            if m:
                inside = ns(m.group(1))
                num = ns(m.group(2))
                row["specified"] = ns(f"{num} / ( {inside} )")

    out["section"] = sec
    return out


def split_packing_from_final_test_in_section(out: dict) -> dict:
    """
    Kalau "Packing" kebaca nyangkut di section.final_test,
    pindahin ke section.packing.
    """
    sec = out.get("section") or {}
    ft = sec.get("final_test")
    if not isinstance(ft, dict):
        return out

    ts = ft.get("technical_specs") or []
    rm = ft.get("raw_materials") or []

    # cari header "10 Packing" / "13 Packing" dst
    packing_i = None
    for i, r in enumerate(ts):
        d = (r.get("description") or "").strip()
        if re.match(r"(?i)^\d+\s+packing\b", d):
            packing_i = i
            break
        # jaga-jaga kalau cuma "Packing" tanpa angka
        if re.match(r"(?i)^packing\b", d):
            packing_i = i
            break

    if packing_i is None:
        return out

    # pastikan section.packing ada
    if "packing" not in sec:
        sec["packing"] = {"technical_specs": [], "raw_materials": []}

    # pindahin tech specs packing (setelah header packing)
    moved_ts = ts[packing_i + 1:]

    # filter yang benar-benar packing (biar nggak kebawa noise)
    keep = []
    for r in moved_ts:
        desc = (r.get("description") or "").strip().lower()
        if any(k in desc for k in ["standard length", "net", "gross", "drum", "end cap", "endcap", "packing"]):
            keep.append(r)

    sec["packing"]["technical_specs"].extend(keep)

    # pindahin raw materials yang jelas packing
    moved_rm = []
    remain_rm = []
    for r in rm:
        t = (r.get("type") or "").lower()
        if ("drum" in t) or ("end cap" in t) or ("endcap" in t) or ("heat shrink" in t):
            moved_rm.append(r)
        else:
            remain_rm.append(r)

    sec["packing"]["raw_materials"].extend(moved_rm)

    # bersihin final_test: buang header packing + isi packing yang sudah dipindah
    ft["technical_specs"] = ts[:packing_i] + [r for r in moved_ts if r not in keep]
    ft["raw_materials"] = remain_rm

    # kalau final_test sekarang kosong (optional), biarin aja (atau bisa dihapus kalau kamu mau)
    sec["final_test"] = ft
    out["section"] = sec
    return out



def build_final_test_list_from_section(out: dict) -> dict:
    sec = out.get("section") or {}
    ftsec = sec.get("final_test")

    # kalau sudah list, biarkan
    if isinstance(ftsec, list):
        return out

    # kalau bukan dict, tidak bisa diproses
    if not isinstance(ftsec, dict):
        return out


    ts = ftsec.get("technical_specs") or []
    final_test = []

    for r in ts:
        test_name = (r.get("description") or "").strip()
        unit = (r.get("unit") or "-").strip() or "-"
        specified = (r.get("specified") or "-").strip() or "-"

        low = test_name.lower()
        if low.startswith("prepared by") or low.startswith("approved by"):
            continue
        if not test_name:
            continue

        final_test.append({
            "test_name": test_name,
            "unit": unit,
            "specified": specified
        })

    # ✅ simpan list final_test ke section
    sec["final_test"] = final_test

    # ✅ hapus bentuk section["final_test"] yang lama (yang masih ada technical_specs/raw_materials)
    # karena sekarang section["final_test"] sudah jadi list.
    # (ftsec itu dict lama)
    # Cara aman: kalau ftsec masih dict, tidak dipakai lagi.
    # Kita tidak perlu menyisakan ftsec.
    # Tapi karena key-nya sama, sudah ketimpa oleh list di atas.

    out["section"] = sec

    # ✅ buang root kalau ada
    out.pop("final_test", None)

    return out

def drop_signature_blocks(out: dict) -> dict:
    """
    Hapus blok tanda tangan yang sering masuk ke section.unknown:
    - Prepared by / Checked by / Approved by
    - baris nama dalam tanda kurung
    - jabatan: Engineer / Supervisor / Manager / etc
    """
    sec = out.get("section") or {}
    unk = sec.get("unknown")
    if not isinstance(unk, dict):
        return out

    ts = unk.get("technical_specs") or []
    if not isinstance(ts, list) or not ts:
        return out

    def ns(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    def looks_like_signature_row(r: dict) -> bool:
        d = ns(r.get("description", "")).lower()
        u = ns(r.get("unit", "")).lower()
        s = ns(r.get("specified", "")).lower()
        blob = " ".join([d, u, s])

        if "prepared by" in blob or "checked by" in blob or "approved by" in blob:
            return True

        # nama dalam kurung: ( Aditya R. )
        if re.search(r"\(\s*[A-Za-z].{1,40}\)", blob):
            return True

        # jabatan umum
        if re.fullmatch(r"(engineer|supervisor|manager|foreman|inspector|qc|qa)", d):
            return True

        return False

    # kalau 70% baris unknown adalah signature → buang semua
    sig_hits = sum(1 for r in ts if isinstance(r, dict) and looks_like_signature_row(r))
    if sig_hits / max(len(ts), 1) >= 0.6:
        sec.pop("unknown", None)
        out["section"] = sec
        return out

    # kalau nggak dominan, buang baris sig saja
    new_ts = []
    for r in ts:
        if isinstance(r, dict) and looks_like_signature_row(r):
            continue
        new_ts.append(r)

    unk["technical_specs"] = new_ts
    if len(new_ts) == 0 and len(unk.get("raw_materials") or []) == 0:
        sec.pop("unknown", None)
    else:
        sec["unknown"] = unk

    out["section"] = sec
    return out

def remove_raw_materials_from_final_test(out: dict) -> dict:
    sec = out.get("section") or {}
    ft = sec.get("final_test")

    # kalau final_test masih dict (punya technical_specs/raw_materials)
    if isinstance(ft, dict):
        ft.pop("raw_materials", None)
        sec["final_test"] = ft

    # kalau final_test sudah list: aman, tidak perlu apa-apa
    out["section"] = sec
    return out


def clean_final_test_list_in_section(out: dict) -> dict:
    sec = out.get("section") or {}
    ft = sec.get("final_test")
    if not isinstance(ft, list):
        return out

    def ns(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    BULLET_RE = re.compile(r"^\s*[-–—•]+\s*")
    NAME_PAREN_RE = re.compile(r"^\(\s*[A-Za-z].{1,60}\)\s*$")  # ( Christian Bayu A. S. )
    ROLE_ONLY_RE = re.compile(r"(?i)^(engineer|supervisor|manager|inspector|foreman|qc|qa|checker|approved|prepared)$")
    SIG_WORDS_RE = re.compile(r"(?i)\b(prepared by|checked by|approved by)\b")

    def clean_leading_bullet(s: str) -> str:
        s = ns(s)
        s = BULLET_RE.sub("", s)
        return ns(s)

    new_ft = []
    for r in ft:
        if not isinstance(r, dict):
            continue

        name = clean_leading_bullet(r.get("test_name", ""))
        unit = ns(r.get("unit", "-")) or "-"
        spec = ns(r.get("specified", "-")) or "-"

        low_name = name.lower()
        low_spec = spec.lower()

        # 1) buang baris signature by words
        blob = f"{low_name} {low_spec}"
        if SIG_WORDS_RE.search(blob):
            continue

        # 2) buang nama dalam kurung (baik di test_name maupun specified)
        if NAME_PAREN_RE.match(name) or NAME_PAREN_RE.match(spec):
            continue

        # 3) buang baris role-only (Engineer / Supervisor, dll)
        #    contoh kasus kamu: test_name="Engineer", specified="Supervisor"
        if ROLE_ONLY_RE.match(low_name) and (spec == "-" or ROLE_ONLY_RE.match(low_spec)):
            continue
        if ROLE_ONLY_RE.match(low_name) and ROLE_ONLY_RE.match(low_spec):
            continue

        # 4) normalisasi Dimension (punya fungsi lain juga, tapi aman)
        if re.search(r"(?i)\bdimension\b", name):
            spec = "Ok"
            unit = "-"

        new_ft.append({"test_name": name, "unit": unit, "specified": spec})

    sec["final_test"] = new_ft
    out["section"] = sec
    return out


def fix_armour_steel_tape_dimension(out: dict) -> dict:
    sec = out.get("section") or {}
    armour = sec.get("armour")
    if not isinstance(armour, dict):
        return out

    ts = armour.get("technical_specs") or []
    if not isinstance(ts, list):
        return out

    for r in ts:
        desc = (r.get("description") or "").strip().lower()
        spec = (r.get("specified") or "").strip()

        # target: "No./dimension Steel Tape" yang kehilangan 50
        if "no./dimension" in desc and "steel tape" in desc:
            # normalisasi spasi biar gampang match
            spec_norm = re.sub(r"\s+", " ", spec)

            # kasus kamu: "2 ( x 0,5 )" / "2 ( x 0.5 )"
            if re.search(r"(?i)^\s*2\s*\(\s*x\s*0[.,]5\s*\)\s*$", spec_norm):
                r["specified"] = re.sub(
                    r"(?i)^\s*2\s*\(\s*x\s*(0[.,]5)\s*\)\s*$",
                    r"2 ( 50 x \1 )",
                    spec_norm
                )

    armour["technical_specs"] = ts
    sec["armour"] = armour
    out["section"] = sec
    return out


def split_conductor_shielding_from_conductor(out: dict) -> dict:
    """
    Jika di section.conductor ada sub-header seperti "2 Conductor Shielding",
    pindahkan baris setelahnya ke section.conductor_shielding.

    Juga pisahkan raw_materials:
      - shielding: semiconductive, conductive, screen, shielding, carbon black, perox...
      - sisanya tetap di conductor
    """
    sec = out.get("section") or {}
    con = sec.get("conductor")
    if not isinstance(con, dict):
        return out

    ts = con.get("technical_specs") or []
    rm = con.get("raw_materials") or []

    if not isinstance(ts, list):
        return out

    def ns(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    # header yang memicu split
    HDR_RE = re.compile(r"(?i)^\s*\d+\s+conductor\s+shielding\b")

    # cari index header
    hdr_i = None
    for i, r in enumerate(ts):
        d = ns((r.get("description") or ""))
        if HDR_RE.match(d):
            hdr_i = i
            break

    if hdr_i is None:
        return out  # tidak ada shielding header -> biarkan

    # pindahkan technical_specs setelah header ke conductor_shielding
    shielding_ts = ts[hdr_i + 1:]
    conductor_ts = ts[:hdr_i]  # header tidak ikut

    # siapkan target section
    if "conductor_shielding" not in sec or not isinstance(sec.get("conductor_shielding"), dict):
        sec["conductor_shielding"] = {"technical_specs": [], "raw_materials": []}

    # merge tech specs (append)
    sec["conductor_shielding"]["technical_specs"].extend(shielding_ts)

    # --- split raw materials ---
    def is_shielding_rm(rm_row: dict) -> bool:
        t = ns(rm_row.get("type", "")).lower()
        # keyword yang umum muncul di conductor shielding/screen
        keys = [
            "semiconductive", "semi-conductive", "conductive",
            "screen", "shield", "shielding",
            "carbon black", "peroxide", "perox"
        ]
        return any(k in t for k in keys)

    shielding_rm = []
    conductor_rm = []
    for r in rm:
        if not isinstance(r, dict):
            continue
        (shielding_rm if is_shielding_rm(r) else conductor_rm).append(r)

    # apply hasil split
    con["technical_specs"] = conductor_ts
    con["raw_materials"] = conductor_rm
    sec["conductor"] = con

    sec["conductor_shielding"]["raw_materials"].extend(shielding_rm)

    out["section"] = sec
    return out


def build_conductor_layers_from_section(out: dict, section_name: str = "conductor") -> dict:
    """
    RULE:
    - layers hanya dibuat kalau ada baris "1st Layer ...", "2nd Layer ...", dst.
    - Kalau hanya header "OD/LP/DL of outer layer" lalu baris "1 + 6 wires" (tanpa 1st/2nd),
      maka: gabungkan jadi technical_specs biasa, TANPA layers.
    - Outer diameter tidak ditempel ke layers.
    """
    sec = out.get("section") or {}
    c = sec.get(section_name)
    if not isinstance(c, dict):
        return out

    ts = c.get("technical_specs") or []
    if not isinstance(ts, list) or not ts:
        return out

    def ns(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    def is_od_header(desc: str) -> bool:
        d = ns(desc).lower()
        return ("od/lp/dl" in d) and ("layer" in d)  # cover "of outer layer", "of layer"

    def parse_nth_layer(desc: str):
        d = ns(desc)
        m = re.match(r"(?i)^(?P<title>\d+(?:st|nd|rd|th)\s+Layer)\s+(?P<rest>.+)$", d)
        if not m:
            return None, None
        return ns(m.group("title")), ns(m.group("rest"))

    def is_construction_only(desc: str) -> bool:
        d = ns(desc)
        # contoh: "1 + 6 wires" / "1 + 6 wire" / "11 wires"
        return re.fullmatch(r"(?i)\d+\s*(?:\+\s*\d+)*\s*wires?", d) is not None

    def clean_spec_noise(s: str) -> str:
        x = ns(s)
        # "/ 99 - 104 / Z 5.8" -> "5.8 / 99 - 104 / Z"
        m = re.match(r"^\s*/\s*(.+?)\s+(\d+(?:[.,]\d+)?)\s*$", x)
        if m:
            x = f"{m.group(2)} / {m.group(1)}"
        x = re.sub(r"\s+\d+\s*\*?\s*$", "", x).strip()
        return x or "-"

    new_ts = []
    layers = []

    od_mode = False
    pending_od_header = None

    # ✅ kunci: layers dibuat hanya jika pernah ketemu Nth Layer
    saw_nth_layer = False
    pending_construction_only = None  # untuk kasus "1 + 6 wires" tanpa Nth Layer

    for r in ts:
        desc = ns(r.get("description"))
        unit = ns(r.get("unit") or "-") or "-"
        spec = ns(r.get("specified") or "-") or "-"

        # 1) ketemu header OD/LP/DL
        if desc and is_od_header(desc):
            od_mode = True
            pending_od_header = desc  # simpan header asli
            saw_nth_layer = False
            pending_construction_only = None
            continue

        # 2) kalau sedang di blok OD/LP/DL
        if od_mode:
            title, construction = parse_nth_layer(desc)

            # ---- CASE A: ada 1st/2nd/.. Layer => bikin layers ----
            if title and construction:
                saw_nth_layer = True

                # kalau sebelumnya sempat ketemu construction-only, jangan gabung (abaikan) karena sudah jelas multi-layer
                pending_construction_only = None

                spec_norm = ns(spec)

                # kalau outer diameter nyatu di spec layer, split (outer diameter masuk technical_specs)
                od_unit = "-"
                od_spec = None

                if re.search(r"(?i)\bouter[-\s]*diameter\b", spec_norm):
                    left, right = re.split(r"(?i)\bouter[-\s]*diameter\b", spec_norm, maxsplit=1)
                    left = ns(left)
                    right = ns(right)
                    layer_spec = clean_spec_noise(left)

                    m = re.match(r"(?i)^(?:approx\.?\s*)?(mm|cm|m)?\s*(.+)$", right)
                    if m:
                        if m.group(1):
                            od_unit = m.group(1)
                        od_spec = ns(m.group(2))
                    else:
                        od_spec = right
                else:
                    layer_spec = clean_spec_noise(spec_norm)

                layers.append({
                    "description": f"OD/LP/DL of outer layer {title}",
                    "construction": construction,
                    "unit": unit if unit else "-",
                    "specified": layer_spec if layer_spec else "-",
                })

                # outer diameter nyatu -> masuk technical_specs
                if od_spec:
                    new_ts.append({
                        "description": "Outer diameter",
                        "unit": od_unit if od_unit else "-",
                        "specified": od_spec if od_spec else "-",
                    })

                continue

            # ---- CASE B: hanya "1 + 6 wires" (tanpa Nth Layer) => simpan dulu ----
            if pending_od_header and is_construction_only(desc) and not saw_nth_layer:
                pending_construction_only = desc
                pending_unit = unit
                pending_spec = spec
                # jangan output dulu; tunggu apakah nanti muncul Nth Layer
                continue

            # ---- CASE C: blok OD/LP/DL selesai (ketemu baris lain) ----
            # Kalau tidak pernah ketemu Nth Layer tapi ada construction-only, gabungkan jadi technical_specs
            if not saw_nth_layer and pending_od_header and pending_construction_only:
                new_ts.append({
                    "description": f"{pending_od_header} {pending_construction_only}",
                    "unit": pending_unit if pending_unit else "-",
                    "specified": pending_spec if pending_spec else "-",
                })

            # reset OD mode dan jatuhkan baris ini ke technical_specs normal
            od_mode = False
            pending_od_header = None
            pending_construction_only = None
            saw_nth_layer = False
            # (no continue)

        # 3) default: technical_specs biasa
        new_ts.append({
            "description": desc if desc else "-",
            "unit": unit if unit else "-",
            "specified": spec if spec else "-",
        })

    # kalau file selesai tapi masih dalam od_mode, flush pending construction-only
    if od_mode and (not saw_nth_layer) and pending_od_header and pending_construction_only:
        new_ts.append({
            "description": f"{pending_od_header} {pending_construction_only}",
            "unit": pending_unit if pending_unit else "-",
            "specified": pending_spec if pending_spec else "-",
        })

    # set balik
    c["technical_specs"] = new_ts
    if layers:
        c["layers"] = layers
    else:
        c.pop("layers", None)

    sec[section_name] = c
    out["section"] = sec
    return out


def fix_final_test_dimension(out: dict) -> dict:
    sec = out.get("section") or {}
    ft = sec.get("final_test")
    if not isinstance(ft, list):
        return out

    for r in ft:
        name = (r.get("test_name") or "")
        if re.search(r"(?i)\bdimension\b", name):
            r["specified"] = "Ok"
            if not r.get("unit"):
                r["unit"] = "-"
    sec["final_test"] = ft
    out["section"] = sec
    return out

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

                    # kalau inline desc kosong dan ada carry_desc (desc muncul duluan), pakai carry_desc
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
                    # normal order: isi desc untuk row pending
                    pending["description"] = txt
                    pending = None
                else:
                    # reverse order: simpan buat row berikutnya
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



def fix_final_test_noise(out: dict) -> dict:
    sec = out.get("section") or {}
    ft = sec.get("final_test")
    if not isinstance(ft, list):
        return out

    def ns(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    SPACED_LETTERS_RE = re.compile(r"(?:\b[A-Za-z]\b\s*){4,}")

    for r in ft:
        spec = ns(r.get("specified", ""))
        if not spec or spec == "-":
            continue

        cleaned = SPACED_LETTERS_RE.sub("", spec)
        cleaned = ns(cleaned)

        if cleaned.lower() == "ok":
            r["specified"] = "Ok"
        elif not cleaned:
            r["specified"] = "-"
        else:
            r["specified"] = cleaned

    sec["final_test"] = ft
    out["section"] = sec
    return out

def fix_pilot_cable_section(out: dict) -> dict:
    """
    Pindahkan bagian insulation pilot cable dari section["insulation"]
    ke section["pilot_cable_70_mm_insulated"] berdasarkan heuristik data:
      - Nom. thickness = 1,1
      - Minimum thickness = 0,89
      - Identification of core mengandung Green...Yellow...
      - Outer diameter = 12,2
    Juga pindahkan raw_materials insulation pilot:
      - XLPE Siloxen
      - Master Batch
    Lalu sisipkan marker "* Insulation" di pilot kalau belum ada.
    """
    sec = out.get("section") or {}
    pilot = sec.get("pilot_cable_70_mm_insulated")
    ins = sec.get("insulation")

    if not isinstance(pilot, dict) or not isinstance(ins, dict):
        return out

    pilot_ts = pilot.get("technical_specs") or []
    pilot_rm = pilot.get("raw_materials") or []
    ins_ts = ins.get("technical_specs") or []
    ins_rm = ins.get("raw_materials") or []

    def ns(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    def looks_like_pilot_insulation_row(r: dict) -> bool:
        desc = ns(r.get("description", "")).lower()
        spec = ns(r.get("specified", "")).lower()

        # ciri khas pilot insulation dari contohmu
        if "green" in spec and "yellow" in spec:
            return True
        if desc == "nom. thickness" and re.search(r"\b1[.,]1\b", spec):
            return True
        if "minimum thickness" in desc and re.search(r"\b0[.,]89\b", spec):
            return True
        if ("outer diameter" in desc or "outer diameter approx" in desc) and re.search(r"\b12[.,]2\b", spec):
            return True
        return False

    # --- split insulation technical_specs: main vs pilot ---
    split_idx = None
    for i, r in enumerate(ins_ts):
        if looks_like_pilot_insulation_row(r):
            split_idx = i
            break

    if split_idx is None:
        # tidak ketemu pilot insulation, biarkan
        return out

    pilot_ins_ts = ins_ts[split_idx:]
    main_ins_ts = ins_ts[:split_idx]

    # --- split insulation raw materials: pilot vs main ---
    def is_pilot_rm(rm: dict) -> bool:
        t = ns(rm.get("type", "")).lower()
        return ("siloxen" in t) or ("master batch" in t)

    pilot_ins_rm = [rm for rm in ins_rm if is_pilot_rm(rm)]
    main_ins_rm = [rm for rm in ins_rm if not is_pilot_rm(rm)]

    # update insulation (utama) supaya tidak kecampur
    ins["technical_specs"] = main_ins_ts
    ins["raw_materials"] = main_ins_rm
    sec["insulation"] = ins

    # --- masukkan ke pilot section ---
    # Pastikan ada marker "* Insulation"
    has_star_ins = any(isinstance(r, dict) and ns(r.get("description","")).lower().startswith("* insulation") for r in pilot_ts)

    if not has_star_ins:
        # taruh marker di akhir (rapi), lalu masukin baris insulation pilot
        pilot_ts.append({"description": "* Insulation", "unit": "-", "specified": "-"})

    # append technical specs pilot insulation
    pilot_ts.extend(pilot_ins_ts)

    # append raw materials pilot insulation
    # (pilot_rm sudah punya copper tape, tambahkan siloxen + master batch)
    pilot_rm.extend(pilot_ins_rm)

    pilot["technical_specs"] = pilot_ts
    pilot["raw_materials"] = pilot_rm
    sec["pilot_cable_70_mm_insulated"] = pilot

    out["section"] = sec
    return out

def split_pilot_cable_conductor_insulation(out: dict) -> dict:
    """
    Pecah section["pilot_cable_70_mm_insulated"] menjadi:
      pilot["conductor"]  dan pilot["insulation"]
    berdasarkan marker:
      "* Conductor" dan "* Insulation"

    - Marker TIDAK ikut dimasukkan ke list specs.
    - Raw materials dibagi pakai keyword sederhana:
        conductor: copper, annealed, tinned, tape (copper tape)
        insulation: siloxen, master batch, xlpe
      sisanya masuk conductor (default) supaya tidak hilang.
    """
    sec = out.get("section") or {}
    pilot = sec.get("pilot_cable_70_mm_insulated")
    if not isinstance(pilot, dict):
        return out

    ts = pilot.get("technical_specs") or []
    rm = pilot.get("raw_materials") or []

    def ns(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    def is_marker(desc: str) -> str:
        d = ns(desc)
        # buang angka depan seperti "7 * Insulation"
        d = re.sub(r"^\s*\d+\s*", "", d).strip()
        low = d.lower()

        if low.startswith("* conductor"):
            return "conductor"
        if low.startswith("* insulation"):
            return "insulation"
        return ""


    # --- split technical specs by marker ---
    pilot_con_ts, pilot_ins_ts = [], []
    mode = None

    for row in ts:
        if not isinstance(row, dict):
            continue
        desc = row.get("description")
        if not isinstance(desc, str):
            continue

        mk = is_marker(desc)
        if mk:
            mode = mk
            continue  # marker tidak dimasukkan

        # kalau belum ketemu marker, biarkan masuk conductor (default)
        if mode == "insulation":
            pilot_ins_ts.append(row)
        else:
            pilot_con_ts.append(row)

    # --- split raw materials by keyword ---
    con_rm, ins_rm, other_rm = [], [], []

    for r in rm:
        if not isinstance(r, dict):
            continue
        t = ns(r.get("type", "")).lower()

        if any(k in t for k in ["siloxen", "master batch", "xlpe"]):
            ins_rm.append(r)
        elif any(k in t for k in ["copper", "annealed", "tinned", "tape"]):
            con_rm.append(r)
        else:
            other_rm.append(r)

    # sisa raw material yang gak kebaca -> masuk conductor biar gak hilang
    con_rm.extend(other_rm)

    # build objek baru
    new_pilot = {
        "conductor": {
            "technical_specs": pilot_con_ts,
            "raw_materials": con_rm
        },
        "insulation": {
            "technical_specs": pilot_ins_ts,
            "raw_materials": ins_rm
        }
    }

    # replace pilot section
    sec["pilot_cable_70_mm_insulated"] = new_pilot
    out["section"] = sec
    return out


def clean_bullets_everywhere(out: dict) -> dict:
    """
    Bersihkan bullet di awal text:
      "-", "–", "—", "•"
    Untuk:
      - section.*.technical_specs[].description
      - section.final_test[].test_name

    TIDAK menghapus "*" khusus marker pilot (* Conductor, * Insulation)
    """
    def ns(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    BULLET_RE = re.compile(r"^\s*[-–—•]+\s*")

    def clean_leading_bullet(s: str) -> str:
        s = ns(s)
        s = BULLET_RE.sub("", s)
        return ns(s)

    sec = out.get("section")
    if isinstance(sec, dict):
        # 1) technical_specs.description
        for sec_name, secobj in sec.items():
            if not isinstance(secobj, dict):
                continue
            ts = secobj.get("technical_specs")
            if isinstance(ts, list):
                for row in ts:
                    if isinstance(row, dict) and isinstance(row.get("description"), str):
                        desc = row["description"]
                        if sec_name == "pilot_cable_70_mm_insulated" and desc.lstrip().startswith("*"):
                            continue
                        row["description"] = clean_leading_bullet(desc)

        # 2) final_test.test_name (di section)
        ft = sec.get("final_test")
        if isinstance(ft, list):
            for row in ft:
                if isinstance(row, dict) and isinstance(row.get("test_name"), str):
                    row["test_name"] = clean_leading_bullet(row["test_name"])
            sec["final_test"] = ft

        out["section"] = sec

    # buang root final_test kalau ada (jaga-jaga)
    out.pop("final_test", None)

    return out

def build_pilot_conductor_layers(out: dict) -> dict:
    sec = out.get("section") or {}
    pilot = sec.get("pilot_cable_70_mm_insulated")
    if not isinstance(pilot, dict):
        return out

    if isinstance(pilot.get("conductor"), dict):
        pilot["conductor"] = build_layers_in_conductor_obj(pilot["conductor"])

    sec["pilot_cable_70_mm_insulated"] = pilot
    out["section"] = sec
    return out

def move_section_key_after(out: dict, key: str, after_key: str) -> dict:
    """
    Pindahkan posisi key di out["section"] supaya tepat setelah after_key,
    tanpa mengubah urutan key lainnya.
    """
    sec = out.get("section")
    if not isinstance(sec, dict):
        return out
    if key not in sec or after_key not in sec:
        return out
    if key == after_key:
        return out

    new_sec = {}
    for k, v in sec.items():
        if k == key:
            continue  # skip dulu
        new_sec[k] = v
        if k == after_key:
            new_sec[key] = sec[key]  # sisipkan tepat setelah after_key

    # jaga-jaga kalau after_key tidak ketemu (harusnya tidak terjadi)
    if key not in new_sec:
        new_sec[key] = sec[key]

    out["section"] = new_sec
    return out


def postprocess_unknown_to_packing(out: dict) -> dict:
    """
    Jika 'unknown' berisi "13 Packing" maka pindahkan ke 'packing'.
    Kalau 'packing' sudah ada, merge tapi dedupe supaya tidak dobel.
    """
    sec = out.get("section", {})
    if "unknown" not in sec:
        return out

    u = sec["unknown"]
    ts_u = u.get("technical_specs", []) or []
    rm_u = u.get("raw_materials", []) or []

    # cari header packing: "13 Packing" / "12 Packing" dll
    packing_idx = None
    for i, row in enumerate(ts_u):
        desc = (row.get("description") or "").strip()
        if re.match(r"^\d+\s+packing\b", desc, flags=re.I):
            packing_idx = i
            break

    if packing_idx is None:
        return out

    # pastikan packing ada
    if "packing" not in sec:
        sec["packing"] = {"technical_specs": [], "raw_materials": []}

    # ambil TS setelah header packing (buang row "13 Packing")
    moved_ts = ts_u[packing_idx + 1:]
    moved_rm = rm_u

    # merge
    sec["packing"]["technical_specs"].extend(moved_ts)
    sec["packing"]["raw_materials"].extend(moved_rm)

    # dedupe technical_specs packing
    seen = set()
    dedup_ts = []
    for r in sec["packing"]["technical_specs"]:
        k = (
            (r.get("description") or "").strip().lower(),
            (r.get("unit") or "").strip().lower(),
            (r.get("specified") or "").strip().lower(),
        )
        if k in seen:
            continue
        seen.add(k)
        dedup_ts.append(r)
    sec["packing"]["technical_specs"] = dedup_ts

    # dedupe raw_materials packing
    seen = set()
    dedup_rm = []
    for r in sec["packing"]["raw_materials"]:
        k = (
            (r.get("type") or "").strip().lower(),
            (r.get("quantity") or "").strip().lower(),
            (r.get("unit") or "").strip().lower(),
        )
        if k in seen:
            continue
        seen.add(k)
        dedup_rm.append(r)
    sec["packing"]["raw_materials"] = dedup_rm

    # bersihkan unknown: sisakan bagian sebelum "13 Packing" (kalau ada)
    remain_ts = ts_u[:packing_idx]
    sec["unknown"]["technical_specs"] = remain_ts
    sec["unknown"]["raw_materials"] = []

    # kalau unknown kosong, hapus
    if len(sec["unknown"]["technical_specs"]) == 0 and len(sec["unknown"]["raw_materials"]) == 0:
        sec.pop("unknown", None)

    out["section"] = sec
    return out

def fix_packing_standard_length(out: dict) -> dict:
    sec = out.get("section", {})
    p = sec.get("packing")
    if not p:
        return out

    ts = p.get("technical_specs", [])
    # cari standard length row
    for r in ts:
        if (r.get("description") or "").strip().lower() == "- standard length":
            val = (r.get("specified") or "").strip()
            # ambil angka pertama aja (500)
            m = re.match(r"^(\d+(?:[.,]\d+)?)\b", val)
            if m:
                r["specified"] = m.group(1)
            break
    return out

DATE_LINE_RE = re.compile(r"^\d{1,2}-[A-Za-z]{3}-\d{2}$")  # 23-Nov-10
REV_HDR_RE = re.compile(r"(?i)\brevision\s+level\b.*\brevision\s+date\b")
DATE_ONLY_RE = re.compile(r"^\d{1,2}-[A-Za-z]{3}-\d{2}$")               # 23-Nov-10
DATE_NUMBERED_LINE_RE = re.compile(r"^\d+\s+(\d{1,2}-[A-Za-z]{3}-\d{2})$")  # 01 23-Nov-10
REV_HDR_RE = re.compile(r"(?i)\brevision\s+level\b.*\brevision\s+date\b")

# ----------------------------
# MAIN EXTRACT
# ----------------------------
def extract(pdf_path: str) -> dict:
    metadata = extract_metadata_header(pdf_path)
    marking = extract_marking_triplet_from_pdf(pdf_path)
    sec = extract_sections(pdf_path)
   
   
    out = {
        "metadata": metadata,
        "marking": marking,
        "section": sec["section"],
       
        
    }
    # ✅ Revision history di-root hanya kalau ada isi
    rh = extract_revision_history_from_pdf(pdf_path, debug=False)
    if rh:
        out["revision_history"] = rh
    else:
        out = move_revision_unknown_to_root(out)  # cuma jalan kalau rh tidak ada

    if isinstance(out.get("revision_history"), list) and len(out["revision_history"]) == 0:
        out.pop("revision_history", None)


    # ----------------------------
    out = relocate_pilot_block_from_other_sections(out)

    out = move_revision_unknown_to_root(out)
    # 1) buang signature/noise
    out = drop_signature_blocks(out)

    # 2) bersihin bullet "-" (kecuali marker '*')
    out = clean_bullets_everywhere(out)
    out = fix_swapped_specified_patterns_everywhere(out)

    # ✅ 3) split conductor shielding dari conductor
    out = split_conductor_shielding_from_conductor(out)
    out = move_section_key_after(out, "conductor_shielding", "conductor")

    # 4) layers untuk conductor (yang sudah bersih dari shielding)
    out = build_conductor_layers_from_section(out, section_name="conductor")

    # 5) pindahin packing yang nyangkut di final_test -> ke section.packing
    out = split_packing_from_final_test_in_section(out)

    # 6) kalau packing kadang masuk unknown -> pindah
    out = move_packing_from_unknown(out)   # atau postprocess_unknown_to_packing(out), pilih salah satu saja

    # 7) rapihin standard length (kalau masih "500 3601" dll)
    out = fix_packing_standard_length(out)

    # 8) ubah section.packing jadi OBJECT format final (standard_length, net_weight, dst)
    out = build_packing_object_from_section(out)

    # 9) final_test: hilangkan raw_materials (aturan global)
    out = remove_raw_materials_from_final_test(out)

    # 10) final_test cleaning rules kamu
    out = build_final_test_list_from_section(out)
    out = clean_final_test_list_in_section(out)  
    out = fix_final_test_dimension(out)
    out = fix_final_test_noise(out)
    out = fix_final_test_dimension(out)


    #harus paling akhir
    out = fix_specified_order_everywhere(out)
    out = fix_armour_steel_tape_dimension(out)


    # 12) pilot rules (kalau dipakai)
    out = fix_pilot_cable_section(out)

    # ✅ pecah jadi {conductor:{...}, insulation:{...}}
    out = split_pilot_cable_conductor_insulation(out)
    out = build_pilot_conductor_layers(out)

     # ✅ penting: pecah packing yang nyangkut di final_test
    out = split_packing_from_final_test_in_section(out)

     # ✅ jangan tampilkan revision_history kalau kosong / tidak ada isi
    if isinstance(out.get("revision_history"), list) and len(out["revision_history"]) == 0:
        out.pop("revision_history", None)

    return out


def iter_pdfs(path: str):
    # bisa file pdf atau folder
    if os.path.isfile(path) and path.lower().endswith(".pdf"):
        yield path
        return
    for root, _, files in os.walk(path):
        for fn in files:
            if fn.lower().endswith(".pdf"):
                yield os.path.join(root, fn)

def main():
    ap = argparse.ArgumentParser()

    # mode lama (single file)
    ap.add_argument("--pdf", help="Single PDF file")

    # mode baru (folder / file)
    ap.add_argument("--pdf_dir", help="PDF file atau folder berisi PDF")

    # output lokal
    ap.add_argument("--out", default="out.json", help="Output JSON untuk mode --pdf")
    ap.add_argument("--out_dir", default=None, help="Output folder JSON untuk mode --pdf_dir")

    # mongo upload
    ap.add_argument("--mongo_uri", default=None)
    ap.add_argument("--db_name", default=None)
    ap.add_argument("--collection", default="tds_documents")

    args = ap.parse_args()

    # validasi input: wajib salah satu --pdf atau --pdf_dir
    if not args.pdf and not args.pdf_dir:
        ap.error("wajib isi salah satu: --pdf atau --pdf_dir")

    # setup mongo kalau diminta
    col = None
    if args.mongo_uri or args.db_name:
        if not (args.mongo_uri and args.db_name):
            ap.error("kalau mau upload Mongo, wajib isi dua-duanya: --mongo_uri dan --db_name")
        client = MongoClient(args.mongo_uri)
        col = client[args.db_name][args.collection]
        col.create_index([("id", ASCENDING)], unique=True)

    # ------------------------------------------------------------
    # MODE 1: single file --pdf (tetap support seperti sekarang)
    # ------------------------------------------------------------
    if args.pdf:
        pdf_path = args.pdf
        data = extract(pdf_path)

        doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
        data["id"] = doc_id  # penting untuk mongo & tracking

        # simpan json lokal
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print("OK ->", args.out)

        # upload mongo kalau diminta
        if col is not None:
            col.update_one({"id": doc_id}, {"$set": data}, upsert=True)
            print(f"UPLOADED -> Mongo {args.db_name}.{args.collection} id={doc_id}")

        return

    # ------------------------------------------------------------
    # MODE 2: batch --pdf_dir (folder atau file)
    # ------------------------------------------------------------
    base = args.pdf_dir
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    n_ok, n_err = 0, 0
    for pdf_path in iter_pdfs(base):
        try:
            data = extract(pdf_path)
            doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
            data["id"] = doc_id

            # save json per file (opsional)
            if args.out_dir:
                out_path = os.path.join(args.out_dir, f"{doc_id}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

            # upload mongo (opsional)
            if col is not None:
                col.update_one({"id": doc_id}, {"$set": data}, upsert=True)

            n_ok += 1
            print("OK:", doc_id)

        except Exception as e:
            n_err += 1
            print("ERR:", pdf_path, "->", repr(e))

    print(f"DONE. ok={n_ok} err={n_err}")


if __name__ == "__main__":
    main()
