from __future__ import annotations
import os
import re
import json
import logging
import argparse
import unicodedata
from datetime import datetime
from difflib import get_close_matches
from collections import defaultdict

# optional libs
try:
    import pdfplumber
except Exception:
    pdfplumber = None
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None
try:
    from PIL import Image
except Exception:
    Image = None
try:
    import pytesseract
except Exception:
    pytesseract = None
try:
    import dateparser
except Exception:
    dateparser = None

# ----- Config -----
CANONICAL_FIELDS = [
    "phase", "activity_aspect", "impacts", "jurisdiction", "type",
    "legal_requirement", "date", "date_iso", "description", "task",
    "responsibility", "comments", "source_file", "source_page", "table_index", "orig_headers"
]

HEADER_KEYWORDS = {
    "phase": ["phase", "project phase"],
    "activity_aspect": ["activity", "aspect", "activity/aspect", "activity / aspect"],
    "impacts": ["impacts", "impact"],
    "jurisdiction": ["jurisdiction", "national", "international", "regional", "local"],
    "type": ["type", "regulation", "law", "decree", "order", "dahir", "standard"],
    "legal_requirement": ["legal requirement", "requirement", "law", "decree", "order", "regulation"],
    "date": ["date", "issued", "published", "effective"],
    "description": ["description", "details", "detail", "summary"],
    "task": ["task", "action", "actions", "what to do"],
    "responsibility": ["responsibility", "responsible", "owner"],
    "comments": ["comments", "notes"]
}

DATE_PATTERNS = [
    re.compile(r'(\d{1,2}\s+[A-Za-z]+(?:\s+\d{4}))'),  # 13 February 2019
    re.compile(r'(\d{1,2}-[A-Za-z]{3}-\d{2,4})'),      # 1-Mar-24
    re.compile(r'([A-Za-z]+\s+\d{1,2},\s*\d{4})'),     # March 1, 2024
    re.compile(r'(\d{1,2}/\d{1,2}/\d{2,4})')           # 01/03/2024
]

TYPE_RE = re.compile(r'\b(Regulation|Guideline|Law|Decree|Order|Dahir|Standard)\b', flags=re.I)
JURISDICTION_RE = re.compile(r'\b(National|International|Regional|Local)\b', flags=re.I)

# ----- Helpers -----
def setup_logging(level=logging.INFO):
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=level)

def normalize_space(s: str | None) -> str | None:
    if s is None:
        return None
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s if s != "" else None

def header_to_field(header_text: str | None) -> str | None:
    if not header_text:
        return None
    h = re.sub(r'[^\w\s/]', '', header_text.lower())
    for fld, kws in HEADER_KEYWORDS.items():
        for kw in kws:
            if kw in h:
                return fld
    # fuzzy fallback
    choices = [kw for kws in HEADER_KEYWORDS.values() for kw in kws]
    match = get_close_matches(h, choices, n=1, cutoff=0.6)
    if match:
        for fld, kws in HEADER_KEYWORDS.items():
            if match[0] in kws:
                return fld
    return None

def parse_date_to_iso(s: str | None) -> str | None:
    if not s:
        return None
    s = s.strip()
    for pat in DATE_PATTERNS:
        m = pat.search(s)
        if m:
            cand = m.group(1)
            if dateparser:
                dt = dateparser.parse(cand, settings={'PREFER_DAY_OF_MONTH': 'first'})
                if dt:
                    return dt.date().isoformat()
            else:
                for fmt in ("%d %B %Y","%d-%b-%y","%d-%b-%Y","%B %d, %Y","%d/%m/%Y","%d/%m/%y"):
                    try:
                        dt = datetime.strptime(cand, fmt)
                        return dt.date().isoformat()
                    except Exception:
                        continue
    if dateparser:
        dt = dateparser.parse(s)
        if dt:
            return dt.date().isoformat()
    return None

# ----- Table extractors -----
def extract_tables_with_pdfplumber(pdf_path: str, start_page: int = 3, ocr_if_empty: bool = False, ocr_lang: str = "eng"):
    """
    Use pdfplumber to extract tables from start_page..end.
    Returns list of canonical-row dicts (only table rows).
    """
    if pdfplumber is None:
        logging.info("pdfplumber not installed; skipping.")
        return []

    rows_out = []
    with pdfplumber.open(pdf_path) as doc:
        total_pages = len(doc.pages)
        start_idx = max(0, start_page - 1)
        for pidx in range(start_idx, total_pages):
            pno = pidx + 1
            page = doc.pages[pidx]
            try:
                tables = page.extract_tables()
            except Exception:
                tables = []
            if not tables:
                logging.debug("No tables found by pdfplumber on page %d", pno)
                continue
            for t_index, table in enumerate(tables):
                if not table:
                    continue
                # normalize rows (list of lists)
                norm_rows = []
                max_cols = 0
                for r in table:
                    if not r:
                        continue
                    cleaned = [normalize_space(c) for c in r]
                    # consider row only if it has at least one non-empty cell
                    if any(cleaned):
                        norm_rows.append(cleaned)
                        max_cols = max(max_cols, len(cleaned))
                if not norm_rows:
                    continue
                # find header row (look at first 3 rows for header words)
                header_idx = None
                for i, r in enumerate(norm_rows[:3]):
                    hits = sum(1 for c in r if c and header_to_field(c))
                    if hits >= 2:
                        header_idx = i
                        break
                if header_idx is None:
                    header_idx = 0
                raw_header = norm_rows[header_idx]
                # ensure header length consistent
                header_cells = raw_header + [None] * (max_cols - len(raw_header))
                mapped = []
                for ci, h in enumerate(header_cells):
                    fld = header_to_field(h) or (f"col_{ci}")
                    mapped.append(fld)
                # iterate data rows
                for data_row in norm_rows[header_idx + 1:]:
                    data_row = data_row + [None] * (max_cols - len(data_row))
                    rowmap = {k: None for k in CANONICAL_FIELDS}
                    for ci, cell in enumerate(data_row):
                        cell_val = normalize_space(cell)
                        fld = mapped[ci] if ci < len(mapped) else f"col_{ci}"
                        if fld in CANONICAL_FIELDS:
                            # join if existing
                            if rowmap.get(fld):
                                rowmap[fld] = (rowmap[fld] + " | " + cell_val) if cell_val else rowmap[fld]
                            else:
                                rowmap[fld] = cell_val
                        else:
                            # unknown columns append to description
                            if cell_val:
                                if rowmap.get("description"):
                                    rowmap["description"] += " | " + cell_val
                                else:
                                    rowmap["description"] = cell_val
                    rowmap["source_file"] = os.path.basename(pdf_path)
                    rowmap["source_page"] = pno
                    rowmap["table_index"] = t_index
                    rowmap["orig_headers"] = header_cells
                    # normalize date_iso
                    if rowmap.get("date"):
                        rowmap["date_iso"] = parse_date_to_iso(rowmap["date"])
                    rows_out.append(rowmap)
    logging.info("pdfplumber produced %d table rows for %s", len(rows_out), os.path.basename(pdf_path))
    return rows_out

def _cluster_x_positions(x_positions, tol=40):
    """Group sorted x positions into cluster centers (simple 1-D clustering)."""
    if not x_positions:
        return []
    x_sorted = sorted(x_positions)
    groups = []
    curr = [x_sorted[0]]
    for x in x_sorted[1:]:
        if x - curr[-1] <= tol:
            curr.append(x)
        else:
            groups.append(curr)
            curr = [x]
    if curr:
        groups.append(curr)
    centers = [sum(g) / len(g) for g in groups]
    return centers

def extract_tables_with_fitz(pdf_path: str, start_page: int = 3, ocr_if_empty: bool = True, ocr_lang: str = "eng"):
    """
    Fallback extractor: use PyMuPDF text blocks (and optional OCR) to reconstruct table rows/columns.
    Only returns rows when a header row can be detected or when there are consistent column centers.
    """
    if fitz is None:
        logging.info("PyMuPDF not installed; skipping fitz fallback.")
        return []

    doc = fitz.open(pdf_path)
    rows_out = []
    for pidx in range(max(0, start_page - 1), len(doc)):
        pno = pidx + 1
        page = doc[pidx]
        try:
            raw_blocks = page.get_text("blocks")
        except Exception:
            raw_blocks = []

        # If no blocks or all empty and OCR is allowed, render image & OCR
        if (not raw_blocks or all(not (b[4] or "").strip() for b in raw_blocks)) and ocr_if_empty and pytesseract and Image:
            try:
                pix = page.get_pixmap(dpi=300)
                mode = "RGBA" if pix.alpha else "RGB"
                img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                ocr_data = pytesseract.image_to_data(img, lang=ocr_lang, output_type=pytesseract.Output.DICT)
                raw_blocks = []
                n = len(ocr_data["text"])
                for i in range(n):
                    txt = ocr_data["text"][i].strip()
                    if not txt:
                        continue
                    left = int(ocr_data["left"][i])
                    top = int(ocr_data["top"][i])
                    width = int(ocr_data["width"][i])
                    height = int(ocr_data["height"][i])
                    raw_blocks.append((left, top, left + width, top + height, txt))
            except Exception as e:
                logging.debug("OCR on page %d failed: %s", pno, e)
                raw_blocks = raw_blocks or []

        # normalize blocks to dicts
        blocks = []
        for b in raw_blocks:
            try:
                x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
            except Exception:
                if len(b) >= 5:
                    x0, y0, x1, y1, text = b[:5]
                else:
                    continue
            text = normalize_space(text)
            if not text:
                continue
            blocks.append({"x0": float(x0), "y0": float(y0), "x1": float(x1), "y1": float(y1), "text": text})

        if not blocks:
            continue

        # group blocks into horizontal rows by y center with a tolerance
        y_tol = 6
        rows = []
        for blk in sorted(blocks, key=lambda z: (z["y0"], z["x0"])):
            cy = (blk["y0"] + blk["y1"]) / 2.0
            placed = False
            for r in rows:
                if abs(r["y"] - cy) <= y_tol:
                    r["blocks"].append(blk)
                    r["y"] = (r["y"] * r["count"] + cy) / (r["count"] + 1)
                    r["count"] += 1
                    placed = True
                    break
            if not placed:
                rows.append({"y": cy, "blocks": [blk], "count": 1})
        # convert row blocks to ordered lists
        row_texts = []
        for r in rows:
            cells = [b["text"] for b in sorted(r["blocks"], key=lambda x: x["x0"])]
            x_centers = [ (b["x0"] + b["x1"]) / 2.0 for b in sorted(r["blocks"], key=lambda x: x["x0"]) ]
            row_texts.append({"page": pno, "cells": cells, "x_centers": x_centers, "blocks": r["blocks"]})

        if not row_texts:
            continue

        # Try to detect header row (first 6 rows)
        header_idx = None
        for i, r in enumerate(row_texts[:6]):
            hits = sum(1 for c in r["cells"] if header_to_field(c))
            if hits >= 2:
                header_idx = i
                break

        # If header not found, try to compute stable column centers from first N rows
        # collect all x_centers from first 10 rows
        all_x = []
        for r in row_texts[:12]:
            all_x.extend(r["x_centers"])
        col_centers = _cluster_x_positions(all_x, tol=40)

        if header_idx is None and (not col_centers or len(col_centers) < 2):
            # couldn't identify a table structure on this page -> skip
            logging.debug("fitz: no clear table structure found on page %d", pno)
            continue

        # if we have a header, derive columns from header block x positions
        if header_idx is not None:
            header_row = row_texts[header_idx]
            # derive column centers from header x_centers (cluster them)
            col_centers = _cluster_x_positions(header_row["x_centers"], tol=40)

        # build table rows mapping: for each row, assign each block to nearest column center
        num_cols = len(col_centers)
        if num_cols < 1:
            continue
        # create orig header labels if header exists
        orig_headers = None
        mapped = None
        if header_idx is not None:
            # map header texts to nearest column center
            header_cells = row_texts[header_idx]["cells"]
            header_xs = row_texts[header_idx]["x_centers"]
            # create mapping from column index -> header text (choose nearest)
            header_for_col = [None] * num_cols
            for htxt, hx in zip(header_cells, header_xs):
                # find nearest col center
                diffs = [abs(hx - cc) for cc in col_centers]
                ci = int(min(range(len(diffs)), key=lambda k: diffs[k]))
                if header_for_col[ci]:
                    header_for_col[ci] += " | " + htxt
                else:
                    header_for_col[ci] = htxt
            orig_headers = header_for_col
            mapped = [ header_to_field(h) or (f"col_{i}") for i, h in enumerate(header_for_col) ]
        else:
            # generic col names
            orig_headers = [f"col_{i}" for i in range(num_cols)]
            mapped = [f"col_{i}" for i in range(num_cols)]

        # iterate rows AFTER header_idx (if header exists) else all rows
        data_rows_iter = row_texts[(header_idx + 1):] if header_idx is not None else row_texts
        for rr in data_rows_iter:
            # prepare empty cells
            cells_by_col = [None] * num_cols
            # assign each block to nearest column
            # use block x center
            for blk in rr["blocks"]:
                cx = (blk["x0"] + blk["x1"]) / 2.0
                diffs = [abs(cx - cc) for cc in col_centers]
                ci = int(min(range(len(diffs)), key=lambda k: diffs[k]))
                txt = normalize_space(blk["text"])
                if not txt:
                    continue
                if cells_by_col[ci]:
                    cells_by_col[ci] += " | " + txt
                else:
                    cells_by_col[ci] = txt
            # if row is entirely empty, skip
            if not any(cells_by_col):
                continue
            # build rowmap using mapped -> CANONICAL_FIELDS
            rowmap = {k: None for k in CANONICAL_FIELDS}
            for ci, val in enumerate(cells_by_col):
                fld = mapped[ci] if ci < len(mapped) else f"col_{ci}"
                if fld in CANONICAL_FIELDS:
                    rowmap[fld] = val if val else None
                else:
                    # append to description
                    if val:
                        if rowmap["description"]:
                            rowmap["description"] += " | " + val
                        else:
                            rowmap["description"] = val
            rowmap["source_file"] = os.path.basename(pdf_path)
            rowmap["source_page"] = pno
            rowmap["table_index"] = 0
            rowmap["orig_headers"] = orig_headers
            if rowmap.get("date"):
                rowmap["date_iso"] = parse_date_to_iso(rowmap["date"])
            rows_out.append(rowmap)

    logging.info("fitz fallback produced %d table rows for %s", len(rows_out), os.path.basename(pdf_path))
    return rows_out

# ----- Postprocess & dedupe -----
def postprocess_entries(entries):
    """Normalize strings, fill date_iso if missing, simple dedupe."""
    out = []
    seen = set()
    for e in entries:
        ee = {}
        for k in CANONICAL_FIELDS:
            v = e.get(k)
            ee[k] = normalize_space(v) if isinstance(v, str) else v
        if not ee.get("date_iso") and ee.get("date"):
            ee["date_iso"] = parse_date_to_iso(ee["date"])
        key = (
            (ee.get("legal_requirement") or "").lower(),
            (ee.get("date_iso") or ee.get("date") or ""),
            f"{ee.get('source_file') or ''}|{ee.get('source_page') or ''}"
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(ee)
    logging.info("Postprocessed: %d unique table rows", len(out))
    return out

# ----- Controller -----
def process_path(input_path: str, output_json: str, start_page: int = 3, ocr_if_empty: bool = False, ocr_lang: str = "eng"):
    files = []
    if os.path.isdir(input_path):
        for f in sorted(os.listdir(input_path)):
            if f.lower().endswith(".pdf"):
                files.append(os.path.join(input_path, f))
    elif os.path.isfile(input_path) and input_path.lower().endswith(".pdf"):
        files = [input_path]
    else:
        raise FileNotFoundError("Provide a PDF file or a directory containing PDFs.")

    all_rows = []
    for pdf in files:
        logging.info("Processing PDF: %s", pdf)
        rows = []
        try:
            rows = extract_tables_with_pdfplumber(pdf, start_page=start_page, ocr_if_empty=ocr_if_empty, ocr_lang=ocr_lang)
        except Exception as e:
            logging.debug("pdfplumber failed for %s: %s", pdf, e)
            rows = []
        # if no rows, attempt fitz fallback
        if not rows:
            try:
                rows = extract_tables_with_fitz(pdf, start_page=start_page, ocr_if_empty=ocr_if_empty, ocr_lang=ocr_lang)
            except Exception as e:
                logging.warning("fitz fallback failed for %s: %s", pdf, e)
                rows = []
        all_rows.extend(rows)

    processed = postprocess_entries(all_rows)
    with open(output_json, "w", encoding="utf-8") as fh:
        json.dump(processed, fh, ensure_ascii=False, indent=2)
    logging.info("Saved %d table rows to %s", len(processed), output_json)
    return processed

# Robust loader for parsing_existing_legal_registers.py
import sys
import importlib.util
from pathlib import Path
from django.conf import settings

_oldreg_module_cache = None

def _load_oldreg_module():
    """
    Try to import `parsing_existing_legal_registers` from the package first.
    If that fails, try to locate the file on disk and load it dynamically.
    Returns the loaded module object, or raises ImportError with a clear message.
    """
    global _oldreg_module_cache
    if _oldreg_module_cache is not None:
        return _oldreg_module_cache

    # 1) Try the canonical package path (if you placed the file under app/services/legal_register/)
    try:
        from app.services.legal_register import parsing_existing_legal_registers as oldreg
        _oldreg_module_cache = oldreg
        return _oldreg_module_cache
    except Exception:
        pass

    # 2) Try dynamic file-based import from a few likely locations
    base = Path(getattr(settings, "BASE_DIR", Path.cwd()))
    candidates = [
        # same folder as pdf_parser if your package exists
        base / "app" / "services" / "legal_register" / "parsing_existing_legal_registers.py",
        # project root (if you dropped the file there)
        base / "parsing_existing_legal_registers.py",
        # alongside this views.py (if you put it next to views)
        Path(__file__).resolve().parent / "parsing_existing_legal_registers.py",
    ]

    for p in candidates:
        if p.exists():
            spec = importlib.util.spec_from_file_location("parsing_existing_legal_registers", str(p))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(mod)
                    _oldreg_module_cache = mod
                    return _oldreg_module_cache
                except Exception as e:
                    raise ImportError(f"Failed to import {p}: {e}") from e

    raise ImportError(
        "Could not import 'parsing_existing_legal_registers'. "
        "Place the file at 'app/services/legal_register/parsing_existing_legal_registers.py' "
        "or at project root, or next to views.py."
    )


# ------- DB integration: list previous legal registers -------
def list_previous_legal_registers_from_db() -> list[str]:
    """
    Return absolute paths of previously uploaded legal register PDFs from DB.
    Reads Document.previous_legal_registers (falls back to source_pdf if needed).
    """
    try:
        from app.models import PreviousLegalRegister
    except Exception:
        PreviousLegalRegister = None  # type: ignore
    try:
        from app.models import Document as DocumentRecord  # fallback
    except Exception:
        DocumentRecord = None  # type: ignore
    out: list[str] = []
    # Prefer dedicated model
    if PreviousLegalRegister is not None:
        try:
            for r in PreviousLegalRegister.objects.all():
                p = (getattr(r, "pdf_path", "") or "").strip()
                if p and p.lower().endswith(".pdf") and os.path.exists(p):
                    out.append(p)
        except Exception:
            pass
    # Fallback to Document.source_pdf if needed
    if not out and DocumentRecord is not None:
        try:
            for d in DocumentRecord.objects.all():
                p = (getattr(d, "source_pdf", "") or "").strip()
                if p and p.lower().endswith(".pdf") and os.path.exists(p):
                    out.append(p)
        except Exception:
            pass
    return out
