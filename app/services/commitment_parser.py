from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import fitz  # PyMuPDF
import pandas as pd


# ------------------------- Header / cell normalization ------------------------- #

def _normalize_header(val: Optional[str]) -> Optional[str]:
    """Normalize a header/cell: strip, collapse spaces, normalize slashes, fix common variants."""
    if val is None:
        return None
    s = str(val).replace("\u00a0", " ").replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)                # collapse spaces
    s = re.sub(r"\s*/\s*", "/", s)            # “Preparation/ construction” → “Preparation/construction”

    # Canonical renames (extend as needed)
    replacements = {
        "Fungible": "Fungibility",
        "Preparation/Construction": "Preparation/construction",
        "Discharge Management": "Discharge management",
        "Health and Safety": "Health & Safety",
        "Health & safety": "Health & Safety",
        "health & safety": "Health & Safety",
        "Affected areas or processes": "Affected Areas or Processes",
        "Impact or hazard addressed": "Impact or Hazard Addressed",
    }
    s = replacements.get(s, s)
    return s


def _ffill(seq: Iterable[Optional[str]]) -> List[Optional[str]]:
    out: List[Optional[str]] = []
    last: Optional[str] = None
    for v in seq:
        if v is not None:
            last = v
        out.append(last)
    return out


def _combine_two_row_headers(raw_df: pd.DataFrame) -> List[str]:
    """
    Build column names by combining the first two rows.
    Forward-fill the top row to identify grouped parents (e.g., 'Affected Areas or Processes', 'Impact'),
    then prefix each child sub-header with its parent.
    """
    n_cols = raw_df.shape[1]
    top = [_normalize_header(raw_df.iat[0, j]) if pd.notna(raw_df.iat[0, j]) else None for j in range(n_cols)]
    sub = [_normalize_header(raw_df.iat[1, j]) if pd.notna(raw_df.iat[1, j]) else None for j in range(n_cols)]

    parents_ff = _ffill(top)
    headers: List[str] = []
    for j in range(n_cols):
        parent = parents_ff[j]
        child = sub[j]

        if parent in ("Affected Areas or Processes",):
            hdr = f"{parent} - {child}" if child else parent
        elif parent == "Impact":
            hdr = f"Impact - {child}" if child else parent
        else:
            # Standalone column: prefer explicit top header, else fallback to sub header
            hdr = _normalize_header(top[j]) or child

        headers.append(_normalize_header(hdr))
    return headers


def _flatten_cell(x) -> str:
    if pd.isna(x):
        return x  # keep NaN for now; caller can replace if needed
    s = str(x).replace("\u00a0", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ------------------------- Core parsing function ------------------------- #

def _best_table_dataframes_on_page(page) -> List[pd.DataFrame]:
    """
    Return candidate table DataFrames for a page, sorted by (rows*cols) desc.
    Filters out tiny tables (header-only or separators).
    """
    dfs: List[Tuple[int, int, pd.DataFrame]] = []
    tf = page.find_tables()
    for t in tf:  # iterate Table objects
        try:
            df = t.to_pandas()
        except Exception:
            continue
        if df is None or df.empty:
            continue
        rows, cols = df.shape
        if rows < 3 or cols < 5:
            continue
        dfs.append((rows * cols, cols, df))

    # Sort by area (rows*cols) desc, then columns desc
    dfs.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [d[2] for d in dfs]


def parse_commitment_register(
    pdf_path: str,
    start_page: int = 3,
    end_page: Optional[int] = None,
) -> pd.DataFrame:
    """
    Extract the Commitment Register table from `start_page` (1-based) through `end_page` (inclusive).
    Defaults to page 3 → last page.

    Returns:
        pandas.DataFrame with one row per commitment and unified column names.
    """
    doc = fitz.open(pdf_path)

    if start_page < 1:
        raise ValueError("start_page must be 1-based (>= 1)")
    start_idx = max(0, start_page - 1)
    if end_page is None:
        end_idx = len(doc) - 1
    else:
        if end_page < start_page:
            raise ValueError("end_page must be >= start_page")
        end_idx = min(len(doc) - 1, end_page - 1)

    frames: List[pd.DataFrame] = []

    for pnum in range(start_idx, end_idx + 1):
        page = doc[pnum]
        candidates = _best_table_dataframes_on_page(page)
        if not candidates:
            continue

        # Take the largest candidate on the page (more robust when small legend tables exist)
        raw_df = candidates[0]

        # Need at least two header rows + body
        if raw_df.shape[0] < 3:
            continue

        headers = _combine_two_row_headers(raw_df)
        body = raw_df.iloc[2:].copy()  # skip the two header rows
        body.columns = headers

        # Drop fully empty rows
        body = body.dropna(how="all")
        if body.empty:
            continue

        # Flatten multiline cells
        for col in body.columns:
            body[col] = body[col].apply(_flatten_cell)

        frames.append(body)

    if not frames:
        return pd.DataFrame()

    # Concatenate pages and normalize final column order (stable across docs)
    result = pd.concat(frames, ignore_index=True)

    # Optional: ensure canonical column ordering if all present
    canonical = [
        "Register Identifier", "Commitment Identifier", "Commitment or Obligation", "Description",
        "Project Phase", "Potential Impact on Scope?", "Status", "Commitment Deadline",
        "First Lead", "Second Lead", "Third Lead",
        "Primary Commitment Documentation", "Impact or Hazard Addressed",
        "Approving Agencies", "Other Stakeholders",
        "Affected Areas or Processes - Preparation/construction",
        "Affected Areas or Processes - Operation",
        "Affected Areas or Processes - Input Management",
        "Affected Areas or Processes - Discharge management",
        "Affected Areas or Processes - Off-Sites",
        "Affected Areas or Processes - Other",
        "Affected Areas or Processes - Fungibility",
        "Impact - CAPEX", "Impact - OPEX", "Impact - Health & Safety",
        "Impact - Social", "Impact - Economic", "Impact - Environmental", "Impact - Regulatory",
        "Comments", "Requires Change Order?",
    ]
    # Reindex if we have a superset of canonical; otherwise keep parsed order
    if set(canonical).issubset(result.columns):
        result = result.reindex(columns=[c for c in canonical if c in result.columns])

    return result


# ------------------------- JSON export + CLI ------------------------- #

def export_commitment_register_to_json(
    pdf_path: str,
    json_path: str,
    compact: bool = False,
    keep_nulls: bool = False,
    start_page: int = 3,
    end_page: Optional[int] = None,
) -> None:
    """
    Parse the register and write JSON.
    - compact: if True, no indentation (smaller file).
    - keep_nulls: if True, preserve nulls; else replace NaN with empty string.
    """
    df = parse_commitment_register(pdf_path, start_page=start_page, end_page=end_page)

    if df.empty:
        records: List[dict] = []
    else:
        if keep_nulls:
            # Keep None for missing values
            records = df.where(pd.notna(df), None).to_dict(orient="records")
        else:
            # Replace NaN with empty strings for friendlier JSON
            df = df.fillna("")
            records = df.to_dict(orient="records")

    out = Path(json_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        if compact:
            json.dump(records, f, ensure_ascii=False, separators=(",", ":"))
        else:
            json.dump(records, f, ensure_ascii=False, indent=2)

