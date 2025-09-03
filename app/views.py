from asyncio.log import logger
import os
import io
import re
import json
from django.conf import settings
from django.http import HttpResponseNotAllowed, JsonResponse ,FileResponse, Http404
from django.shortcuts import render
from asgiref.sync import sync_to_async
import fitz
from app.services.commitment_pdf_generation import generate_commitment_register
from app.services.web_scrapping import PDFDownloader, PDFScraper

from glob import glob
from typing import Dict, Any, List, Tuple
from django.views.decorators.http import require_GET , require_POST
from .services.pdf_parser import  DocumentConfigs, TableValidationConfig, PDFParser, process_pdf_batch, validate_parsing_setup

import logging
import pandas as pd
from .services.commitment_register_analyzer import CommitmentRegisterAnalyzer
import pytesseract

pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD_PATH

PDF_DIR = os.path.join(settings.MEDIA_ROOT, "pdfs")
KB_DIR = os.path.join(settings.MEDIA_ROOT, "kb")
EXPORT_DIR = os.path.join(settings.MEDIA_ROOT, "exports")
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(KB_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

KB_PATH = os.path.join(KB_DIR, "parsed_results.json")


def home(request):
    return render(request, 'app/home.html')


async def scrape_view(request):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])

    base_url = request.POST.get("base_url", "").strip()
    if not base_url:
        return JsonResponse({"error": "Base URL is required"}, status=400)

    try:
        max_sections = int(request.POST.get("max_sections", 10))
    except ValueError:
        max_sections = 10
    headless = "headless" in request.POST

    # 1) Scrape
    scraper = PDFScraper(base_url=base_url, max_sections=max_sections, headless=headless)
    pdf_links = await scraper.run()

    # 2) Download + extract
    output_dir = os.path.join(getattr(settings, "MEDIA_ROOT", "media"), "pdfs")
    downloader = PDFDownloader(output_dir=output_dir, timeout=20)
    pdf_texts = await sync_to_async(downloader.run, thread_sensitive=True)(pdf_links)

    # Return JSON only
    return JsonResponse({
        "pdf_links_count": len(pdf_links),
        "pdf_texts_count": len(pdf_texts),
        "links": pdf_links,  # optional, could be large
        "texts": [{"filename": fn, "snippet": snip} for fn, snip in pdf_texts],  # optional
    })



def status_view(request):
    return JsonResponse({"status": "ok", "app": "app"})




def _build(results_map: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Create a compact, JSON-safe KB entry per document from parser results.
    Avoid heavy page-level details to keep payload small.
    """
    kb: List[Dict[str, Any]] = []

    for _path, result in results_map.items():
        if not isinstance(result, dict) or "error" in result:
            # Skip errored docs here; caller may report separately
            continue

        meta = result.get("metadata", {}) or {}
        summary = result.get("analysis_summary", {}) or {}
        ga = result.get("global_analysis", {}) or {}

        # Pollutants summary (flatten)
        pollutants_summary: List[Dict[str, Any]] = []
        for code, info in (ga.get("pollutants_summary") or {}).items():
            pollutants_summary.append({
                "code": code,
                "name": info.get("name", ""),
                "pages": info.get("pages", []),
                "has_limits": info.get("has_limits", False),
            })

        # Limit values (already JSON-friendly per parser)
        limit_values = ga.get("limit_values_summary", []) or []

        quality = ga.get("document_quality", {}) or {}
        recs = ga.get("extraction_recommendations", []) or []

        # Ensure avg_confidence is JSON-safe float
        try:
            avg_conf = float(summary.get("avg_confidence", 0.0))
        except Exception:
            avg_conf = 0.0

        kb.append({
            "filename": result.get("filename", ""),
            "document_type": result.get("document_type", "autre"),
            "pages": int(meta.get("pages", 0) or 0),
            "avg_confidence": avg_conf,
            "pollutants": pollutants_summary,
            "limit_values": limit_values,
            "quality": quality,
            "recommendations": recs,
        })

    return kb

@require_GET
def process_pdfs_view(request):
    """
    Parse ALL PDFs in MEDIA_ROOT/pdfs using RobustPDFParser + process_pdf_batch.
    Returns a compact knowledge base (KB) JSON.
    """
    pdf_paths = sorted(glob(os.path.join(PDF_DIR, "*.pdf")))
    if not pdf_paths:
        return JsonResponse({"error": "Aucun PDF trouvé à traiter"}, status=404)

    # Default validation config (you can tweak these)
    default_config = TableValidationConfig(
        max_columns=12,
        min_rows=2,
        max_null_percentage=0.4,
        min_content_ratio=0.3,
    )

    batch = process_pdf_batch(pdf_paths, default_config)
    results_map = batch.get("results", {}) or {}
    summary = batch.get("summary", {}) or {}

    kb = _build(results_map)

    # Collect per-file errors to help the UI
    errors: List[Dict[str, Any]] = []
    for path, res in results_map.items():
        if isinstance(res, dict) and "error" in res:
            errors.append({
                "filename": os.path.basename(path),
                "error": res.get("error", "unknown"),
            })

    return JsonResponse({
        "docs_processed": len(kb),
        "kb": kb,
        "errors": errors,
        "summary": {
            "total_files": int(summary.get("total_files", 0) or 0),
            "successful": int(summary.get("successful", 0) or 0),
            "failed": int(summary.get("failed", 0) or 0),
            "avg_confidence": float(summary.get("avg_confidence", 0.0) or 0.0),
            "processing_time": float(summary.get("processing_time", 0.0) or 0.0),
        },
    })

@require_GET
def process_pdf_view(request):
    """
    Parse a SINGLE PDF by filename (query param: ?filename=my.pdf).
    Uses doc-type–aware config after initial classification for better results.
    """
    filename = request.GET.get("filename")
    if not filename:
        return JsonResponse({"error": "Paramètre 'filename' requis"}, status=400)

    full_path = os.path.join(PDF_DIR, filename)
    if not os.path.exists(full_path):
        return JsonResponse({"error": f"Fichier introuvable: {filename}"}, status=404)

    try:
        # First pass: instantiate to classify & analyze
        parser = PDFParser(full_path)

        # Swap in a doc-type–specific config before parsing
        parser.config = DocumentConfigs.get_config(parser.doc_type)

        # Parse with the tuned config
        result = parser.parse()

        kb = _build({full_path: result})
        return JsonResponse({
            "doc": kb[0] if kb else {},
            "raw": result,  # if payload is too big for your UI, remove or trim this
        })

    except Exception as e:
        logger.exception("Erreur lors du parsing du fichier: %s", filename)
        return JsonResponse({"error": str(e)}, status=500)


@require_GET
def parser_validate_view(request):
    """
    Quick health check for parser environment (Tesseract, versions).
    GET /parser/validate/
    """
    info = validate_parsing_setup()
    return JsonResponse(info)


def _load_knowledge_base() -> List[Dict[str, str]]:
    """
    Load KB from KB_PATH if exists; otherwise, build a minimal KB by reading PDFs in PDF_DIR.
    The analyzer expects items with 'filename' and 'content'.
    """
    if os.path.exists(KB_PATH):
        try:
            with open(KB_PATH, "r", encoding="utf-8") as f:
                kb = json.load(f)
            # Normalize shape (accept older schema with 'snippet')
            norm = []
            for item in kb:
                if isinstance(item, dict):
                    filename = item.get("filename", "")
                    content = item.get("content") or item.get("snippet") or ""
                    if filename and content:
                        norm.append({"filename": filename, "content": content})
            if norm:
                return norm
        except Exception:
            pass

    # 2) Fallback: build KB from PDFs
    kb = []
    for path in sorted(glob(os.path.join(PDF_DIR, "*.pdf"))):
        try:
            doc = fitz.open(path)
            try:
                text = "".join(page.get_text() for page in doc)
            finally:
                doc.close()
            # Keep it reasonably small
            text = (text or "")[:8000]
            kb.append({"filename": os.path.basename(path), "content": text})
        except Exception:
            continue
    return kb


def _flatten_df(df: pd.DataFrame) -> tuple[List[str], List[Dict[str, Any]]]:
   
    if isinstance(df.columns, pd.MultiIndex):
        flat_cols = [" | ".join([lvl for lvl in tup if lvl]) for tup in df.columns.to_list()]
        flat_df = df.copy()
        flat_df.columns = flat_cols
    else:
        flat_cols = [str(c) for c in df.columns]
        flat_df = df

    rows = flat_df.fillna("").to_dict(orient="records")
    return flat_cols, rows


def _build_commitments_from_request(payload: Dict[str, Any], analyzer: CommitmentRegisterAnalyzer) -> List[List[str]]:
    cols = list(analyzer.columns)
    ncols = len(cols)
    col_index = {col: i for i, col in enumerate(cols)}

    def empty_row():
        return [""] * ncols

    def set_field(row, col_tuple, value):
        idx = col_index.get(col_tuple)
        if idx is not None:
            row[idx] = value

    # If commitments are provided explicitly, use them
    commitments_payload = payload.get("commitments")
    rows = []
    if isinstance(commitments_payload, list) and commitments_payload:
        for i, item in enumerate(commitments_payload, start=1):
            desc = (item.get("description") or "").strip()
            cid = (item.get("id") or f"C-{i:03d}").strip()
            if not desc:
                continue
            row = empty_row()
            set_field(row, ('Commitment Register Overview', 'Commitment Identifier', ''), cid)
            set_field(row, ('Commitment Register Overview', 'Description', ''), desc)
            rows.append(row)
        if rows:
            return rows

    # Otherwise, derive naive commitments from description (bullets/sentences)
    desc_text = (payload.get("project_description") or "").strip()
    bullets = []
    for line in desc_text.splitlines():
        line = line.strip()
        if re.match(r"^(\d+[KATEX_INLINE_CLOSE.\s]+|[-•*]\s+)", line):
            bullets.append(re.sub(r"^(\d+[KATEX_INLINE_CLOSE.\s]+|[-•*]\s+)", "", line).strip())

    if not bullets:
        # Split into sentences as a fallback
        sentences = re.split(r"(?<=[.!?])\s+", desc_text)
        bullets = [s.strip() for s in sentences if len(s.strip()) > 20][:3]

    if not bullets:
        bullets = [desc_text[:160] + ("..." if len(desc_text) > 160 else "")]

    for i, b in enumerate(bullets, start=1):
        row = empty_row()
        set_field(row, ('Commitment Register Overview', 'Commitment Identifier', ''), f"C-{i:03d}")
        set_field(row, ('Commitment Register Overview', 'Description', ''), b)
        rows.append(row)

    return rows


@require_POST
def analyze_commitments_view(request):

    try:
        payload = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    project_description = (payload.get("project_description") or "").strip()
    if not project_description:
        return JsonResponse({"error": "project_description is required"}, status=400)

    # Prepare analyzer and KB
    analyzer = CommitmentRegisterAnalyzer(project_description=project_description)
    kb = _load_knowledge_base()

    # Prepare commitments data
    commitments_data = _build_commitments_from_request(payload, analyzer)

    # Run analysis (Gemini calls happen inside)
    df_final = analyzer.analyze_commitments(commitments_data, kb)

    # Flatten for UI
    headers, rows = _flatten_df(df_final)

    # Store in session for download
    request.session["commitment_register_rows"] = rows
    request.session["commitment_register_headers"] = headers
    request.session.modified = True

    # Compute avg_confidence from the analyzer summary if present
    avg_confidence = 0.0
    try:
        # df_final has no summary; get it by re-running minimal stats or rely on analyzer logs.
        # Here we skip and just return 0.0; you can enhance the analyzer to return summary out.
        pass
    except Exception:
        pass

    return JsonResponse({
        "result": rows,
        "rows_count": len(rows),
        "avg_confidence": avg_confidence
    })


@require_GET
def download_commitment_register_view(request):
    """
    Download the last analyzed commitment register as an Excel file.
    Uses rows/headers stored in the session by analyze_commitments_view.
    """
    rows = request.session.get("commitment_register_rows")
    headers = request.session.get("commitment_register_headers")
    if not rows or not headers:
        raise Http404("Aucun résultat d'analyse disponible pour téléchargement.")

    df = pd.DataFrame(rows)
    # Ensure column order
    missing = [h for h in headers if h not in df.columns]
    for h in missing:
        df[h] = ""
    df = df[headers]

    # Create Excel in memory
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Commitment Register", index=False)
    buf.seek(0)

    return FileResponse(
        buf,
        as_attachment=True,
        filename="commitment_register.xlsx",
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def _reconstruct_multiindex(headers: List[str]) -> List[Tuple[str, str, str]]:
    """
    Turn flattened headers like "Level1 | Level2 | Level3" back into 3-level tuples.
    If header has only 1 or 2 parts, pad missing levels with ''.
    """
    tuples = []
    for h in headers:
        parts = [p.strip() for p in str(h).split(" | ")]
        while len(parts) < 3:
            parts.append("")
        tuples.append((parts[0], parts[1], parts[2]))
    return tuples



def _reconstruct_multiindex(headers: List[str]) -> List[Tuple[str, str, str]]:
    """
    Turn flattened headers like "Level1 | Level2 | Level3" back into 3-level tuples.
    If header has only 1 or 2 parts, pad missing levels with ''.
    """
    tuples = []
    for h in headers:
        parts = [p.strip() for p in str(h).split(" | ")]
        while len(parts) < 3:
            parts.append("")
        tuples.append((parts[0], parts[1], parts[2]))
    return tuples


@require_GET
def download_commitment_register_view(request):
    """
    Generates and streams a PDF using your service.generate_commitment_register().
    Requires that analyze_commitments_view has stored:
      - session["commitment_register_rows"]: list[dict]
      - session["commitment_register_headers"]: list[str] (flattened headers)
    """
    rows = request.session.get("commitment_register_rows")
    headers = request.session.get("commitment_register_headers")

    if not rows or not headers:
        # No analysis data in session
        raise Http404("Aucun résultat d'analyse disponible. Veuillez d'abord exécuter l'analyse.")

    # Build a flat DataFrame in the exact header order
    df_flat = pd.DataFrame(rows)
    # Ensure all columns exist
    for h in headers:
        if h not in df_flat.columns:
            df_flat[h] = ""
    df_flat = df_flat[headers]  # reorder

    # Rebuild MultiIndex columns
    col_tuples = _reconstruct_multiindex(headers)
    df_flat.columns = pd.MultiIndex.from_tuples(col_tuples)

    # Attach the DataFrame to the request so your service can read request.df_initial
    setattr(request, "df_initial", df_flat)

    # Call your entrypoint (returns a FileResponse or HttpResponse)
    return generate_commitment_register(request)

