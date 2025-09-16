from asyncio.log import logger
import os
import io
import re
import json
from tempfile import NamedTemporaryFile
import traceback
from django.conf import settings
from django.http import HttpResponseNotAllowed, JsonResponse ,FileResponse, Http404, HttpResponse
from django.shortcuts import render
from django.shortcuts import get_object_or_404
from asgiref.sync import sync_to_async
from django.urls import reverse
import fitz
from app.services.commitment_pdf_generation import generate_commitment_register
from app.services.web_scrapping import PDFDownloader, PDFScraper, load_ministry_data_from_db
from django.views.decorators.csrf import csrf_exempt
from glob import glob
from typing import Dict, Any, List, Tuple
from django.views.decorators.http import require_GET , require_POST
from .services.pdf_parser import  DocumentConfigs, TableValidationConfig, PDFParser, process_pdf_batch
import re
import fitz
import pandas as pd
import json
from pathlib import Path
from typing import Optional, List, Any, Dict
import logging
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
    return render(request, 'app/index.html')

def commitment(request):
    return render(request, 'app/commitment.html')

def legal_register(request):
    """Render the Legal Register wizard page."""
    return render(request, 'app/legal_register.html')


@require_GET
def download_generated_register_view(request, pk: int):
    """
    Stream a generated register PDF stored in DB (fallback to file_path if needed).
    URL: /registers/<pk>/download/
    """
    from app.models import GeneratedRegister

    reg = get_object_or_404(GeneratedRegister, pk=pk)
    filename = reg.filename or f"{reg.kind}_register_{reg.created_at:%Y%m%d_%H%M%S}.pdf"

    # Prefer DB blob
    if reg.pdf_data:
        resp = HttpResponse(reg.pdf_data, content_type="application/pdf")
        resp["Content-Disposition"] = f'attachment; filename="{filename}"'
        return resp

    # Fallback to stored file_path if present and exists
    if reg.file_path and os.path.exists(reg.file_path):
        return FileResponse(open(reg.file_path, "rb"), as_attachment=True, filename=os.path.basename(reg.file_path))

    raise Http404("No PDF available for this generated register.")


@csrf_exempt
async def scrape_view(request):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])

    # Accept one or multiple base URLs (multi-select)
    base_urls = request.POST.getlist("base_urls") or []
    single = (request.POST.get("base_url", "").strip() or None)
    if single:
        base_urls.append(single)
    # Dedup and sanitize
    base_urls = [u.strip() for u in base_urls if u.strip()]
    base_urls = list(dict.fromkeys(base_urls))
    if not base_urls:
        # Fallback to DB-stored sources
        try:
            pairs = load_ministry_data_from_db()
            base_urls = [u for (_n, u) in pairs if u]
        except Exception:
            base_urls = []
    if not base_urls:
        return JsonResponse({"error": "Select at least one website (or configure sources)."}, status=400)

    try:
        max_sections = int(request.POST.get("max_sections", 10))
    except ValueError:
        max_sections = 10
    headless = "headless" in request.POST or True

    all_links = set()
    # 1) Scrape per site and aggregate
    for url in base_urls:
        scraper = PDFScraper(base_url=url, max_sections=max_sections, headless=headless)
        links = await scraper.run()
        for l in links:
            all_links.add(l)

    # 2) Download + extract (once for all)
    output_dir = os.path.join(getattr(settings, "MEDIA_ROOT", "media"), "pdfs")
    downloader = PDFDownloader(output_dir=output_dir, timeout=20)
    pdf_texts = await sync_to_async(downloader.run, thread_sensitive=True)(sorted(all_links))

    return JsonResponse({
        "pdf_links_count": len(all_links),
        "pdf_texts_count": len(pdf_texts),
        "links": sorted(all_links),
        "texts": [{"filename": fn, "snippet": snip} for fn, snip in pdf_texts],
        "sources": base_urls,
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

# --- Helper Functions ---

def _detect_hardware_and_model():
    """
    Auto-detects available hardware (CPU/GPU/MPS) and selects the best LayoutLM model.
    Returns a dictionary with device and model name.
    """
    hardware_info = {"device": "cpu", "model_name": None}
    try:
        import torch
        if torch.cuda.is_available():
            hardware_info["device"] = "cuda:0"
            logger.info("GPU (CUDA) détecté. Utilisation pour l'inférence.")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            hardware_info["device"] = "mps"
            logger.info("Apple MPS détecté. Utilisation pour l'inférence.")
        else:
            logger.info("Pas de GPU détecté. Utilisation du CPU pour l'inférence.")
    except Exception:
        logger.warning("PyTorch non installé ou indisponible. Le parser fonctionnera en mode fallback.")

    # Allow env override to force a specific model
    env_override = os.getenv("LAYOUTLM_MODEL_NAME")
    if env_override:
        hardware_info["model_name"] = env_override
        logger.info(f"Using model from env LAYOUTLM_MODEL_NAME: {env_override}")
        return hardware_info

    local_model_path = os.path.join(
        settings.BASE_DIR, 'models', 'layoutlmv3-base-finetuned-publaynet'
    )

    def _is_valid_model_dir(path: str) -> bool:
        try:
            if not os.path.isdir(path):
                return False
            has_config = os.path.isfile(os.path.join(path, "config.json"))
            has_weights = os.path.isfile(os.path.join(path, "pytorch_model.bin")) or \
                          os.path.isfile(os.path.join(path, "model.safetensors"))
            return has_config and has_weights
        except Exception:
            return False

    if _is_valid_model_dir(local_model_path):
        hardware_info["model_name"] = local_model_path
        logger.info(f"Using local LayoutLM model from: {local_model_path}")
    else:
        if os.path.isdir(local_model_path):
            logger.warning(
                f"Local model directory exists but is not a valid HF repo (missing config/weights): {local_model_path}. "
                "Falling back to a remote fine-tuned model."
            )
        else:
            logger.info(f"Local model not found at path: {local_model_path}. Falling back to a remote fine-tuned model.")
        hardware_info["model_name"] = os.getenv(
            "LAYOUTLM_FALLBACK_REMOTE",
            "HYPJUDY/layoutlmv3-base-finetuned-publaynet"
        )
    return hardware_info


def _build_kb(results_map: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Builds a compact knowledge base from the raw parsing results.
    This function is now adapted to the output of the new LayoutLM-based parser.
    """
    kb: List[Dict[str, Any]] = []
    for path, result in results_map.items():
        if not isinstance(result, dict) or "error" in result:
            continue

        global_analysis = result.get("global_analysis", {})

        all_tables = []
        for page in result.get("pages", []):
            tables_on_page = page.get("content", {}).get("tables", [])
            if tables_on_page:
                for tbl in tables_on_page:
                    tbl["page_number"] = page.get("page_number")
                all_tables.extend(tables_on_page)

        doc_entry = {
            "filename": os.path.basename(path),
            "doc_type": result.get("document_type", "unknown"),
            "pages_count": result.get("metadata", {}).get("pages", 0),
            "avg_confidence": result.get("analysis_summary", {}).get("avg_confidence", 0.0),
            "quality": global_analysis.get("document_quality", {}),
            "pollutants": global_analysis.get("pollutants_summary", {}),
            "limit_values": global_analysis.get("limit_values_summary", []),
            "tables": all_tables,
            "recommendations": global_analysis.get("extraction_recommendations", [])
        }
        kb.append(doc_entry)
    return kb


@require_GET
def process_pdfs_view(request):
    """
    Parses ALL PDFs using the LayoutLM-based PDFParser.
    Auto-detects hardware and returns a compact knowledge base (KB) JSON.
    """
    # Expect a PDF_DIR constant; otherwise derive from settings
    try:
        PDF_DIR = settings.PDF_DIR  # type: ignore
    except Exception:
        PDF_DIR = os.path.join(settings.MEDIA_ROOT, "pdfs")

    pdf_paths = sorted(glob(os.path.join(PDF_DIR, "*.pdf")))
    if not pdf_paths:
        return JsonResponse({"error": "Aucun PDF trouvé à traiter"}, status=404)

    hardware = _detect_hardware_and_model()
    default_config = TableValidationConfig()

    batch = process_pdf_batch(
        pdf_files=pdf_paths,
        config=default_config,
        layout_model_name=hardware["model_name"],
        layout_device=hardware["device"]
    )

    results_map = batch.get("results", {})
    summary = batch.get("summary", {})
    kb = _build_kb(results_map)

    errors = [
        {"filename": os.path.basename(path), "error": res.get("error", "unknown")}
        for path, res in results_map.items() if isinstance(res, dict) and "error" in res
    ]

    return JsonResponse({
        "docs_processed": len(kb),
        "kb": kb,
        "errors": errors,
        "summary": {
            "total_files": summary.get("total_files", 0),
            "successful": summary.get("successful", 0),
            "failed": summary.get("failed", 0),
            "avg_confidence": summary.get("avg_confidence", 0.0),
            "processing_time": summary.get("processing_time", 0.0),
            "hardware_used": {
                "device": hardware["device"],
                "model": hardware["model_name"]
            }
        },
    })


@require_GET
def process_pdf_view(request):
    """
    Parses a SINGLE PDF using a document-type-aware config with LayoutLM.
    """
    # Expect a PDF_DIR constant; otherwise derive from settings
    try:
        PDF_DIR = settings.PDF_DIR  # type: ignore
    except Exception:
        PDF_DIR = os.path.join(settings.MEDIA_ROOT, "pdfs")

    filename = request.GET.get("filename")
    if not filename:
        return JsonResponse({"error": "Paramètre 'filename' requis"}, status=400)

    full_path = os.path.join(PDF_DIR, filename)
    if not os.path.exists(full_path):
        return JsonResponse({"error": f"Fichier introuvable: {filename}"}, status=404)

    try:
        hardware = _detect_hardware_and_model()

        # Fast classification without loading the heavy model
        initial_parser = PDFParser(full_path)
        doc_type = initial_parser.doc_type

        tuned_table_config = DocumentConfigs.get_config(doc_type)

        final_parser = PDFParser(
            full_path,
            config=tuned_table_config,
            layout_model_name=hardware["model_name"],
            layout_device=hardware["device"]
        )

        result = final_parser.parse()
        kb = _build_kb({full_path: result})

        response_payload = {
            "doc": kb[0] if kb else {},
        }
        if request.GET.get("include_raw", "false").lower() == "true":
            response_payload["raw"] = result

        return JsonResponse(response_payload)

    except Exception as e:
        logger.exception("Erreur lors du parsing du fichier: %s", filename)
        return JsonResponse({
            "error": str(e),
            "traceback": traceback.format_exc() if getattr(settings, "DEBUG", False) else "An internal error occurred."
        }, status=500)

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
    Load KB from KB_PATH if present; else build from PDFs (quick text scrape).
    Shape: [{'filename': str, 'content': str}, ...]
    """
    if os.path.exists(KB_PATH):
        try:
            with open(KB_PATH, "r", encoding="utf-8") as f:
                kb = json.load(f)
            norm = []
            for item in kb:
                if isinstance(item, dict):
                    filename = (item.get("filename") or "").strip()
                    content = (item.get("content") or item.get("snippet") or "").strip()
                    if filename and content:
                        norm.append({"filename": filename, "content": content})
            if norm:
                return norm
        except Exception:
            pass

    # Fallback: build from PDFs
    kb = []
    for path in sorted(glob(os.path.join(PDF_DIR, "*.pdf"))):
        try:
            doc = fitz.open(path)
            try:
                text = "".join(page.get_text() for page in doc)
            finally:
                doc.close()
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
    """
    Build matrix of rows using the analyzer's column structure.
    We set only 'Commitment Identifier' and 'Description' here; everything else blank.
    """
    cols = list(analyzer.columns)
    ncols = len(cols)
    col_index = {col: i for i, col in enumerate(cols)}

    def empty_row() -> List[str]:
        return [""] * ncols

    def set_field(row: List[str], col_tuple: Tuple[str, str, str], value: str):
        idx = col_index.get(col_tuple)
        if idx is not None:
            row[idx] = value

    rows: List[List[str]] = []

    # 1) If explicit commitments provided
    commitments_payload = payload.get("commitments")
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

    # 2) Else, derive naive commitments from project_description (bullets/sentences)
    desc_text = (payload.get("project_description") or "").strip()

    # fix: robust bullet regex (numbers like "1. ", dashes, bullets)
    bullets = []
    for line in desc_text.splitlines():
        line = line.strip()
        if re.match(r"^(\d+[\.\)]\s+|[-•*]\s+)", line):
            bullets.append(re.sub(r"^(\d+[\.\)]\s+|[-•*]\s+)", "", line).strip())

    if not bullets:
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
    """
    Run the LLM analysis.
    - project_description is REQUIRED (returns 400 if missing/empty).
    - Results stored in session for Excel/PDF endpoints.
    """
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    project_description = (payload.get("project_description") or "").strip()
    if not project_description:
        return JsonResponse({"error": "project_description is required"}, status=400)

    try:
        analyzer = CommitmentRegisterAnalyzer(project_description=project_description)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)

    kb = _load_knowledge_base()
    commitments_data = _build_commitments_from_request(payload, analyzer)

    try:
        df_final = analyzer.analyze_commitments(commitments_data, kb)
    except Exception as e:
        logger.exception("Analysis failed")
        return JsonResponse({"error": f"Analysis failed: {e}"}, status=500)

    headers, rows = _flatten_df(df_final)

    # Persist to session for download endpoints
    request.session["commitment_register_rows"] = rows
    request.session["commitment_register_headers"] = headers
    request.session.modified = True

    return JsonResponse({
        "result": rows,
        "rows_count": len(rows),
        "avg_confidence": 0.0,  # placeholder; analyzer does not compute it
    })



@require_POST
def save_commitment_results_view(request):

    try:
        payload = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    # Accept both shapes
    rows = payload.get("rows")
    headers = payload.get("headers")
    if rows is None and isinstance(payload.get("results"), list):
        rows = payload["results"]  # tolerate {results: [...]}

    if not isinstance(rows, list):
        return JsonResponse({"error": "'rows' must be a list of objects (or provide 'results')"}, status=400)

    # Clean & normalize rows → list[dict[str, str]]
    cleaned_rows = []
    for row in rows:
        if not isinstance(row, dict):
            return JsonResponse({"error": "Each row must be an object"}, status=400)
        cleaned = {}
        for k, v in row.items():
            cleaned[str(k)] = "" if v is None else (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else str(v))
        cleaned_rows.append(cleaned)

    # Determine headers
    if isinstance(headers, list) and headers:
        normalized_headers = [str(h) for h in headers]
    else:
        session_headers = request.session.get("commitment_register_headers") or []
        if session_headers:
            normalized_headers = [str(h) for h in session_headers]
        elif cleaned_rows:
            # infer from first row
            normalized_headers = list(cleaned_rows[0].keys())
        else:
            normalized_headers = []

    # Order columns when we know the headers
    if normalized_headers:
        ordered_rows = [{h: row.get(h, "") for h in normalized_headers} for row in cleaned_rows]
    else:
        ordered_rows = cleaned_rows

    # Persist to session for PDF/Excel
    request.session["commitment_register_rows"] = ordered_rows
    request.session["commitment_register_headers"] = normalized_headers
    request.session.modified = True

    return JsonResponse({"status": "ok", "rows_count": len(ordered_rows), "headers_count": len(normalized_headers)})

def _reconstruct_multiindex(headers: List[str]) -> List[Tuple[str, str, str]]:
    """
    Turn flattened headers ("Level1 | Level2 | Level3") back into 3-level tuples.
    """
    tuples: List[Tuple[str, str, str]] = []
    for h in headers:
        parts = [p.strip() for p in str(h).split(" | ")]
        while len(parts) < 3:
            parts.append("")
        tuples.append((parts[0], parts[1], parts[2]))
    return tuples


@require_GET
def download_commitment_register_excel_view(request):
    """
    Download last analyzed register as Excel.
    (Kept separate from PDF route to avoid name collisions.)
    """
    rows = request.session.get("commitment_register_rows")
    headers = request.session.get("commitment_register_headers")
    if not rows or not headers:
        raise Http404("Aucun résultat d'analyse disponible pour téléchargement.")

    df = pd.DataFrame(rows)
    for h in headers:
        if h not in df.columns:
            df[h] = ""
    df = df[headers]

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


@require_GET
def download_commitment_register_view(request):
    """
    Generate & stream the PDF using your unchangeable generator.
    Requires analyze_commitments_view to have populated session state.
    """
    rows = request.session.get("commitment_register_rows")
    headers = request.session.get("commitment_register_headers")

    if not rows or not headers:
        raise Http404("Aucun résultat d'analyse disponible. Veuillez d'abord exécuter l'analyse.")

    df_flat = pd.DataFrame(rows)
    for h in headers:
        if h not in df_flat.columns:
            df_flat[h] = ""
    df_flat = df_flat[headers]

    col_tuples = _reconstruct_multiindex(headers)
    df_flat.columns = pd.MultiIndex.from_tuples(col_tuples)

    # Attach for your generator (expects request.df_initial)
    setattr(request, "df_initial", df_flat)

    return generate_commitment_register(request)


# ============================================================
# Commitment Register Parser Integration
# ============================================================


def _normalize_header(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    s = str(val).replace("\u00a0", " ").replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*/\s*", "/", s)
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
    return replacements.get(s, s)

def _ffill(seq: List[Optional[str]]) -> List[Optional[str]]:
    out, last = [], None
    for v in seq:
        if v is not None:
            last = v
        out.append(last)
    return out

def _combine_two_row_headers(raw_df: pd.DataFrame) -> List[str]:
    n_cols = raw_df.shape[1]
    top = [_normalize_header(raw_df.iat[0, j]) if pd.notna(raw_df.iat[0, j]) else None for j in range(n_cols)]
    sub = [_normalize_header(raw_df.iat[1, j]) if pd.notna(raw_df.iat[1, j]) else None for j in range(n_cols)]
    parents_ff = _ffill(top)
    headers: List[str] = []
    for j in range(n_cols):
        parent, child = parents_ff[j], sub[j]
        if parent == "Affected Areas or Processes":
            hdr = f"{parent} - {child}" if child else parent
        elif parent == "Impact":
            hdr = f"Impact - {child}" if child else parent
        else:
            hdr = _normalize_header(top[j]) or child
        headers.append(_normalize_header(hdr))
    return headers

def _flatten_cell(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).replace("\u00a0", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", s).strip()

# ------------------------- Core Parser ------------------------- #

def parse_commitment_register(pdf_path: str, start_page: int = 3, end_page: Optional[int] = None) -> pd.DataFrame:
    doc = fitz.open(pdf_path)
    start_idx = max(0, start_page - 1)
    end_idx = len(doc) - 1 if end_page is None else min(len(doc) - 1, end_page - 1)
    frames: List[pd.DataFrame] = []

    for pnum in range(start_idx, end_idx + 1):
        page = doc[pnum]
        tables = page.find_tables()
        if not tables:
            continue
        raw_df = max([t.to_pandas() for t in tables if t.to_pandas().shape[0] >= 3], key=lambda df: df.shape[0]*df.shape[1], default=None)
        if raw_df is None or raw_df.empty:
            continue
        headers = _combine_two_row_headers(raw_df)
        body = raw_df.iloc[2:].copy()
        body.columns = headers
        body = body.dropna(how="all")
        if body.empty:
            continue
        for col in body.columns:
            body[col] = body[col].apply(_flatten_cell)
        frames.append(body)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def export_commitment_register_to_json(pdf_path: str, json_path: str, compact: bool = False, keep_nulls: bool = False):
    df = parse_commitment_register(pdf_path)
    if df.empty:
        records: List[Dict[str, Any]] = []
    else:
        if keep_nulls:
            records = df.where(pd.notna(df), None).to_dict(orient="records")
        else:
            df = df.fillna("")
            records = df.to_dict(orient="records")

    out = Path(json_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        if compact:
            json.dump(records, f, ensure_ascii=False, separators=(",", ":"))
        else:
            json.dump(records, f, ensure_ascii=False, indent=2)

    return out

# ------------------------- Django View ------------------------- #

@require_GET
def parse_commitment_register_view(request):
    """
    Parse a PDF Commitment Register (page 3 → end) and return JSON.
    Example: GET /parse_commitments/?filename=Q37440-00-EN-REG-00002.pdf
    """
    filename = request.GET.get("filename")
    if not filename:
        return JsonResponse({"error": "Paramètre 'filename' requis"}, status=400)

    pdf_path = os.path.join(PDF_DIR, filename)
    if not os.path.exists(pdf_path):
        return JsonResponse({"error": f"Fichier introuvable: {filename}"}, status=404)

    try:
        df = parse_commitment_register(pdf_path)
        rows = df.fillna("").to_dict(orient="records")
        return JsonResponse({
            "rows_count": len(rows),
            "result": rows,
        })
    except Exception as e:
        logger.exception("Erreur lors du parsing du fichier commitment register")
        return JsonResponse({"error": str(e)}, status=500)


@require_GET
def list_commitment_registers_view(request):
    """
    List previously generated commitment registers stored in DB.
    Returns light metadata + a download URL.
    """
    from app.models import GeneratedRegister

    qs = GeneratedRegister.objects.filter(kind__iexact="commitment").order_by("-created_at")[:200]
    docs = []
    for r in qs:
        size = None
        try:
            if r.pdf_data:
                size = len(r.pdf_data)
            elif r.file_path and os.path.exists(r.file_path):
                size = os.path.getsize(r.file_path)
        except Exception:
            pass

        docs.append({
            "id": r.pk,
            "filename": r.filename or f"commitment_{r.pk}.pdf",
            "created_at": getattr(r, "created_at", None).isoformat() if getattr(r, "created_at", None) else None,
            "size": size,
            "download_url": reverse("download_generated_register", kwargs={"pk": r.pk}),
        })
    return JsonResponse({"docs": docs})


@require_POST
def upload_commitment_register_view(request):
    """
    Upload previous commitment registers (PDF or JSON).
    - PDF: parsed into rows using parse_commitment_register
    - JSON: either list[dict] or {"rows":[...]}
    The last parsed file is loaded into the session so the user can edit/export immediately.
    """
    files = request.FILES.getlist("files") or ([request.FILES["file"]] if "file" in request.FILES else [])
    if not files:
        return JsonResponse({"error": "No file provided (accepts PDF or JSON)"}, status=400)

    total_rows = 0
    last_rows: List[Dict[str, Any]] = []

    for f in files:
        name = (f.name or "").lower()

        if name.endswith(".json"):
            try:
                data = json.load(f)
            except Exception as e:
                return JsonResponse({"error": f"Invalid JSON: {e}"}, status=400)
            rows = data.get("rows") if isinstance(data, dict) else (data if isinstance(data, list) else [])
            if not isinstance(rows, list):
                rows = []
            rows = [r for r in rows if isinstance(r, dict)]
            total_rows += len(rows)
            last_rows = rows or last_rows

        elif name.endswith(".pdf"):
            # Save to a temp file then parse
            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                for chunk in f.chunks():
                    tmp.write(chunk)
                tmp_path = tmp.name
            try:
                df = parse_commitment_register(tmp_path)
                rows = df.fillna("").to_dict(orient="records")
                total_rows += len(rows)
                last_rows = rows or last_rows
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        else:
            # ignore unknown extensions
            continue

    # If we parsed anything, store to session so the UI can display & export
    if last_rows:
        headers = list(last_rows[0].keys())
        request.session["commitment_register_rows"] = last_rows
        request.session["commitment_register_headers"] = headers
        request.session.modified = True

    return JsonResponse({
        "uploaded": len(files),
        "rows_count": total_rows,
        "preview_rows": last_rows[:200],
    })


@require_GET
def load_commitment_register_view(request, pk: int):
    """
    Load a previously generated commitment register from DB (PDF), parse it,
    and push rows/headers into the session for immediate edit/export.
    """
    from app.models import GeneratedRegister

    reg = get_object_or_404(GeneratedRegister, pk=pk)
    pdf_bytes = None

    if reg.pdf_data:
        pdf_bytes = reg.pdf_data
    elif reg.file_path and os.path.exists(reg.file_path):
        with open(reg.file_path, "rb") as fh:
            pdf_bytes = fh.read()
    else:
        raise Http404("No PDF found for this register.")

    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        df = parse_commitment_register(tmp_path)
        rows = df.fillna("").to_dict(orient="records")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    headers = list(rows[0].keys()) if rows else []
    request.session["commitment_register_rows"] = rows
    request.session["commitment_register_headers"] = headers
    request.session.modified = True

    return JsonResponse({"rows_count": len(rows), "result": rows})
