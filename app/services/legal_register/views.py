# views.py
import asyncio
from collections import defaultdict
import json
import logging
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
import tempfile
import time 
from typing import Any, Dict, List, Optional
import uuid
from django.views.decorators.http import require_GET
import json, os, tempfile, time, uuid
from pathlib import Path

from django.conf import settings
from django.http import (
    JsonResponse,
    HttpResponseBadRequest,
    HttpResponseServerError,
    HttpResponse,
    FileResponse,
    Http404,
)
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.utils import timezone
from asgiref.sync import sync_to_async

from app.services.legal_register import pdf_parser
from app.services.legal_register.llm_traitment import run_llm_analysis
from app.services.legal_register.parsing_existing_legal_registers import _load_oldreg_module
from app.services.legal_register.pdf_generation import generate_complete_report_pdf
from app.services.legal_register.pdf_parser import extract_and_parse_pdf


# Import your scraper class (adjust if located elsewhere)
from .web_scrapping import BOPdfScraper
from app.models import ScrapeJob, Document as DocumentRecord, AnalysisResult as AnalysisRecord, GeneratedRegister, ScrapeSource

# Gemini trainer (few-shot ICL)
try:
    _HAS_GEMINI = True
except Exception:
    _HAS_GEMINI = False

logger = logging.getLogger(__name__)

# Default output directory (can be overridden in Django settings)
DEFAULT_OUTPUT_DIR = Path(getattr(settings, "BOPDFSCRAPER_OUTPUT_DIR", "pdfs"))

# ========= Paths for JSON, Few-shot memory =========
JSON_OUTPUT_DIR = Path(getattr(settings, "BOPARSER_JSON_DIR", "json")).resolve()
JSON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FEWSHOT_MEMORY_PATH = Path(
    getattr(settings, "FEWSHOT_MEMORY_PATH", JSON_OUTPUT_DIR / "fewshot_memory.json")
).resolve()
FEWSHOT_MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)



LEGAL_REGISTER_TABLE_DATA_KEY= "legal_register_rows"
LEGAL_REGISTER_HEADERS_KEY = "legal_register_headers"

# Few-shot memory path (already used in your system)
FEWSHOT_MEMORY_PATH = Path("json/fewshot_memory.json")

_HAS_GEMINI = True  # Set to True if Gemini/LLM module available
FIXED_CATEGORIES = [
    "General Environmental & Sustainability Regulations",
    "Solid Waste",
    "Water and Liquid discharges",
    "Noise and vibrations",
    "Energy",
    "Air",
]

# ========= Utils =========


def _to_plain_dict(x):
    if hasattr(x, "model_dump") and callable(x.model_dump):  # pydantic v2
        return x.model_dump()
    if hasattr(x, "dict") and callable(x.dict):              # pydantic v1
        return x.dict()
    if isinstance(x, dict):
        return x
    try:
        return json.loads(json.dumps(x, default=lambda o: getattr(o, "__dict__", str(o))))
    except Exception:
        return {}


def _to_str(v):
    if v is None:
        return ""
    if isinstance(v, (list, tuple, set)):
        return "; ".join(_to_str(x) for x in v if x is not None)
    try:
        return str(v)
    except Exception:
        return ""

def _row_dict_to_table_row(d: Dict[str, Any]) -> List[str]:
    """
    Map any row dict -> fixed 11-cell list for the PDF:
    [Phase, Activity/Aspect, Impacts, Jurisdiction, Type, Legal Requirement, Date, Description, Task, Responsibility, Comments]
    Only these 11 cells are produced (everything else is ignored).
    """
    return [
        _to_str(d.get("phase", "")),
        _to_str(d.get("activity_aspect", "") or d.get("activity", "") or d.get("aspect", "")),
        _to_str(d.get("impacts", "")),
        _to_str(d.get("jurisdiction", "National") or "National"),
        _to_str(d.get("type", "")),
        _to_str(d.get("legal_requirement", "") or d.get("legal_requirement_text", "") or d.get("title", "")),
        _to_str(d.get("date", "") or d.get("publication_date_gregorian", "") or d.get("publication_date_hijri", "")),
        _to_str(d.get("description", "")),
        _to_str(d.get("task", "")),
        _to_str(d.get("responsibility", "")),
        _to_str(d.get("comments", "")),
    ]

def _build_table_structured_sorted(rows_flat: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group rows by their fixed category (category_title) and return them
    in the fixed business order FIXED_CATEGORIES only.
    """
    groups: Dict[str, List[List[str]]] = {c: [] for c in FIXED_CATEGORIES}
    for r in rows_flat:
        cat = r.get("category_title") or ""
        if cat not in FIXED_CATEGORIES:
            cat = "General Environmental & Sustainability Regulations"  # fallback to a valid bucket
        groups[cat].append(_row_dict_to_table_row(r))
    structured = []
    for cat in FIXED_CATEGORIES:
        rows = groups[cat]
        if rows:
            structured.append({"category_title": cat, "rows": rows})
    return structured

def _json_path_for_pdf(pdf_path: Path) -> Path:
    return JSON_OUTPUT_DIR / (Path(pdf_path).stem + ".json")

def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _to_plain_dict(x):
    # Pydantic v2
    if hasattr(x, "model_dump") and callable(x.model_dump):
        return x.model_dump()
    # Pydantic v1
    if hasattr(x, "dict") and callable(x.dict):
        return x.dict()
    if isinstance(x, dict):
        return x
    # Last resort: try to serialize
    try:
        return json.loads(json.dumps(x, default=lambda o: getattr(o, "__dict__", str(o))))
    except Exception:
        return {}
def _resolve_output_dir() -> Path:
    """Helper to return configured output dir path."""
    return Path(getattr(settings, "BOPDFSCRAPER_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))

def _is_within(base: Path, target: Path) -> bool:
    try:
        return os.path.commonpath([str(base), str(target)]) == str(base)
    except Exception:
        return False

def _build_trainer(api_key: Optional[str]):
    if not _HAS_GEMINI:
        raise RuntimeError("Gemini module not available. Install google-generativeai and ensure gemini_training.py exists.")
    trainer = make_trainer(api_key=api_key or os.getenv("GEMINI_API_KEY"))
    # Load few-shot memory if exists
    try:
        if FEWSHOT_MEMORY_PATH.exists():
            trainer.memory.load(str(FEWSHOT_MEMORY_PATH))
    except Exception as e:
        logger.warning("Failed to load few-shot memory: %s", e)
    return trainer

def _persist_memory(trainer) -> None:
    try:
        trainer.memory.save(str(FEWSHOT_MEMORY_PATH))
    except Exception as e:
        logger.error("Failed to save few-shot memory: %s", e)


# ========= Scraper Endpoints  =========

@csrf_exempt
async def start_scrape(request):
    if request.method != "POST":
        return HttpResponseBadRequest("Only POST allowed. Send JSON body.")

    try:
        body = request.body.decode("utf-8") or "{}"
        data = json.loads(body)
    except Exception:
        return HttpResponseBadRequest("Invalid JSON body.")

    base_url: Optional[str] = data.get("base_url")
    max_pages = data.get("max_pages")
    download = bool(data.get("download", True))
    overwrite = bool(data.get("overwrite", False))
    extract_preview = bool(data.get("extract_preview", True))
    headless = bool(data.get("headless", True))
    output_dir = data.get("output_dir", str(DEFAULT_OUTPUT_DIR))

    init_kwargs = {
        "output_dir": output_dir,
        "headless": headless,
        "download_concurrency": int(data.get("download_concurrency", 4)),
        "http_timeout": int(data.get("http_timeout", 30)),
        "max_download_retries": int(data.get("max_download_retries", 3)),
        "pdf_preview_chars": int(data.get("pdf_preview_chars", 1000)),
    }
    if base_url:
        init_kwargs["base_url"] = base_url

    scraper = BOPdfScraper(**init_kwargs)
    # Persist job start
    try:
        job = await sync_to_async(ScrapeJob.objects.create)(
            base_url=base_url or "",
            params=init_kwargs,
            status="running",
            output_dir=str(output_dir),
        )
    except Exception:
        job = None
    try:
        result = await scraper.run(
            max_pages=max_pages,
            download=download,
            overwrite=overwrite,
            extract_preview=extract_preview,
        )
        # Update job end
        try:
            if job is not None:
                def _update():
                    j = ScrapeJob.objects.get(pk=job.pk)
                    j.status = "success"
                    j.finished_at = timezone.now()
                    j.stats = {
                        "pdf_links_count": result.get("pdf_links_count") or (len(result.get("links", [])) if isinstance(result.get("links"), list) else None),
                        "downloaded_count": result.get("downloaded_count") or result.get("pdf_texts_count")
                    }
                    # store links and files if not too large
                    j.links = result.get("links") if isinstance(result.get("links"), list) and len(result.get("links")) <= 1000 else None
                    j.files = result.get("files_downloaded") if isinstance(result.get("files_downloaded"), list) and len(result.get("files_downloaded")) <= 1000 else None
                    j.save()
                await sync_to_async(_update, thread_sensitive=True)()
        except Exception:
            pass
        return JsonResponse(result, safe=False)
    except Exception as exc:
        logger.exception("Scrape failed")
        try:
            if job is not None:
                await sync_to_async(ScrapeJob.objects.filter(pk=job.pk).update)(
                    status="failed", error=str(exc), finished_at=timezone.now()
                )
        except Exception:
            pass
        return HttpResponseServerError(
            json.dumps({"error": str(exc)}), content_type="application/json"
        )
    finally:
        try:
            if hasattr(scraper, "close"):
                await scraper.close()
        except Exception:
            logger.exception("Error while closing scraper session")


def list_files(request):
    outdir = _resolve_output_dir()
    ext = request.GET.get("ext")
    if not outdir.exists():
        return JsonResponse({"files": []})

    files = []
    for p in sorted(outdir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not p.is_file():
            continue
        if ext and not p.name.lower().endswith(ext.lower()):
            continue
        stat = p.stat()
        files.append(
            {
                "name": p.name,
                "path": str(p),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            }
        )
    return JsonResponse({"files": files})


def download_file(request, filename: str):
    outdir = _resolve_output_dir().resolve()
    safe_name = Path(filename).name
    target = (outdir / safe_name).resolve()
    if not (_is_within(outdir, target) and target.exists()):
        return HttpResponse(status=404)
    return FileResponse(open(target, "rb"), as_attachment=True, filename=safe_name)


@csrf_exempt
def cleanup_files(request):
    keep = request.GET.get("keep_latest") or (request.POST.get("keep_latest") if request.method == "POST" else None)
    keep_latest = None
    if keep is not None:
        try:
            keep_latest = int(keep)
        except Exception:
            return HttpResponseBadRequest("keep_latest must be an integer")

    outdir = str(_resolve_output_dir())
    tmp = BOPdfScraper(output_dir=outdir)
    try:
        tmp.cleanup_output_dir(keep_latest=keep_latest)
        return JsonResponse({"status": "ok", "kept_latest": keep_latest})
    except Exception as exc:
        logger.exception("Cleanup failed")
        return HttpResponseServerError(json.dumps({"error": str(exc)}), content_type="application/json")


@require_GET
def list_scrape_sources(request):
    """Return predefined scraping sources (name + url) for UI dropdowns."""
    try:
        sources = [
            {"id": s.id, "name": s.name, "url": s.url}
            for s in ScrapeSource.objects.all().order_by('name')
        ]
    except Exception:
        sources = []
    return JsonResponse({"sources": sources})


def preview_html(request):
    outdir = _resolve_output_dir()
    files = []
    if outdir.exists():
        for p in sorted(outdir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if p.is_file():
                files.append(p.name)

    base_path = request.build_absolute_uri("/").rstrip("/")
    lines = ["<html><head><meta charset='utf-8'><title>BO PDFs</title></head><body>"]
    lines.append("<h1>Downloaded PDFs</h1>")
    if not files:
        lines.append("<p><em>No files found.</em></p>")
    else:
        lines.append("<ul>")
        for name in files:
            url = f"{base_path}/bo_scrape/file/{name}"
            lines.append(f"<li><a href=\"{url}\">{name}</a></li>")
        lines.append("</ul>")
    lines.append("</body></html>")
    return HttpResponse("\n".join(lines), content_type="text/html; charset=utf-8")


def dashboard(request):
    return render(request, 'app/legal_register.html')

# ========= Upload and parse old legal registers =========
@csrf_exempt
def upload_old_register(request):
    """
    Upload an old legal register and parse it to canonical rows using your
    parsing_existing_legal_registers.process_path() (PDF) or accept JSON as-is.

    Accepts: .pdf  (parsed to rows via process_path)
             .json (copied into JSON_OUTPUT_DIR; if it contains rows we pass them back)

    Response (success):
      {
        "status": "ok",
        "kind": "pdf|json",
        "saved_json_file": "old_register_YYYYmmdd_HHMMSS.json",
        "rows_count": <int or null>,
        "parsed": {"rows": [...] }   # for immediate analyze via parsed_list
      }
    """
    if request.method != "POST":
        return HttpResponseBadRequest("Only POST allowed.")
    if "file" not in request.FILES:
        return HttpResponseBadRequest("No file uploaded.")

    try:
        oldreg = _load_oldreg_module()
    except ImportError as e:
        return JsonResponse({"error": str(e)}, status=500)

    up = request.FILES["file"]
    name = Path(up.name).name
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""

    # Save upload to MEDIA/uploads
    upload_dir = _resolve_upload_dir()
    upload_dir.mkdir(parents=True, exist_ok=True)
    src_path = upload_dir / name
    with open(src_path, "wb") as fh:
        for chunk in up.chunks():
            fh.write(chunk)

    kind = None
    saved_json_basename = None
    parsed_for_inline = None
    rows_count = None

    try:
        if ext == "pdf":
            kind = "pdf"
            # Use your process_path on a single file, produce canonical rows JSON
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_json = JSON_OUTPUT_DIR / f"old_register_{ts}.json"
            rows = oldreg.process_path(str(src_path), str(out_json), start_page=3, ocr_if_empty=False, ocr_lang="eng")
            saved_json_basename = out_json.name
            parsed_for_inline = {"rows": rows}
            rows_count = len(rows or [])

        elif ext == "json":
            kind = "json"
            try:
                payload = json.loads(src_path.read_text(encoding="utf-8"))
            except Exception as e:
                return JsonResponse({"error": f"Invalid JSON: {e}"}, status=400)
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_json = JSON_OUTPUT_DIR / f"old_register_{ts}.json"
            _write_json(out_json, payload)
            saved_json_basename = out_json.name
            # try to expose rows for immediate analysis
            if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
                parsed_for_inline = {"rows": payload["rows"]}
                rows_count = len(payload["rows"])
            elif isinstance(payload, list):
                parsed_for_inline = {"rows": payload}
                rows_count = len(payload)

        else:
            return HttpResponseBadRequest("Unsupported file type. Use .pdf or .json.")

        # (Optional) register in DB list for convenience
        try:
            if saved_json_basename:
                DocumentRecord.objects.update_or_create(
                    filename=saved_json_basename,
                    defaults={
                        "doc_type": "old_register",
                        "json_path": str(JSON_OUTPUT_DIR / saved_json_basename),
                        "source_pdf": str(src_path) if ext == "pdf" else "",
                        "status": "uploaded",
                    }
                )
                # Also record in PreviousLegalRegister for a dedicated admin/model section
                try:
                    from app.models import PreviousLegalRegister
                    PreviousLegalRegister.objects.create(
                        filename=saved_json_basename or name,
                        pdf_path=str(src_path) if ext == "pdf" else "",
                        json_path=str(JSON_OUTPUT_DIR / saved_json_basename),
                        rows_count=rows_count or 0,
                        status="uploaded",
                    )
                except Exception:
                    pass
        except Exception:
            pass

        return JsonResponse({
            "status": "ok",
            "kind": kind,
            "saved_json_file": saved_json_basename,
            "rows_count": rows_count,
            "parsed": parsed_for_inline,
        }, json_dumps_params={"ensure_ascii": False, "indent": 2})

    except Exception as e:
        logger.exception("upload_old_register failed")
        return JsonResponse({"error": str(e)}, status=500)

# ========= Parsing API (unchanged) =========
PDF_STORAGE = os.path.join(settings.BASE_DIR, "data", "pdfs")
JSON_STORAGE = os.path.join(settings.BASE_DIR, "data", "jsons")
os.makedirs(PDF_STORAGE, exist_ok=True)
os.makedirs(JSON_STORAGE, exist_ok=True)

@csrf_exempt
def parse_upload(request):
    """
    Upload a PDF, parse it, return structured JSON.
    """
    if request.method != "POST":
        return HttpResponseBadRequest("Only POST method is allowed.")

    if "file" not in request.FILES:
        return HttpResponseBadRequest("No PDF file uploaded.")

    uploaded_file = request.FILES["file"]
    file_path = os.path.join(PDF_STORAGE, uploaded_file.name)

    # Save uploaded PDF
    with open(file_path, "wb") as f:
        for chunk in uploaded_file.chunks():
            f.write(chunk)

    return _parse_and_return(file_path)


def parse_file(request, filename):
    """
    Parse an existing PDF by filename (scraped/stored already).
    """
    file_path = os.path.join(PDF_STORAGE, filename)

    if not os.path.exists(file_path):
        raise Http404(f"PDF file '{filename}' not found.")

    return _parse_and_return(file_path)

@csrf_exempt
def parse_dir(request):
    """
    Parse PDFs from the scraper output directory (by default) and return aggregated results.
    - POST JSON options:
        - pdf_dir: optional path to PDFs; defaults to scraper output dir
        - recursive: bool (default True)
        - write_json: bool (default True)
        - skip_up_to_date: bool (default True)
    Response:
      {
        "count": <number of docs included>,
        "results": [ { "filename": ..., "parsed": {...}, "json_file": "...", "source_pdf": "...", "status": "parsed|cached" }, ... ],
        "pdf_dir": "...",
        "json_dir": "..."
      }
    """
    # Defaults
    pdf_dir = _resolve_output_dir().resolve()
    recursive = True
    write_json = True
    skip_up_to_date = True

    # Read POST body if provided
    if request.method == "POST" and request.body:
        try:
            body = json.loads(request.body.decode("utf-8") or "{}")
            if body.get("pdf_dir"):
                # Allow overriding directory (optional)
                pdf_dir = Path(body["pdf_dir"]).resolve()
            recursive = bool(body.get("recursive", True))
            write_json = bool(body.get("write_json", True))
            skip_up_to_date = bool(body.get("skip_up_to_date", True))
        except Exception as e:
            logger.warning("parse_dir: invalid JSON body (%s). Using defaults.", e)

    # Fallback to legacy storage if scraper dir is missing
    if not pdf_dir.exists():
        legacy = Path(PDF_STORAGE).resolve()
        if legacy.exists():
            pdf_dir = legacy

    if not pdf_dir.exists():
        return JsonResponse(
            {"count": 0, "results": [], "pdf_dir": str(pdf_dir), "json_dir": str(JSON_OUTPUT_DIR.resolve())},
            json_dumps_params={"ensure_ascii": False, "indent": 2}
        )

    # Collect PDFs
    it = pdf_dir.rglob("*.pdf") if recursive else pdf_dir.glob("*.pdf")
    pdf_paths = sorted(it, key=lambda p: p.stat().st_mtime, reverse=False)

    results = []
    parsed_count = 0

    for p in pdf_paths:
        try:
            json_path = _json_path_for_pdf(p)

            # If we can skip and re-use cached JSON, do that (but still include in response)
            if skip_up_to_date and json_path.exists() and json_path.stat().st_mtime >= p.stat().st_mtime:
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        cached = json.load(f)
                    results.append({
                        "filename": p.name,
                        "parsed": cached,
                        "json_file": str(json_path),
                        "source_pdf": str(p),
                        "status": "cached",
                    })
                    parsed_count += 1
                    continue
                except Exception as cache_err:
                    logger.warning("Failed to read cached JSON for %s: %s. Re-parsing...", p, cache_err)

            # Parse fresh
            parsed = extract_and_parse_pdf(str(p))
            if write_json:
                _write_json(json_path, parsed)

            results.append({
                "filename": p.name,
                "parsed": parsed,
                "json_file": str(json_path) if write_json else None,
                "source_pdf": str(p),
                "status": "parsed",
            })
            parsed_count += 1

        except Exception as e:
            logger.error("Failed parsing %s: %s", p, e)
            results.append({
                "filename": p.name,
                "error": str(e),
                "source_pdf": str(p),
                "status": "error",
            })

    payload = {
        "count": parsed_count,
        "results": results,
        "pdf_dir": str(pdf_dir),
        "json_dir": str(JSON_OUTPUT_DIR.resolve()),
    }
    # Persist/update Document records (best-effort)
    try:
        for item in results:
            fn = item.get("filename")
            if not fn:
                continue
            defaults = {
                "doc_type": (item.get("parsed") or {}).get("document_type", ""),
                "pages_count": (item.get("parsed") or {}).get("metadata", {}).get("pages", 0) or 0,
                "avg_confidence": (item.get("parsed") or {}).get("analysis_summary", {}).get("avg_confidence", 0.0) or 0.0,
                "json_path": item.get("json_file") or "",
                "source_pdf": item.get("source_pdf") or "",
                "status": item.get("status") or "",
                "error": item.get("error") or "",
            }
            obj, _ = DocumentRecord.objects.update_or_create(
                filename=fn, defaults=defaults
            )
            # Store full JSON content if available
            parsed = item.get("parsed")
            if isinstance(parsed, dict):
                try:
                    DocumentRecord.objects.filter(pk=obj.pk).update(json_content=parsed)
                except Exception:
                    pass
    except Exception:
        logger.exception("Failed to persist Document records")

    return JsonResponse(payload, safe=False, json_dumps_params={"ensure_ascii": False, "indent": 2})


@require_GET
def list_db_docs(request):
    """List parsed documents stored in DB for selection in the UI."""
    try:
        docs_qs = DocumentRecord.objects.all().order_by('-created_at')
        # Optional filters
        doc_type = request.GET.get('doc_type')
        if doc_type:
            docs_qs = docs_qs.filter(doc_type__iexact=doc_type)
        name_q = request.GET.get('q')
        if name_q:
            docs_qs = docs_qs.filter(filename__icontains=name_q)
        docs = []
        for d in docs_qs[:500]:
            docs.append({
                'id': d.id,
                'filename': d.filename,
                'doc_type': d.doc_type,
                'pages_count': d.pages_count,
                'avg_confidence': d.avg_confidence,
                'json_file': os.path.basename(d.json_path or ''),
                'created_at': d.created_at.isoformat() if d.created_at else None,
            })
        return JsonResponse({'docs': docs})
    except Exception as e:
        logger.exception('list_db_docs error')
        return JsonResponse({'docs': [], 'error': str(e)})


@csrf_exempt
def save_structured_view(request):
    """Save structured table data to session so the PDF generator can use edited rows."""
    if request.method != 'POST':
        return HttpResponseBadRequest('POST expected')
    try:
        body = json.loads(request.body.decode('utf-8') or '{}')
    except Exception:
        body = {}
    structured = body if isinstance(body, list) else body.get('structured_data')
    if not isinstance(structured, list):
        return HttpResponseBadRequest('structured_data array required')
    try:
        request.session[LEGAL_REGISTER_TABLE_DATA_KEY] = structured
        request.session.modified = True
        return JsonResponse({'status': 'ok', 'categories': len(structured), 'rows': sum(len(c.get('rows', [])) for c in structured)})
    except Exception as e:
        logger.exception('save_structured_view error')
        return JsonResponse({'error': str(e)}, status=500)

def deps_report(request):
    """
    Generate a dependency report across parsed PDFs (example: list ministries/departments).
    """
    report = {}

    for fname in os.listdir(PDF_STORAGE):
        if fname.lower().endswith(".pdf"):
            try:
                file_path = os.path.join(PDF_STORAGE, fname)
                parsed = extract_and_parse_pdf(file_path)

                # Example: collect ministries from metadata
                ministry = parsed.get("metadata", {}).get("ministry", "Unknown")
                report.setdefault(ministry, 0)
                report[ministry] += 1
            except Exception as e:
                logger.error(f"Deps report failed for {fname}: {e}")

    return JsonResponse(report, json_dumps_params={"ensure_ascii": False, "indent": 2})


# ----------- Helpers -----------

def _parse_and_return(file_path):
    """Helper to parse and return JSON with error handling."""
    try:
        parsed_data = extract_and_parse_pdf(file_path)
        return JsonResponse(parsed_data, safe=False, json_dumps_params={"ensure_ascii": False, "indent": 2})
    except Exception as e:
        logger.exception(f"Error while parsing {file_path}")
        return HttpResponseBadRequest(f"Error while parsing {os.path.basename(file_path)}: {e}")

def _list_dir(dirpath: Path, ext: Optional[str] = None) -> Dict[str, Any]:
    if not dirpath.exists():
        return {"files": []}
    files = []
    for p in sorted(dirpath.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not p.is_file():
            continue
        if ext and not p.name.lower().endswith(ext.lower()):
            continue
        st = p.stat()
        files.append({
            "name": p.name,
            "path": str(p),
            "size": st.st_size,
            "mtime": st.st_mtime,
        })
    return {"files": files}

def list_pdfs(request):
    pdf_dir = _resolve_output_dir().resolve()
    ext = request.GET.get("ext", ".pdf")
    return JsonResponse(_list_dir(pdf_dir, ext=ext))

def list_jsons(request):
    ext = request.GET.get("ext", ".json")
    return JsonResponse(_list_dir(JSON_OUTPUT_DIR, ext=ext))


def download_pdf(request, filename: str):
    pdf_dir = _resolve_output_dir().resolve()
    safe = Path(filename).name
    target = (pdf_dir / safe).resolve()
    if not (_is_within(pdf_dir, target) and target.exists()):
        return HttpResponse(status=404)
    return FileResponse(open(target, "rb"), as_attachment=True, filename=safe)

def download_json(request, filename: str):
    safe = Path(filename).name
    target = (JSON_OUTPUT_DIR / safe).resolve()
    if not (_is_within(JSON_OUTPUT_DIR, target) and target.exists()):
        return HttpResponse(status=404)
    return FileResponse(open(target, "rb"), as_attachment=True, filename=safe)

# ========= LLM Legal Register Parsing =========
from .pdf_parser import extract_and_parse_pdf

def _is_within(base: Path, other: Path) -> bool:
    try:
        other.resolve().relative_to(base.resolve())
        return True
    except Exception:
        return False

def _resolve_upload_dir() -> Path:
    base = Path(getattr(settings, "MEDIA_ROOT", "media")) / "uploads"
    base.mkdir(parents=True, exist_ok=True)
    return base

# ---------- UI view (renders your HTML template) ----------
def legal_register_dashboard(request):
    return render(request, "legal_register_dashboard.html", {})  # place your HTML here

# ---------- LLM API ----------
@csrf_exempt
def analyze_legal_register_llm(request):
    """
    POST JSON:
    {
      "description": "...",
      "parsed": {...} | "parsed_list": [ {...}, ... ] | "json_files": ["file1.json", ...],
      "use_gemini": true,
      "relevance_mode": "hybrid",
      "history_rows": [ ...canonical rows... ],
      "history_json_files": [ "/mnt/data/legal_register_extracted (2).json" ],
      ...
    }
    """
    rid = uuid.uuid4().hex[:8]
    t0 = time.perf_counter()

    def as_bool(v, default=False):
        if v is None: return default
        if isinstance(v, bool): return v
        return str(v).strip().lower() in ("1","true","yes","y","on")

    content_type = (request.META.get("CONTENT_TYPE") or "").lower()
    description = ""
    parsed_docs: List[Dict[str, Any]] = []

    controls: Dict[str, Any] = {
        "use_llm_fill": True,
        "relevance_mode": "hybrid",
        "relevance_threshold": 0.12,
        "relevance_gray_margin": 0.06,
        "relevance_keep_top_k": 30,
        "max_llm_relevance": 10,
        "max_llm_classify": 20,
        "max_llm_rows": 25,
        "api_key": None,
        # history defaults
        "history_relevance_boost": 0.08,
        "prefer_history_fields": True,
        "merge_strategy": "history_then_llm",
        "history_rows": [],
        "history_json_files": [],
    }

    # If you want auto-history when client doesn't pass anything:
    default_history = getattr(settings, "HISTORY_JSON_DEFAULT_PATHS", [
        "/mnt/data/legal_register_extracted (2).json",  # <— your uploaded path
    ])
    controls["history_json_files"].extend([p for p in default_history if isinstance(p, str)])

    if "application/json" in content_type:
        try:
            data = json.loads(request.body.decode("utf-8") or "{}")
        except Exception:
            return JsonResponse({"error": "Invalid JSON body"}, status=400)

        description = (data.get("description") or "").strip()
        if not description:
            return JsonResponse({"error": "description is required"}, status=400)

        controls["use_llm_fill"] = as_bool(data.get("use_gemini"), True)
        for k in ("relevance_mode","relevance_threshold","relevance_gray_margin",
                  "relevance_keep_top_k","max_llm_relevance","max_llm_classify","max_llm_rows"):
            if k in data:
                controls[k] = data[k]
        controls["api_key"] = (data.get("api_key") or "").strip() or None

        # history incoming
        if isinstance(data.get("history_rows"), list):
            controls["history_rows"] = [r for r in data["history_rows"] if isinstance(r, dict)]
        if isinstance(data.get("history_json_files"), list):
            controls["history_json_files"].extend([str(p) for p in data["history_json_files"]])

        # parsed sources
        if isinstance(data.get("parsed"), dict):
            parsed_docs.append({"parsed": data["parsed"], "source": "inline"})
        elif isinstance(data.get("parsed_list"), list):
            for i, doc in enumerate(data["parsed_list"]):
                if isinstance(doc, dict):
                    parsed_docs.append({"parsed": doc, "source": f"inline[{i}]"})
        elif isinstance(data.get("json_files"), list):
            for name in data["json_files"]:
                safe_name = Path(name).name
                path = (JSON_OUTPUT_DIR / safe_name).resolve()
                if _is_within(JSON_OUTPUT_DIR, path) and path.exists():
                    with open(path, "r", encoding="utf-8") as f:
                        parsed_docs.append({"parsed": json.load(f), "source": str(path)})
        else:
            for p in sorted(JSON_OUTPUT_DIR.glob("*.json")):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        parsed_docs.append({"parsed": json.load(f), "source": str(p)})
                except Exception:
                    pass

    else:
        uploaded = request.FILES.get("file")
        description = (request.POST.get("description") or "").strip()
        if not description:
            return JsonResponse({"error": "description is required"}, status=400)

        controls["use_llm_fill"] = as_bool(request.POST.get("use_gemini"), True)
        controls["relevance_mode"] = (request.POST.get("relevance_mode") or controls["relevance_mode"]).strip().lower()
        for k in ("relevance_threshold","relevance_gray_margin","relevance_keep_top_k",
                  "max_llm_relevance","max_llm_classify","max_llm_rows"):
            if request.POST.get(k) is not None:
                try:
                    controls[k] = type(controls[k])(request.POST.get(k))
                except Exception:
                    pass
        controls["api_key"] = (request.POST.get("api_key") or "").strip() or None

        # optional history_json_files via form field
        if request.POST.get("history_json_files"):
            try:
                controls["history_json_files"].extend(json.loads(request.POST["history_json_files"]))
            except Exception:
                pass

        if uploaded:
            pdf_dir = _resolve_upload_dir().resolve()
            pdf_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = (pdf_dir / Path(uploaded.name).name).resolve()
            with tempfile.NamedTemporaryFile(dir=str(pdf_dir), delete=False) as tmp:
                for chunk in uploaded.chunks():
                    tmp.write(chunk)
                tmp_path = Path(tmp.name).resolve()
            os.replace(tmp_path, pdf_path)
            try:
                parsed_json = extract_and_parse_pdf(str(pdf_path))
            except Exception as e:
                return JsonResponse({"error": f"PDF parsing failed: {e}"}, status=500)
            parsed_docs.append({"parsed": parsed_json, "source": str(pdf_path)})
        else:
            for p in sorted(JSON_OUTPUT_DIR.glob("*.json")):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        parsed_docs.append({"parsed": json.load(f), "source": str(p)})
                except Exception:
                    pass

    if not parsed_docs:
        return JsonResponse({"error": "No parsed documents found to analyze."}, status=400)

    table_structured, stats = run_llm_analysis(
        description=description,
        parsed_docs=parsed_docs,
        controls=controls,
    )

    request.session[LEGAL_REGISTER_TABLE_DATA_KEY] = table_structured
    request.session.modified = True

    try:
        results_path = JSON_OUTPUT_DIR / "results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(table_structured, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    try:
        if AnalysisRecord is not None:
            rows_count = sum(len(c.get("rows", [])) for c in table_structured)
            kb_sources = [str(item.get("source")) for item in parsed_docs if isinstance(item, dict) and item.get("source")]
            AnalysisRecord.objects.create(
                description=description,
                structured_data=table_structured,
                raw_result=table_structured,
                kb_snapshot=kb_sources,
                rows_count=rows_count,
                status="success",
                meta=stats if hasattr(AnalysisRecord, "meta") else None,  # harmless if field absent
            )
    except Exception:
        pass

    dt = time.perf_counter() - t0
    stats["elapsed_sec"] = round(dt, 2)
    return JsonResponse(table_structured, safe=False, json_dumps_params={"ensure_ascii": False, "indent": 2})

@csrf_exempt
def download_legal_register_pdf(request):
    rows = request.session.get(LEGAL_REGISTER_TABLE_DATA_KEY)
    if not rows:
        raise Http404("No legal register available. Run LLM analysis first.")
    try:
        exports_dir = Path(getattr(settings, "MEDIA_ROOT", "media")) / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        pdf_path = exports_dir / f"legal_register_{ts}.pdf"
        generate_complete_report_pdf(rows, str(pdf_path))
        file_size = pdf_path.stat().st_size if pdf_path.exists() else 0
        with open(pdf_path, "rb") as fh:
            pdf_bytes = fh.read()
        try:
            if GeneratedRegister is not None:
                GeneratedRegister.objects.create(
                    kind="legal",
                    file_path=str(pdf_path),
                    file_size=file_size,
                    rows_count=sum(len(c.get("rows", [])) for c in rows) if isinstance(rows, list) else 0,
                    filename=pdf_path.name,
                    pdf_data=pdf_bytes,
                )
        except Exception:
            pass
        response = HttpResponse(pdf_bytes, content_type="application/pdf")
        response["Content-Disposition"] = f'attachment; filename="{pdf_path.name}"'
        return response
    except Exception as e:
        return JsonResponse({"error": f"PDF generation failed: {e}"}, status=500)


