# views.py
import asyncio
from collections import defaultdict
import json
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import tempfile
import time 
import traceback
from typing import Any, Dict, List, Optional
import uuid

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

from app.services.legal_register import pdf_parser
from app.services.legal_register.llm_traitment import  make_trainer
from app.services.legal_register.pdf_generation import generate_complete_report_pdf
from app.services.legal_register.pdf_parser import *
from app.services.legal_register.pdf_parser import _detect_type
from app.services.legal_register.pdf_parser import _extract_number


# Import your scraper class (adjust if located elsewhere)
from .web_scrapping import BOPdfScraper

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
# def _parse_json_body(request) -> Dict[str, Any]:
#     if not request.body:
#         return {}
#     try:
#         return json.loads(request.body.decode("utf-8"))
#     except Exception:
#         return {}

# def _to_bool(val: Any, default: bool = False) -> bool:
#     if isinstance(val, bool):
#         return val
#     if val is None:
#         return default
#     s = str(val).strip().lower()
#     return s in ("1", "true", "yes", "y", "on")

# def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
#     path.parent.mkdir(parents=True, exist_ok=True)
#     tmp = path.with_suffix(path.suffix + ".tmp")
#     with open(tmp, "w", encoding="utf-8") as f:
#         json.dump(data, f, ensure_ascii=False, indent=2)
#     os.replace(tmp, path)

# def _build_service_from_params(params: Dict[str, Any]) -> BOPdfParsingService:
#     dpi = int(params.get("dpi", DEFAULT_DPI))
#     force_ocr = _to_bool(params.get("force_ocr", False), default=False)
#     cache_dir = str(params.get("cache_dir", DEFAULT_CACHE_DIR))
#     return BOPdfParsingService(cache_dir=cache_dir, dpi=dpi, force_ocr=force_ocr)

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
    try:
        result = await scraper.run(
            max_pages=max_pages,
            download=download,
            overwrite=overwrite,
            extract_preview=extract_preview,
        )
        return JsonResponse(result, safe=False)
    except Exception as exc:
        logger.exception("Scrape failed")
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
    return render(request, 'app/legal.html')


# ========= Parsing API (unchanged) =========

# @csrf_exempt
# def parse_upload(request):
#     if request.method != "POST":
#         return HttpResponseBadRequest("Only POST allowed. Send multipart/form-data.")

#     uploaded = request.FILES.get("file")
#     if not uploaded:
#         return HttpResponseBadRequest("Missing file field 'file' (multipart/form-data).")

#     overwrite = _to_bool(request.POST.get("overwrite"), default=False)
#     write_json = _to_bool(request.POST.get("write_json"), default=True)
#     svc = _build_service_from_params(request.POST)

#     safe_name = Path(uploaded.name).name
#     if not safe_name.lower().endswith(".pdf"):
#         return HttpResponseBadRequest("Only .pdf files are accepted.")

#     pdf_dir = _resolve_output_dir().resolve()
#     pdf_dir.mkdir(parents=True, exist_ok=True)
#     dest = pdf_dir / safe_name

#     if dest.exists() and not overwrite:
#         return JsonResponse({"error": "File already exists", "file": safe_name}, status=409)

#     with NamedTemporaryFile(dir=str(pdf_dir), delete=False) as tmp:
#         for chunk in uploaded.chunks():
#             tmp.write(chunk)
#         tmp_path = Path(tmp.name)
#     os.replace(tmp_path, dest)

#     try:
#         data = svc.parse_pdf(str(dest))
#         json_path = None
#         if write_json:
#             out_path = JSON_OUTPUT_DIR / f"{dest.stem}.json"
#             _atomic_write_json(out_path, data)
#             json_path = str(out_path)

#         return JsonResponse({
#             "ok": True,
#             "file": safe_name,
#             "saved_path": str(dest),
#             "json_path": json_path,
#             "data": data,
#         })
#     except Exception as exc:
#         logger.exception("Parse failed for upload: %s", dest)
#         return JsonResponse({"ok": False, "error": str(exc)}, status=500)


# @csrf_exempt
# def parse_file(request, filename: Optional[str] = None):
#     if request.method not in ("POST", "GET"):
#         return HttpResponseBadRequest("Only GET or POST allowed.")

#     body = _parse_json_body(request)
#     write_json = _to_bool(body.get("write_json"), default=True)
#     if not filename:
#         filename = body.get("filename")
#     if not filename:
#         return HttpResponseBadRequest("Missing filename.")

#     safe_name = Path(filename).name
#     if not safe_name.lower().endswith(".pdf"):
#         return HttpResponseBadRequest("Only .pdf files are accepted.")

#     pdf_dir = _resolve_output_dir().resolve()
#     pdf_path = (pdf_dir / safe_name).resolve()
#     if not (_is_within(pdf_dir, pdf_path) and pdf_path.exists()):
#         return HttpResponseBadRequest("PDF not found or invalid filename.")

#     svc = _build_service_from_params(body)

#     try:
#         data = svc.parse_pdf(str(pdf_path))
#         json_path = None
#         if write_json:
#             out_path = (JSON_OUTPUT_DIR / f"{pdf_path.stem}.json").resolve()
#             if not _is_within(JSON_OUTPUT_DIR, out_path):
#                 return HttpResponseBadRequest("Invalid JSON output path resolution.")
#             _atomic_write_json(out_path, data)
#             json_path = str(out_path)

#         return JsonResponse({
#             "ok": True,
#             "file": safe_name,
#             "pdf_path": str(pdf_path),
#             "json_path": json_path,
#             "data": data,
#         })
#     except Exception as exc:
#         logger.exception("Parse failed for file: %s", pdf_path)
#         return JsonResponse({"ok": False, "error": str(exc)}, status=500)


# @csrf_exempt
# def parse_dir(request):
#     if request.method != "POST":
#         return HttpResponseBadRequest("Only POST allowed. Send JSON body.")

#     body = _parse_json_body(request)
#     pattern = body.get("pattern", "*.pdf")
#     recursive = _to_bool(body.get("recursive"), default=False)
#     limit = int(body.get("limit", 0) or 0)
#     write_json = _to_bool(body.get("write_json"), default=True)
#     max_workers = int(body.get("max_workers", 0) or 0)
#     filenames = body.get("filenames")

#     pdf_dir = _resolve_output_dir().resolve()
#     pdf_dir.mkdir(parents=True, exist_ok=True)

#     files: List[Path] = []
#     if isinstance(filenames, list) and filenames:
#         for name in filenames:
#             p = (pdf_dir / Path(name).name).resolve()
#             if _is_within(pdf_dir, p) and p.exists() and p.suffix.lower() == ".pdf":
#                 files.append(p)
#     else:
#         if recursive:
#             files = [p for p in pdf_dir.rglob(pattern) if p.is_file() and p.suffix.lower() == ".pdf"]
#         else:
#             files = [p for p in pdf_dir.glob(pattern) if p.is_file() and p.suffix.lower() == ".pdf"]

#     files = sorted(files)
#     if limit and limit > 0:
#         files = files[:limit]

#     if not files:
#         return JsonResponse({"count": 0, "results": []})

#     svc = _build_service_from_params(body)
#     out_dir = str(JSON_OUTPUT_DIR) if write_json else None

#     if max_workers and max_workers > 0:
#         results = svc.parse_many([str(p) for p in files], output_dir=out_dir, max_workers=max_workers)
#     else:
#         results = svc.parse_many([str(p) for p in files], output_dir=out_dir)

#     return JsonResponse({"count": len(files), "results": results})


# def deps_report(request):
#     svc = BOPdfParsingService()
#     return JsonResponse(svc.dependency_report())
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
    return JsonResponse(payload, safe=False, json_dumps_params={"ensure_ascii": False, "indent": 2})

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

# Session key used by the PDF download view
LEGAL_REGISTER_TABLE_DATA_KEY = "legal_register_table_structured"

def _to_plain_dict(x):
    # Pydantic v2
    if hasattr(x, "model_dump") and callable(x.model_dump):
        return x.model_dump()
    # Pydantic v1
    if hasattr(x, "dict") and callable(x.dict):
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

def _extract_nested(d: Dict[str, Any], *keys: str) -> Optional[str]:
    def get_path(obj, path):
        cur = obj
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
        return cur
    for k in keys:
        v = get_path(d, k)
        if v not in (None, ""):
            return _to_str(v)
    return None

def _extract_dates_from_row_and_doc(row: Dict[str, Any], doc: Dict[str, Any]) -> str:
    g = (
        row.get("date")
        or row.get("publication_date_gregorian")
        or _extract_nested(doc, "metadata.date_gregorian", "bulletin_metadata.date_gregorian")
        or ""
    )
    h = (
        row.get("publication_date_hijri")
        or _extract_nested(doc, "metadata.date_hijri", "bulletin_metadata.date_hijri")
        or ""
    )
    g = _to_str(g).strip()
    h = _to_str(h).strip()
    if g and h:
        return f" Hijri: {h} | {g}"
    return g or h or ""


def _row_dict_to_table_row(d: Dict[str, Any]) -> List[str]:
    """
    Map any row dict -> fixed 11-cell list for the PDF:
    [Phase, Activity/Aspect, Impacts, Jurisdiction, Type, Legal Requirement, Date, Description, Task, Responsibility, Comments]
    """
    return [
        _to_str(d.get("phase", "")),
        _to_str(d.get("activity_aspect", "") or d.get("activity", "") or d.get("aspect", "")),
        _to_str(d.get("impacts", "")),
        _to_str(d.get("jurisdiction", "National") or "National"),
        _to_str(d.get("type", "")),
        _to_str(d.get("legal_requirement", "") or d.get("legal_requirement_text", "") or d.get("title", "")),
        _to_str(d.get("date", "")),
        _to_str(d.get("description", "")),
        _to_str(d.get("task", "")),
        _to_str(d.get("responsibility", "")),
        _to_str(d.get("comments", "")),
    ]

def _build_table_structured(rows_flat: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: List[Dict[str, Any]] = []
    index: Dict[str, int] = {}
    for r in rows_flat:
        cat = _to_str(r.get("category_title") or "Uncategorized")
        if cat not in index:
            index[cat] = len(grouped)
            grouped.append({"category_title": cat, "rows": []})
        grouped[index[cat]]["rows"].append(_row_dict_to_table_row(r))
    return grouped

# -------------------- Relevance utilities --------------------

_FR_STOP = {
    "le","la","les","de","des","du","un","une","et","en","dans","sur","pour","par","au","aux","avec",
    "d","l","à","ou","que","qui","se","son","sa","ses","ces","ce","cette","leurs","leur","nos","notre",
    "est","sont","été","être","fait","faites","afin","ainsi","comme","plus","moins","sans","sous","entre"
}
_EN_STOP = {
    "the","of","and","in","to","for","by","on","at","with","a","an","or","as","is","are","be","been",
    "from","this","that","these","those","it","its","their","there","which","such","also","including"
}
_STOP = _FR_STOP | _EN_STOP

def _normalize_text(s: str) -> str:
    s = _to_str(s).lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")  # strip diacritics
    s = re.sub(r"[^a-z0-9\s\-_/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tokenize(s: str) -> set:
    norm = _normalize_text(s)
    return set(t for t in norm.split() if len(t) >= 3 and t not in _STOP)

def _item_text_for_relevance(row: Dict[str, Any]) -> str:
    parts = [
        row.get("category_title", ""),
        row.get("type", ""),
        row.get("legal_requirement_raw", ""),
        row.get("description_raw", ""),
    ]
    return " ".join(_to_str(p) for p in parts if p)

def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def _heuristic_relevance_score(project_desc: str, row: Dict[str, Any]) -> float:
    dtoks = _tokenize(project_desc)
    itoks = _tokenize(_item_text_for_relevance(row))
    score = _jaccard(dtoks, itoks)
    ttoks = _tokenize(row.get("type", "")) | _tokenize(row.get("category_title", ""))
    if ttoks & dtoks:
        score += 0.03
    return min(score, 1.0)

# -------------------- Category mapping --------------------
def _heuristic_category(text: str) -> Optional[str]:
    """
    Fast mapping from multilingual text to one of FIXED_CATEGORIES.
    """
    t = _normalize_text(text)
    # Solid Waste
    if re.search(r"\b(dechet|dechets|waste|landfill|decharge|ordure|recycl|compost|inciner|dangerous waste|hazardous)\b", t):
        return "Solid Waste"
    # Water & Liquid discharges
    if re.search(r"\b(eau|eaux|water|liquid|effluent|wastewater|deversement|rejet|assainissement|sewer|drainage|discharge)\b", t):
        return "Water and Liquid discharges"
    # Noise & vibrations
    if re.search(r"\b(bruit|noise|acoustic|acoustique|vibration|vibrations)\b", t):
        return "Noise and vibrations"
    # Air
    if re.search(r"\b(air|atmospher|emission|gaz|polluant|pm10|pm2|nox|so2|co2|chimique atmos|qualite de l air)\b", t):
        return "Air"
    # Energy
    if re.search(r"\b(energie|énergie|energy|efficiency|rendement|fuel|electric|consommation|renewable|pv|solar)\b", t):
        return "Energy"
    # General
    if re.search(r"\b(environnemen|environment|sustainab|loi cadre|framework|general)\b", t):
        return "General Environmental & Sustainability Regulations"
    return None

def _build_llm_classify_prompt(project_desc: str, legal_req: str, type_hint: str) -> str:
    cats = "; ".join(FIXED_CATEGORIES)
    return f"""
Classify the following legal requirement into ONE category from this fixed list:
{cats}

Rules:
- Output ONLY JSON: {{ "category": "<one of the list exactly>" }}
- If uncertain, choose "General Environmental & Sustainability Regulations".

Project description (context):
\"\"\"{(project_desc or '').strip()[:1000]}\"\"\"

Type hint: {type_hint or ""}

Legal requirement (verbatim, may be French/Arabic/English):
\"\"\"{(legal_req or '').strip()[:1500]}\"\"\"
""".strip()

def _llm_classify_category(trainer, project_desc: str, legal_req: str, type_hint: str) -> Optional[str]:
    client = getattr(trainer, "client", None)
    generate_fn = getattr(client, "generate_json", None)
    if not callable(generate_fn):
        return None
    raw = generate_fn(_build_llm_classify_prompt(project_desc, legal_req, type_hint))
    try:
        data = json.loads(raw)
    except Exception:
        # try to extract a JSON block
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except Exception:
            return None
    cat = _to_str(data.get("category", "")).strip()
    return cat if cat in FIXED_CATEGORIES else None

# -------------------- LLM fill  --------------------

def _llm_is_relevant(trainer, project_desc: str, base_row: Dict[str, Any]) -> Optional[bool]:
    client = getattr(trainer, "client", None)
    generate_fn = getattr(client, "generate_json", None)
    if not callable(generate_fn):
        return None
    raw = generate_fn(_build_llm_relevance_prompt(project_desc, base_row))
    data = _try_parse_json_str(raw) or {}
    rel = str(data.get("relevant", "")).strip().lower()
    if rel in ("true", "false"):
        return rel == "true"
    if isinstance(data.get("relevant"), bool):
        return bool(data["relevant"])
    return None
def _normalize_phase_en(s: str) -> str:
    """Normalize LLM phase suggestions to English canonical values."""
    if not s:
        return ""
    s0 = str(s).strip().lower()
    # Aliases
    if any(k in s0 for k in ["design", "engineering", "feasibility", "basic design", "detailed design", "study"]):
        return "Design"
    if any(k in s0 for k in ["construct", "construction", "build", "erection", "site works", "civil works"]):
        return "Construction"
    if any(k in s0 for k in ["operate", "operation", "operational", "operations"]):
        return "Operation"
    if any(k in s0 for k in ["decommission", "decommissioning", "closure", "dismantling"]):
        return "Decommissioning"
    # Commissioning often bridges construction/operation; default to Construction
    if "commission" in s0:
        return "Construction"
    # Fallback: capitalize first letter only if short, else empty
    return s.strip().capitalize() if len(s0) <= 20 else ""


def _extract_json_block(s: str) -> Optional[str]:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n", "", s)
        s = re.sub(r"\n```$", "", s)
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None

def _try_parse_json_str(s: str) -> Optional[Dict[str, Any]]:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n", "", s)
        s = re.sub(r"\n```$", "", s)
    block = re.search(r"\{.*\}", s, re.S)
    if block:
        s = block.group(0)
    try:
        return json.loads(s)
    except Exception:
        s = s.replace("“", '"').replace("”", '"').replace("’", "'")
        s = re.sub(r",(\s*[}```])", r"\1", s)
        try:
            return json.loads(s)
        except Exception:
            return None


def _build_llm_relevance_prompt(project_desc: str, base_row: Dict[str, Any]) -> str:
    return f"""
Decide if the following legal text is RELEVANT to the described project.

Rules:
- Reply ONLY in JSON, nothing else.
- If relevance is uncertain, return "relevant": false.

Format:
{{ "relevant": true|false, "reason": "" }}

Project description:
\"\"\"{(project_desc or '').strip()[:2000]}\"\"\"

Legal text (summary):
- Category: {base_row.get('category_title','')}
- Type: {base_row.get('type','')}
- Legal requirement: {base_row.get('legal_requirement','')}
- Date: {base_row.get('date','')}
""".strip()

def _build_llm_row_prompt(project_desc: str, category_title: str, type_value: str, legal_requirement: str, date_value: str) -> str:
    cats = "; ".join(FIXED_CATEGORIES)
    return f"""
You are assisting a multinational company to build a legal register table. All outputs MUST be in English.

Task: Propose content for the following fields of a single table row using:
- the project description (below),
- the legal requirement text (below),
- and general EHS/Sustainability knowledge.

Hard rules:
- If a field is not obvious or not applicable, return an empty string "" for that field.
- Do NOT invent specific citations or references.
- Jurisdiction is handled by the application (defaults to "National").
- Phase must be one of: "Design", "Construction", "Operation", "Decommissioning" — or "" if unclear.
- Also provide a concise English paraphrase of the legal requirement, preserving law numbers and dates.

Allowed business categories (for reference): {cats}

Project description:
\"\"\"{(project_desc or '').strip()[:1800]}\"\"\"

Legal context:
- Category (business): {category_title or ''}
- Type hint: {type_value or ''}
- Legal requirement (verbatim): {legal_requirement or ''}
- Date: {date_value or ''}

Return ONLY this JSON (English text for every field):
{{
  "phase": "",
  "activity_aspect": "",
  "impacts": "",
  "description": "",
  "task": "",
  "responsibility": "",
  "comments": "",
  "legal_requirement_en": ""
}}
""".strip()

def _llm_fill_fields_for_row(trainer, project_desc: str, base_row: Dict[str, Any]) -> Dict[str, Any]:
    client = getattr(trainer, "client", None)
    generate_fn = getattr(client, "generate_json", None)
    if not callable(generate_fn):
        return {}
    prompt = _build_llm_row_prompt(
        project_desc,
        base_row.get("fixed_category") or base_row.get("category_title") or "",
        base_row.get("type_hint") or "",
        base_row.get("legal_requirement_raw") or "",
        base_row.get("date") or "",
    )
    raw = generate_fn(prompt)
    data = _try_parse_json_str(raw) or {}
    out = {
        "phase": _normalize_phase_en(_to_str(data.get("phase", ""))),
        "activity_aspect": _to_str(data.get("activity_aspect", "")),
        "impacts": _to_str(data.get("impacts", "")),
        "description": _to_str(data.get("description", "")),
        "task": _to_str(data.get("task", "")),
        "responsibility": _to_str(data.get("responsibility", "")),
        "comments": _to_str(data.get("comments", "")),
        "legal_requirement_en": _to_str(data.get("legal_requirement_en", "")),
    }
    return out

@csrf_exempt
@csrf_exempt
def analyze_legal_register_llm(request):
    """
    Select only legal texts/acts relevant to the project description, translate Legal Requirement to EN,
    classify each into fixed business categories, and build the strict structured_data array:
    [
      { "category_title": "<fixed business category>", "rows": [ [11 cells], ... ] },
      ...
    ]
    Columns mapping:
      - Phase / Activity/Aspect / Impacts / Description / Task / Responsibility / Comments: generated by LLM in English ("" allowed)
      - Jurisdiction: "National"
      - Type: fixed business category (one of FIXED_CATEGORIES)
      - Legal Requirement: concise English paraphrase produced by LLM, preserving law identifiers/dates
      - Date: from parsed JSON (Gregorian + Hijri if both)
    Relevance filtering: heuristic + (optional) LLM confirmation for borderline items.
    """
    rid = uuid.uuid4().hex[:8]
    t0 = time.perf_counter()
    logger.info("[LLM][%s] selective-EN analyze start (content_type=%s)",
                rid, request.META.get("CONTENT_TYPE", ""))

    def as_bool(v, default=False):
        if v is None:
            return default
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

    # -------- Read inputs --------
    content_type = request.META.get("CONTENT_TYPE", "")
    description = ""
    parsed_docs: List[Dict[str, Any]] = []

    # Controls (tune as needed)
    use_llm_fill = True                # LLM to fill EN fields + paraphrase legal req in EN
    relevance_mode = "hybrid"          # heuristic | llm | hybrid
    relevance_threshold = 0.12
    relevance_gray_margin = 0.06
    relevance_keep_top_k = 30
    max_llm_relevance = 10             # LLM relevance checks
    max_llm_classify = 20              # LLM category classification checks
    max_llm_rows = 25                  # LLM row fills
    api_key: Optional[str] = None

    if "application/json" in content_type:
        try:
            data = json.loads(request.body.decode("utf-8") or "{}")
        except Exception:
            logger.warning("[LLM][%s] Invalid JSON body", rid)
            return JsonResponse({"error": "Invalid JSON body"}, status=400)

        description = (data.get("description") or "").strip()
        if not description:
            return JsonResponse({"error": "description is required"}, status=400)

        use_llm_fill = as_bool(data.get("use_gemini"), True)
        relevance_mode = (data.get("relevance_mode") or relevance_mode).strip().lower()
        relevance_threshold = float(data.get("relevance_threshold", relevance_threshold))
        relevance_gray_margin = float(data.get("relevance_gray_margin", relevance_gray_margin))
        relevance_keep_top_k = int(data.get("relevance_keep_top_k", relevance_keep_top_k))
        max_llm_relevance = int(data.get("max_llm_relevance", max_llm_relevance))
        max_llm_classify = int(data.get("max_llm_classify", max_llm_classify))
        max_llm_rows = int(data.get("max_llm_rows", max_llm_rows))
        api_key = (data.get("api_key") or "").strip() or None

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
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            parsed_docs.append({"parsed": json.load(f), "source": str(path)})
                    except Exception as e:
                        logger.warning("[LLM][%s] Failed to load %s: %s", rid, path, e)
                else:
                    logger.warning("[LLM][%s] json_file not found: %s", rid, name)
        else:
            for p in sorted(JSON_OUTPUT_DIR.glob("*.json")):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        parsed_docs.append({"parsed": json.load(f), "source": str(p)})
                except Exception as e:
                    logger.warning("[LLM][%s] Failed to load %s: %s", rid, p, e)
    else:
        uploaded = request.FILES.get("file")
        description = (request.POST.get("description") or "").strip()
        if not description:
            return JsonResponse({"error": "description is required"}, status=400)

        use_llm_fill = as_bool(request.POST.get("use_gemini"), True)
        relevance_mode = (request.POST.get("relevance_mode") or relevance_mode).strip().lower()
        relevance_threshold = float(request.POST.get("relevance_threshold", relevance_threshold))
        relevance_gray_margin = float(request.POST.get("relevance_gray_margin", relevance_gray_margin))
        relevance_keep_top_k = int(request.POST.get("relevance_keep_top_k", relevance_keep_top_k))
        max_llm_relevance = int(request.POST.get("max_llm_relevance", max_llm_relevance))
        max_llm_classify = int(request.POST.get("max_llm_classify", max_llm_classify))
        max_llm_rows = int(request.POST.get("max_llm_rows", max_llm_rows))
        api_key = (request.POST.get("api_key") or "").strip() or None

        if uploaded:
            pdf_dir = _resolve_output_dir().resolve()
            pdf_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = (pdf_dir / Path(uploaded.name).name).resolve()
            with tempfile.NamedTemporaryFile(dir=str(pdf_dir), delete=False) as tmp:
                for chunk in uploaded.chunks():
                    tmp.write(chunk)
                tmp_path = Path(tmp.name).resolve()
            os.replace(tmp_path, pdf_path)
            logger.info("[LLM][%s] Uploaded PDF saved to %s", rid, pdf_path)

            try:
                parsed_json = extract_and_parse_pdf(str(pdf_path))
            except Exception as e:
                logger.exception("[LLM][%s] PDF parsing failed for %s", rid, pdf_path)
                return JsonResponse({"error": f"PDF parsing failed: {e}"}, status=500)

            parsed_docs.append({"parsed": parsed_json, "source": str(pdf_path)})
        else:
            for p in sorted(JSON_OUTPUT_DIR.glob("*.json")):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        parsed_docs.append({"parsed": json.load(f), "source": str(p)})
                except Exception as e:
                    logger.warning("[LLM][%s] Failed to load %s: %s", rid, p, e)

    if not parsed_docs:
        logger.info("[LLM][%s] No parsed documents to analyze", rid)
        return JsonResponse({"error": "No parsed documents found to analyze."}, status=400)

    # -------- Build base rows (no hallucination) --------
    def base_rows_from_parsed(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if not isinstance(doc, dict):
            return rows
        cats = doc.get("categories")
        if isinstance(cats, list):
            for cat in cats:
                cat_title = _to_str((cat or {}).get("title", "")) or ""
                for r in (cat or {}).get("rows", []) or []:
                    if not isinstance(r, dict):
                        continue
                    legal_req = _to_str(r.get("legal_requirement") or r.get("title") or "")
                    type_hint = _to_str(r.get("type") or cat_title)
                    date_val = _extract_dates_from_row_and_doc(r, doc)
                    rows.append({
                        "category_title": cat_title,            # will be replaced by fixed_category
                        "type_hint": type_hint,                # hint to help classification
                        "legal_requirement_raw": legal_req,    # original text
                        "description_raw": _to_str(r.get("description","")),
                        "date": date_val,
                        "jurisdiction": "National",
                        # placeholders for EN fields
                        "phase": "", "activity_aspect": "", "impacts": "",
                        "description": "", "task": "", "responsibility": "", "comments": "",
                    })
            return rows
        legal_texts = doc.get("legal_texts")
        if isinstance(legal_texts, list):
            for t in legal_texts:
                if not isinstance(t, dict):
                    continue
                type_hint = _to_str(t.get("type", ""))
                legal_req = _to_str(t.get("title", ""))
                date_val = _extract_dates_from_row_and_doc(
                    {"publication_date_gregorian": t.get("publication_date_gregorian",""),
                     "publication_date_hijri": t.get("publication_date_hijri","")},
                    doc
                )
                rows.append({
                    "category_title": type_hint,
                    "type_hint": type_hint,
                    "legal_requirement_raw": legal_req,
                    "description_raw": _to_str(t.get("description","")),
                    "date": date_val,
                    "jurisdiction": "National",
                    "phase": "", "activity_aspect": "", "impacts": "",
                    "description": "", "task": "", "responsibility": "", "comments": "",
                })
        return rows

    all_base_rows: List[Dict[str, Any]] = []
    for item in parsed_docs:
        all_base_rows.extend(base_rows_from_parsed(item["parsed"]))
    logger.info("[LLM][%s] Base rows total: %d", rid, len(all_base_rows))

    # -------- Map to fixed business categories --------
    trainer_for_classify = None
    if use_llm_fill or relevance_mode in ("llm","hybrid"):
        try:
            trainer_for_classify = _build_trainer(api_key)
            logger.info("[LLM][%s] Gemini trainer initialized", rid)
        except Exception as e:
            trainer_for_classify = None
            logger.warning("[LLM][%s] Gemini disabled for classify/fill: %s", rid, e)

    llm_classify_calls = 0
    for r in all_base_rows:
        text = " ".join([
            r.get("legal_requirement_raw",""),
            r.get("type_hint",""),
            r.get("category_title",""),
            description,
        ])
        cat = _heuristic_category(text)
        if cat is None and trainer_for_classify is not None and llm_classify_calls < max_llm_classify:
            cat = _llm_classify_category(trainer_for_classify, description, r.get("legal_requirement_raw",""), r.get("type_hint",""))
            llm_classify_calls += 1
        if cat is None or cat not in FIXED_CATEGORIES:
            cat = "General Environmental & Sustainability Regulations"
        r["fixed_category"] = cat
    logger.info("[LLM][%s] Category mapping done (llm_classify_calls=%d)", rid, llm_classify_calls)

    # -------- Relevance filter (keep only rows relevant to the project) --------
    scored = []
    for r in all_base_rows:
        s = _heuristic_relevance_score(description, {
            "category_title": r.get("fixed_category",""),
            "type": r.get("type_hint",""),
            "legal_requirement_raw": r.get("legal_requirement_raw",""),
            "description_raw": r.get("description_raw",""),
        })
        scored.append((s, r))
    scored.sort(key=lambda x: x[0], reverse=True)

    selected: List[Dict[str, Any]] = []
    llm_relevance_calls = 0
    if relevance_mode == "heuristic":
        for s, r in scored:
            if s >= relevance_threshold:
                selected.append(r)
        if not selected and relevance_keep_top_k > 0:
            selected = [r for _, r in scored[:relevance_keep_top_k]]
    elif relevance_mode == "llm":
        candidates = [r for _, r in scored[:max(5, max_llm_relevance)]]
        for r in candidates:
            if trainer_for_classify is None or llm_relevance_calls >= max_llm_relevance:
                break
            # reuse classify model client to answer relevance quickly
            raw = getattr(trainer_for_classify, "client", None)
            gen = getattr(raw, "generate_json", None)
            if callable(gen):
                prompt = f'''
Reply ONLY JSON {{"relevant": true|false}}
Project description:
"{description[:1200]}"
Legal requirement:
"{r.get("legal_requirement_raw","")[:1200]}"
'''
                ans = gen(prompt)
                try:
                    data = json.loads(ans)
                    if bool(data.get("relevant", False)):
                        selected.append(r)
                except Exception:
                    pass
                llm_relevance_calls += 1
        if not selected:
            for s, r in scored:
                if s >= relevance_threshold:
                    selected.append(r)
            if not selected and relevance_keep_top_k > 0:
                selected = [r for _, r in scored[:relevance_keep_top_k]]
    else:  # hybrid
        lower = max(0.0, relevance_threshold - relevance_gray_margin)
        for s, r in scored:
            if s >= relevance_threshold:
                selected.append(r)
            elif lower <= s < relevance_threshold and trainer_for_classify is not None and llm_relevance_calls < max_llm_relevance:
                raw = getattr(trainer_for_classify, "client", None)
                gen = getattr(raw, "generate_json", None)
                if callable(gen):
                    prompt = f'''
Reply ONLY JSON {{"relevant": true|false}}
Project description:
"{description[:1200]}"
Legal requirement:
"{r.get("legal_requirement_raw","")[:1200]}"
'''
                    ans = gen(prompt)
                    try:
                        data = json.loads(ans)
                        if bool(data.get("relevant", False)):
                            selected.append(r)
                    except Exception:
                        pass
                    llm_relevance_calls += 1
        if not selected and relevance_keep_top_k > 0:
            selected = [r for _, r in scored[:relevance_keep_top_k]]

    logger.info("[LLM][%s] Relevance: total=%d | selected=%d | mode=%s | thr=%.2f | LLM_checks=%d",
                rid, len(all_base_rows), len(selected), relevance_mode, relevance_threshold, llm_relevance_calls)

    # -------- LLM fill (English) only for selected rows --------
    enriched_rows: List[Dict[str, Any]] = []
    llm_fill_calls = 0
    for idx, base in enumerate(selected, start=1):
        filled = dict(base)
        # type column & group title must be the fixed business category
        fixed_cat = base.get("fixed_category") or "General Environmental & Sustainability Regulations"
        filled["category_title"] = fixed_cat
        filled["type"] = fixed_cat

        # Fill EN fields + EN legal requirement paraphrase
        if use_llm_fill and trainer_for_classify is not None and llm_fill_calls < max_llm_rows:
            try:
                logger.info("[LLM][%s] Fill row %d/%d via LLM", rid, idx, len(selected))
                sugg = _llm_fill_fields_for_row(trainer_for_classify, description, base)
                # merge
                for k in ("phase","activity_aspect","impacts","description","task","responsibility","comments"):
                    if k in sugg:
                        filled[k] = sugg[k]  # may be ""
                # EN legal requirement text (preferred)
                en_req = _to_str(sugg.get("legal_requirement_en","")).strip()
                if not en_req:
                    # fallback: keep original if LLM didn't produce; you may add a small translator here if needed
                    en_req = _to_str(base.get("legal_requirement_raw",""))
                filled["legal_requirement"] = en_req
                llm_fill_calls += 1
            except Exception as e:
                logger.warning("[LLM][%s] Fill row %d failed: %s", rid, idx, e)
                filled["legal_requirement"] = _to_str(base.get("legal_requirement_raw",""))
        else:
            filled["legal_requirement"] = _to_str(base.get("legal_requirement_raw",""))
        enriched_rows.append(filled)

    logger.info("[LLM][%s] Fill complete: selected=%d | llm_fills=%d",
                rid, len(selected), llm_fill_calls)

    # -------- Build strict table grouped & sorted by fixed categories --------
    table_structured = _build_table_structured_sorted(enriched_rows)
    total_rows_in_table = sum(len(c["rows"]) for c in table_structured)
    logger.info("[LLM][%s] Table built: categories=%d rows=%d", rid, len(table_structured), total_rows_in_table)

    # Save to session + file
    request.session[LEGAL_REGISTER_TABLE_DATA_KEY] = table_structured
    request.session.modified = True
    try:
        results_path = JSON_OUTPUT_DIR / "results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(table_structured, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning("[LLM][%s] Could not write results.json: %s", rid, e)

    dt = time.perf_counter() - t0
    logger.info("[LLM][%s] Done in %.2fs", rid, dt)
    return JsonResponse(table_structured, safe=False, json_dumps_params={"ensure_ascii": False, "indent": 2})

# ========= Download Legal Register PDF =========
def download_legal_register_pdf(request):
    rows = request.session.get(LEGAL_REGISTER_TABLE_DATA_KEY)
    if not rows:
        raise Http404("No legal register available. Run LLM analysis first.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            pdf_path = tmp_pdf.name

        generate_complete_report_pdf(rows, pdf_path)

        with open(pdf_path, "rb") as fh:
            pdf_bytes = fh.read()
        try:
            os.remove(pdf_path)
        except Exception:
            pass

        response = HttpResponse(pdf_bytes, content_type="application/pdf")
        response["Content-Disposition"] = 'attachment; filename="legal_register.pdf"'
        return response
    except Exception as e:
        return JsonResponse({"error": f"PDF generation failed: {e}"}, status=500)

