# views.py
import json
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import tempfile
import traceback
from typing import Any, Dict, List, Optional

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
from django.shortcuts import render, redirect
from django.contrib import messages

from app.services.legal_register.llm_traitment import make_trainer
from app.services.legal_register.pdf_generation import generate_complete_report_pdf
from app.services.legal_register.pdf_parser import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DPI,
    BOPdfParsingService,
    extract_text_pages_hybrid,  # added
)

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



LEGAL_REGISTER_ROWS_KEY = "legal_register_rows"
LEGAL_REGISTER_HEADERS_KEY = "legal_register_headers"

# Few-shot memory path (already used in your system)
FEWSHOT_MEMORY_PATH = Path("json/fewshot_memory.json")

_HAS_GEMINI = True  # Set to True if Gemini/LLM module available

# ========= Utils =========

def _parse_json_body(request) -> Dict[str, Any]:
    if not request.body:
        return {}
    try:
        return json.loads(request.body.decode("utf-8"))
    except Exception:
        return {}

def _to_bool(val: Any, default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return default
    s = str(val).strip().lower()
    return s in ("1", "true", "yes", "y", "on")

def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _build_service_from_params(params: Dict[str, Any]) -> BOPdfParsingService:
    dpi = int(params.get("dpi", DEFAULT_DPI))
    force_ocr = _to_bool(params.get("force_ocr", False), default=False)
    cache_dir = str(params.get("cache_dir", DEFAULT_CACHE_DIR))
    return BOPdfParsingService(cache_dir=cache_dir, dpi=dpi, force_ocr=force_ocr)

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

@csrf_exempt
def parse_upload(request):
    if request.method != "POST":
        return HttpResponseBadRequest("Only POST allowed. Send multipart/form-data.")

    uploaded = request.FILES.get("file")
    if not uploaded:
        return HttpResponseBadRequest("Missing file field 'file' (multipart/form-data).")

    overwrite = _to_bool(request.POST.get("overwrite"), default=False)
    write_json = _to_bool(request.POST.get("write_json"), default=True)
    svc = _build_service_from_params(request.POST)

    safe_name = Path(uploaded.name).name
    if not safe_name.lower().endswith(".pdf"):
        return HttpResponseBadRequest("Only .pdf files are accepted.")

    pdf_dir = _resolve_output_dir().resolve()
    pdf_dir.mkdir(parents=True, exist_ok=True)
    dest = pdf_dir / safe_name

    if dest.exists() and not overwrite:
        return JsonResponse({"error": "File already exists", "file": safe_name}, status=409)

    with NamedTemporaryFile(dir=str(pdf_dir), delete=False) as tmp:
        for chunk in uploaded.chunks():
            tmp.write(chunk)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, dest)

    try:
        data = svc.parse_pdf(str(dest))
        json_path = None
        if write_json:
            out_path = JSON_OUTPUT_DIR / f"{dest.stem}.json"
            _atomic_write_json(out_path, data)
            json_path = str(out_path)

        return JsonResponse({
            "ok": True,
            "file": safe_name,
            "saved_path": str(dest),
            "json_path": json_path,
            "data": data,
        })
    except Exception as exc:
        logger.exception("Parse failed for upload: %s", dest)
        return JsonResponse({"ok": False, "error": str(exc)}, status=500)


@csrf_exempt
def parse_file(request, filename: Optional[str] = None):
    if request.method not in ("POST", "GET"):
        return HttpResponseBadRequest("Only GET or POST allowed.")

    body = _parse_json_body(request)
    write_json = _to_bool(body.get("write_json"), default=True)
    if not filename:
        filename = body.get("filename")
    if not filename:
        return HttpResponseBadRequest("Missing filename.")

    safe_name = Path(filename).name
    if not safe_name.lower().endswith(".pdf"):
        return HttpResponseBadRequest("Only .pdf files are accepted.")

    pdf_dir = _resolve_output_dir().resolve()
    pdf_path = (pdf_dir / safe_name).resolve()
    if not (_is_within(pdf_dir, pdf_path) and pdf_path.exists()):
        return HttpResponseBadRequest("PDF not found or invalid filename.")

    svc = _build_service_from_params(body)

    try:
        data = svc.parse_pdf(str(pdf_path))
        json_path = None
        if write_json:
            out_path = (JSON_OUTPUT_DIR / f"{pdf_path.stem}.json").resolve()
            if not _is_within(JSON_OUTPUT_DIR, out_path):
                return HttpResponseBadRequest("Invalid JSON output path resolution.")
            _atomic_write_json(out_path, data)
            json_path = str(out_path)

        return JsonResponse({
            "ok": True,
            "file": safe_name,
            "pdf_path": str(pdf_path),
            "json_path": json_path,
            "data": data,
        })
    except Exception as exc:
        logger.exception("Parse failed for file: %s", pdf_path)
        return JsonResponse({"ok": False, "error": str(exc)}, status=500)


@csrf_exempt
def parse_dir(request):
    if request.method != "POST":
        return HttpResponseBadRequest("Only POST allowed. Send JSON body.")

    body = _parse_json_body(request)
    pattern = body.get("pattern", "*.pdf")
    recursive = _to_bool(body.get("recursive"), default=False)
    limit = int(body.get("limit", 0) or 0)
    write_json = _to_bool(body.get("write_json"), default=True)
    max_workers = int(body.get("max_workers", 0) or 0)
    filenames = body.get("filenames")

    pdf_dir = _resolve_output_dir().resolve()
    pdf_dir.mkdir(parents=True, exist_ok=True)

    files: List[Path] = []
    if isinstance(filenames, list) and filenames:
        for name in filenames:
            p = (pdf_dir / Path(name).name).resolve()
            if _is_within(pdf_dir, p) and p.exists() and p.suffix.lower() == ".pdf":
                files.append(p)
    else:
        if recursive:
            files = [p for p in pdf_dir.rglob(pattern) if p.is_file() and p.suffix.lower() == ".pdf"]
        else:
            files = [p for p in pdf_dir.glob(pattern) if p.is_file() and p.suffix.lower() == ".pdf"]

    files = sorted(files)
    if limit and limit > 0:
        files = files[:limit]

    if not files:
        return JsonResponse({"count": 0, "results": []})

    svc = _build_service_from_params(body)
    out_dir = str(JSON_OUTPUT_DIR) if write_json else None

    if max_workers and max_workers > 0:
        results = svc.parse_many([str(p) for p in files], output_dir=out_dir, max_workers=max_workers)
    else:
        results = svc.parse_many([str(p) for p in files], output_dir=out_dir)

    return JsonResponse({"count": len(files), "results": results})


def deps_report(request):
    svc = BOPdfParsingService()
    return JsonResponse(svc.dependency_report())


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
@csrf_exempt
def analyze_legal_register_llm(request):
    """
    Receives a PDF (uploaded or previously parsed) and:
    - Optionally refines it via Gemini LLM
    - Stores structured rows in session
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    uploaded = request.FILES.get("file")
    use_gemini = _to_bool(request.POST.get("use_gemini"), default=True)
    api_key = request.POST.get("api_key", "").strip() or None
    refine_strategy = request.POST.get("refine_strategy", "fill_missing")
    add_to_fewshot = _to_bool(request.POST.get("add_to_fewshot"), default=False)

    if not uploaded:
        return JsonResponse({"error": "No PDF uploaded"}, status=400)

    output_dir = Path("pdfs")
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / Path(uploaded.name).name
    with tempfile.NamedTemporaryFile(dir=str(output_dir), delete=False) as tmp:
        for chunk in uploaded.chunks():
            tmp.write(chunk)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, pdf_path)

    svc = BOPdfParsingService(dpi=DEFAULT_DPI)
    try:
        parsed_json = svc.parse_pdf(str(pdf_path))
    except Exception as e:
        return JsonResponse({"error": f"PDF parsing failed: {e}"}, status=500)

    refined_json = parsed_json
    used_gemini = False
    raw_text = ""

    if use_gemini:
        try:
            trainer = _build_trainer(api_key)
            raw_text, _ = extract_text_pages_hybrid(str(pdf_path), dpi=svc.dpi, cache_dir=svc.cache_dir, force_ocr=svc.force_ocr)
            refined_issue = trainer.refine_parsed_json(parsed_json, raw_text=raw_text, strategy=refine_strategy)
            refined_json = refined_issue.dict(ensure_ascii=False)
            used_gemini = True
        except Exception as e:
            refined_json = parsed_json

    # Optionally add to few-shot
    if add_to_fewshot:
        try:
            if not raw_text:
                raw_text, _ = extract_text_pages_hybrid(str(pdf_path))
            trainer = _build_trainer(api_key)
            trainer.add_example(raw_text, refined_json)
            _persist_memory(trainer)
        except Exception:
            pass

    # Transform to rows for PDF generation
    rows = []
    for cat in refined_json.get("categories", []):
        for row in cat.get("rows", []):
            flat_row = {"category_title": cat.get("title", "")}
            flat_row.update(row)
            rows.append(flat_row)

    # Save rows in session for download
    request.session[LEGAL_REGISTER_ROWS_KEY] = rows
    request.session[LEGAL_REGISTER_HEADERS_KEY] = list(rows[0].keys()) if rows else []
    request.session.modified = True

    return JsonResponse({
        "ok": True,
        "used_gemini": used_gemini,
        "rows_count": len(rows),
        "json": refined_json,
    })


# ========= Download Legal Register PDF =========
def download_legal_register_pdf(request):
    rows = request.session.get(LEGAL_REGISTER_ROWS_KEY)
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
