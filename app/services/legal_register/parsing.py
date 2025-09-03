from __future__ import annotations
import os
import re
import json
import hashlib
import argparse
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal, InvalidOperation

# Third-party (hard dependency)
try:
    import fitz  # PyMuPDF
except Exception as e:
    raise RuntimeError("PyMuPDF (fitz) is required. Install with `pip install pymupdf`.") from e

# OCR (optional)
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False
    Image = None  # type: ignore

# Regex with Unicode classes (optional)
try:
    import regex as regex_mod
except Exception:
    regex_mod = None

# Language detection (optional)
try:
    import langid
except Exception:
    langid = None

# Dates (optional)
try:
    import dateparser
except Exception:
    dateparser = None

try:
    from dateutil import parser as dateutil_parser
except Exception:
    dateutil_parser = None

# ========================= Config =========================
DEFAULT_DPI = 200
DEFAULT_CACHE_DIR = "./ocr_cache"
MIN_TEXT_LENGTH_FOR_PAGEMODE = 40
MAX_WORKERS = max(1, (os.cpu_count() or 2) - 1)

MONTHS_FR = {
    "janvier": "01",
    "février": "02", "fevrier": "02",
    "mars": "03",
    "avril": "04",
    "mai": "05",
    "juin": "06",
    "juillet": "07",
    "août": "08", "aout": "08",
    "septembre": "09",
    "octobre": "10",
    "novembre": "11",
    "décembre": "12", "decembre": "12",
}

FR_MONTHS_IDX = {
    1: "janvier", 2: "février", 3: "mars", 4: "avril", 5: "mai", 6: "juin",
    7: "juillet", 8: "août", 9: "septembre", 10: "octobre", 11: "novembre", 12: "décembre"
}

def format_date_fr(iso_date: str) -> str:
    """ '2025-06-23' -> '23 juin 2025' """
    if not iso_date or not re.match(r'^\d{4}-\d{2}-\d{2}$', iso_date):
        return (iso_date or '').strip()
    y, m, d = iso_date.split('-')
    try:
        return f"{int(d)} {FR_MONTHS_IDX[int(m)]} {int(y)}"
    except Exception:
        return (iso_date or '').strip()

CURRENCY_MAP = {
    "€": "EUR",
    "eur": "EUR",
    "euro": "EUR",
    "euros": "EUR",
    "dh": "MAD",
    "dhs": "MAD",
    "mad": "MAD",
}

# ========================= Utils =========================

def _strip_controls(text: str) -> str:
    return ''.join(ch for ch in text or '' if unicodedata.category(ch)[0] != 'C' or ch in '\n\r\t')

def has_arabic(text: str) -> bool:
    if not text:
        return False
    if regex_mod:
        try:
            return bool(regex_mod.search(r'[\p{Arabic}]', text))
        except Exception:
            pass
    return bool(re.search(r'[\u0600-\u06FF]', text))

def detect_lang(text: str) -> str:
    if not text:
        return 'fr'
    if has_arabic(text):
        return 'ar'
    if langid:
        lid, conf = langid.classify(text[:4000])
        if lid in {'fr','ar'} and conf >= 0.65:
            return lid
    return 'fr'

def normalize_ocr_common(text: str) -> str:
    if not text:
        return text
    t = (text.replace('\u00A0', ' ')
            .replace('’', "'").replace('“', '"').replace('”', '"')
            .replace('–', '-').replace('—', '-'))
    # fix typical OCR digit confusions only when surrounded by digits
    if regex_mod:
        def fix_digits(m):
            s = m.group(0)
            return (s.replace('O','0').replace('o','0').replace('l','1').replace('I','1').replace('S','5'))
        try:
            t = regex_mod.sub(r'(?<=\d)[OolIS]{1,}(?=\d)', fix_digits, t)
            t = regex_mod.sub(r'(?:(?<=\b)|(?<=-))[OolIS0-9]{2,}(?=\b|[-/])', fix_digits, t)
        except Exception:
            pass
    t = re.sub(r'[ \t]{2,}', ' ', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    t = re.sub(r'(\w+)-\n(\w+)', r'\1\2', t)  # join hyphenated breaks
    return t

def normalize_plain(text: str) -> str:
    if not text:
        return text
    t = (text.replace('\u00A0', ' ')
            .replace('’', "'").replace('“', '"').replace('”', '"')
            .replace('–', '-').replace('—', '-'))
    t = re.sub(r'[ \t]{2,}', ' ', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    t = re.sub(r'(\w+)-\n(\w+)', r'\1\2', t)
    return t

def parse_french_date_to_iso(date_str: str) -> Optional[str]:
    if not date_str:
        return None
    s = date_str.strip().replace('1er', '1')
    s_low = unicodedata.normalize('NFKD', s).lower().replace(',', ' ')
    if dateparser:
        dt = dateparser.parse(s_low, languages=['fr'])
        if dt:
            return dt.strftime('%Y-%m-%d')
    m = re.search(r'(\d{1,2})\s+([a-zêéèîïôûùâàäöü]+)\s+(\d{4})', s_low, re.IGNORECASE)
    if m:
        d, mon, y = m.groups()
        mon_num = MONTHS_FR.get(mon)
        if mon_num:
            return f"{int(y):04d}-{mon_num}-{int(d):02d}"
    if dateutil_parser:
        try:
            dt = dateutil_parser.parse(s, dayfirst=True)
            return dt.strftime('%Y-%m-%d')
        except Exception:
            pass
    return None

def fr_amount_string(amount: float) -> str:
    """Format 578300000.0 -> '578.300.000,00'"""
    d = Decimal(str(amount)).quantize(Decimal('0.01'))
    s = f"{d:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    return s

# ========================= Extraction (text/OCR) =========================

@dataclass
class PageResult:
    page_number: int
    text: str
    used_ocr: bool
    lang: str

def _render_page_to_pil(page: fitz.Page, dpi: int = DEFAULT_DPI) :
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    mode = 'RGB' if pix.n < 4 else 'RGBA'
    return Image.frombytes(mode, [pix.width, pix.height], pix.samples)

def ocr_page_image(img: Image.Image, ocr_lang: str = 'fra+ara', ocr_config: str = '--psm 6') -> str: # type: ignore
    if not OCR_AVAILABLE:
        return ''
    return pytesseract.image_to_string(img, lang=ocr_lang, config=ocr_config)

def extract_text_pages_hybrid(pdf_path: str, dpi: int = DEFAULT_DPI,
                              cache_dir: str = DEFAULT_CACHE_DIR,
                              force_ocr: bool = False) -> Tuple[str, List[PageResult]]:
    """Text-first with OCR fallback per page; cached by file mtime + dpi"""
    os.makedirs(cache_dir, exist_ok=True)
    try:
        stat = os.stat(pdf_path)
        key_material = f"{os.path.abspath(pdf_path)}::{stat.st_mtime_ns}::dpi={dpi}"
    except Exception:
        key_material = f"{os.path.abspath(pdf_path)}::nodest"
    cache_key = hashlib.sha256(key_material.encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")

    if os.path.exists(cache_file) and not force_ocr:
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            pages = [PageResult(**p) for p in data.get('pages', [])]
            return "\n\n".join(p.text for p in pages), pages
        except Exception:
            pass

    doc = fitz.open(pdf_path)
    n_pages = doc.page_count
    page_results: List[Optional[PageResult]] = [None] * n_pages
    pages_to_ocr: List[int] = []

    # Pass 1: embedded text
    for i in range(n_pages):
        page = doc.load_page(i)
        try:
            txt = page.get_text('text')
        except Exception:
            txt = ''
        txt = normalize_plain((txt or '').strip())
        if txt and len(txt) >= MIN_TEXT_LENGTH_FOR_PAGEMODE and not (force_ocr and OCR_AVAILABLE):
            page_results[i] = PageResult(page_number=i+1, text=txt, used_ocr=False, lang=detect_lang(txt))
        else:
            pages_to_ocr.append(i)

    # Pass 2: OCR (only if module available)
    if pages_to_ocr and OCR_AVAILABLE:
        def ocr_worker(idx: int) -> PageResult:
            p = doc.load_page(idx)
            img = _render_page_to_pil(p, dpi=dpi)
            sample = pytesseract.image_to_string(img.crop((0, 0, min(1000, img.width), min(220, img.height))),
                                                 lang='fra+ara', config='--psm 6')[:2000]
            lang_hint = 'ara' if detect_lang(sample) == 'ar' else 'fra'
            try:
                otext = ocr_page_image(img, ocr_lang=f"{lang_hint}+fra+ara", ocr_config='--psm 6')
                otext = normalize_ocr_common(otext)
            except Exception:
                otext = ''
            return PageResult(page_number=idx+1, text=otext, used_ocr=True, lang=detect_lang(otext))

        workers = min(MAX_WORKERS, max(1, len(pages_to_ocr)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(ocr_worker, i): i for i in pages_to_ocr}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    page_results[idx] = fut.result()
                except Exception:
                    page_results[idx] = PageResult(page_number=idx+1, text='', used_ocr=True, lang='fr')
    else:
        # If OCR not available, mark those pages as empty
        for idx in pages_to_ocr:
            page_results[idx] = PageResult(page_number=idx+1, text='', used_ocr=False, lang='fr')

    pages_final: List[PageResult] = [p if p else PageResult(page_number=i+1, text='', used_ocr=False, lang='fr')
                                     for i, p in enumerate(page_results)]

    joined = "\n\n".join(p.text for p in pages_final)
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({"pages": [p.__dict__ for p in pages_final]}, f, ensure_ascii=False)
    except Exception:
        pass
    return joined, pages_final

# ========================= Parsing =========================

def extract_page_markers(text: str) -> Dict[int, int]:
    markers = {}
    pattern = r'\n\s*([0-9]{2,5})\s+(?:BULLETIN(?:\s+OFFICIEL)?|النشرة)\b'
    for m in re.finditer(pattern, text or '', re.IGNORECASE):
        try:
            p = int(m.group(1))
        except Exception:
            continue
        markers[p] = m.start()
    return markers

def find_issn(text: str) -> Optional[str]:
    m = re.search(r'ISSN\s*[:\s]*([0-9]{4}\s*-\s*[0-9]{4})', text or '', re.IGNORECASE)
    return m.group(1).replace(' ', '') if m else None

def find_issue_and_dates(text: str) -> Tuple[str, str, str]:
    issue = ''
    hijri = ''
    greg = ''
    m_issue = re.search(r'N[°oº]\s*([0-9]{2,6})', text or '', re.IGNORECASE)
    if m_issue:
        issue = m_issue.group(1)
    # hijri (FR words) + (greg)
    if regex_mod:
        m = regex_mod.search(r'([0-9]{1,2}\s+[^\(\)\d\n]{3,30}\s+[0-9]{3,4})\s*\(\s*([^\)]+)\s*\)', text or '', regex_mod.IGNORECASE)
    else:
        m = re.search(r'([0-9]{1,2}\s+[A-Za-zéèêëîïôûùâàäöü\-]+?\s+[0-9]{3,4})\s*\(\s*([^\)]+)\s*\)', text or '', re.IGNORECASE)
    if m:
        hijri = m.group(1).strip()
        greg = parse_french_date_to_iso(m.group(2).strip()) or ''
    else:
        mg = re.search(r'\(\s*([0-9]{1,2}\s+[A-Za-zéèêëîïôûùâàäöü]+\s+[0-9]{4})\s*\)', text or '')
        if mg:
            greg = parse_french_date_to_iso(mg.group(1)) or ''
    return issue, hijri, greg

def parse_bulletin_metadata(full_text: str) -> Dict[str, str]:
    issue, hijri, greg = find_issue_and_dates(full_text)
    issn = find_issn(full_text) or ''
    m_year = re.search(r'([A-Z][a-zéèêàâîïôûù\- ]+année)\s+[–-]\s+N', full_text or '')
    pub_year_desc = m_year.group(1).strip() if m_year else ''
    return {
        "issue_number": issue or '',
        "publication_year_description": pub_year_desc or '',
        "date_hijri": hijri or '',
        "date_gregorian": greg or '',
        "issn": issn or ''
    }

# -------- TOC: robust, stateful line stitching ----------
def parse_table_of_contents(full_text: str) -> List[Dict[str, Any]]:
    lines = full_text.splitlines()
    # Find SOMMAIRE / TABLE DES MATIERES
    start_idx = None
    for i, ln in enumerate(lines[:1500]):
        if re.search(r'^\s*(SOMMAIRE|TABLE\s+DES\s+MATI[EÈ]RES)\b', ln, flags=re.IGNORECASE):
            start_idx = i
            break
    if start_idx is None:
        # Fallback: single-line dotted leaders
        toc = []
        dotted = re.compile(r'^(.*?)\s*\.{2,}\s*(\d{2,5})$')
        for raw in lines[:1500]:
            s = raw.strip()
            m = dotted.match(s)
            if m and len(m.group(1).strip(' .-')) > 10:
                toc.append({"category": "", "title": m.group(1).strip(' .-'), "description": "", "page": int(m.group(2))})
        return toc

    toc_entries: List[Dict[str, Any]] = []
    buf: List[str] = []
    current_category = ""
    dotted = re.compile(r'(.*?)\s*\.{2,}\s*(\d{2,5})\s*$')

    for ln in lines[start_idx:start_idx+2500]:
        s = ln.strip()
        if not s:
            continue
        # categories
        if re.search(r'^(TEXTES?\s+G[EÉ]N[ÉE]RAUX|TEXTES?\s+PARTICULIERS)', s, flags=re.IGNORECASE):
            current_category = re.sub(r'\s+', ' ', s.upper())
            continue
        # skip page headers/footers that creep in
        if re.search(r'^\d{1,4}\s+(BULLETIN(?:\s+OFFICIEL)?|النشرة)\b', s, flags=re.IGNORECASE):
            continue
        m = dotted.search(s)
        if m:
            piece = m.group(1).strip(' .\t-–—')
            if piece:
                buf.append(piece)
            title = ' '.join([re.sub(r'\s+', ' ', b) for b in buf]).strip()
            if len(title) >= 10:
                try:
                    page = int(m.group(2))
                    toc_entries.append({
                        "category": current_category,
                        "title": title,
                        "description": "",
                        "page": page,
                    })
                except Exception:
                    pass
            buf = []
        else:
            buf.append(s)

    # dedupe on (title, page)
    uniq = {}
    for e in toc_entries:
        key = (e['title'], e['page'])
        if key not in uniq:
            uniq[key] = e
    return list(uniq.values())

# -------- Money: precise, robust (FR separators) --------
def extract_money_with_precision(chunk: str) -> Optional[Dict[str, Any]]:
    """Return {'amount': float, 'currency': 'EUR'|'MAD', 'spelled'?: str} or None"""
    if not chunk:
        return None

    rx_spelled = re.compile(
        r"montant\s+de\s+([a-zàâçéèêëîïôûùüÿ\-’'\s]{8,160})\s*\(\s*([0-9][0-9\.\s\u202F,]+)\s*(€|euros?|eur|dh?s?|mad)\s*\)",
        re.IGNORECASE,
    )
    rx_plain = re.compile(
        r"\(\s*([0-9][0-9\.\s\u202F,]+)\s*(€|euros?|eur|dh?s?|mad)\s*\)"
        r"|([0-9][0-9\.\s\u202F,]+)\s*(€|euros?|eur|dh?s?|mad)\b",
        re.IGNORECASE,
    )

    spelled = None
    m = rx_spelled.search(chunk)
    if m:
        spelled = ' '.join(m.group(1).strip().split())
        num_raw = m.group(2)
        cur_raw = m.group(3)
    else:
        m = rx_plain.search(chunk)
        if not m:
            return None
        if m.group(1) and m.group(2):
            num_raw, cur_raw = m.group(1), m.group(2)
        else:
            num_raw, cur_raw = m.group(3), m.group(4)

    s = (num_raw or '').replace('\u202F', ' ').replace('\xa0', ' ').replace(' ', '')
    if ',' in s:
        s = s.replace('.', '').replace(',', '.')
    else:
        s = s.replace('.', '')

    try:
        # keep float for JSON compatibility; avoids Decimal->str fallback
        amount_val = float(s)
    except Exception:
        try:
            amount_val = float(Decimal(s))
        except Exception:
            return None

    cur = (cur_raw or '').lower()
    currency = 'EUR' if cur in ('€', 'eur', 'euro', 'euros') else ('MAD' if cur in ('dh', 'dhs', 'mad') else cur.upper())

    out = {"amount": amount_val, "currency": currency}
    if spelled:
        out["spelled"] = spelled
    return out

# -------- Signatories: contextual & filtered --------
BLOCKLIST = {
    "BULLETIN","ANNEXE","ARTICLE","SOMMAIRE","PAGE","TEXTES","SEEDS","INTERNATIONAL",
    "DÉCRÈTE","DECRETE","ARRÊTE","ARRETE","DÉCIDE","DECIDE","CHAPITRE","SECTION",
    "TABLE","IMPRIMERIE","OFFICIELLE","RABAT","KING","KAMAL"  # extend as needed
}

def _looks_like_name(name: str) -> bool:
    if not name or ' ' not in name:
        return False
    up = name.upper()
    # must have vowels
    if not re.search(r'[AEIOUY]', up):
        return False
    # 2–6 tokens
    parts = [p for p in up.split() if p]
    if not (2 <= len(parts) <= 6):
        return False
    # blocklist as whole words
    for tok in BLOCKLIST:
        if re.search(rf'\b{re.escape(tok)}\b', up):
            return False
    # avoid numbers / annex refs
    if re.search(r'\d', up):
        return False
    return True

def extract_signatories(block_text: str) -> List[Dict[str, str]]:
    """Strict: last 20% of text only; contextual patterns first; then conservative fallback."""
    if not block_text:
        return []
    L = len(block_text)
    tail = block_text[int(L*0.8):]

    results: List[Dict[str, str]] = []

    # Pattern A: "Pour contreseing : <Title>\n<NAME>."
    for m in re.finditer(r'Pour\s+contreseing\s*[:\-]?\s*([^\n,]+)\s*\n\s*([A-ZÉÈÊÂÎÏÔÛÙÄËÖÜ\'\-\s]{3,80})\.?', tail, flags=re.IGNORECASE):
        title = m.group(1).strip()
        name = ' '.join(m.group(2).split())
        if _looks_like_name(name):
            results.append({"name": name, "title": title})

    # Pattern B: "Fait à ..., le ...\n<Title>,\n<NAME>."
    for m in re.finditer(r'Fait\s+à\s+[^\n,]+,\s*le\s+[^\n]+\n\s*([A-Za-zéèêàâîïôûù\'\-\s]{3,80})\s*,\s*\n\s*([A-ZÉÈÊÂÎÏÔÛÙÄËÖÜ\'\-\s]{3,80})\.?', tail, flags=re.IGNORECASE):
        title = m.group(1).strip()
        name = ' '.join(m.group(2).split())
        if _looks_like_name(name):
            results.append({"name": name, "title": title})

    # Conservative fallback: stand-alone uppercase line (rarely used now)
    for m in re.finditer(r'\n\s*([A-ZÉÈÊÂÎÏÔÛÙÄËÖÜ\'\-\s]{3,80})\s*\.?\s*\n', tail):
        name = ' '.join(m.group(1).split())
        if _looks_like_name(name):
            results.append({"name": name, "title": ""})

    # De-duplicate
    seen = set()
    out: List[Dict[str, str]] = []
    for s in results:
        key = (s.get('name','').strip(), s.get('title','').strip())
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out

# -------- Legal block detection & field extraction --------
LEGAL_START_PATTERNS = [
    r'\bDécret\s+n[°oº]\s*[0-9][0-9\-\s]{0,20}[0-9]',
    r'\bDécret\s+du\s+\d{1,2}\s+[A-Za-zéèêëîïôûùâàäöü]+\s+\d{4}',
    r'\bArrêté\s+conjoint\s+n[°oº]?\s*[0-9][0-9\-\s]{0,20}[0-9]',
    r'\bArr[ée]t[ée]\s+n[°oº]?\s*[0-9][0-9\-\s]{0,20}[0-9]',
    r'\bArr[ée]t[ée]\s+du\s+\d{1,2}\s+[A-Za-zéèêëîïôûùâàäöü]+\s+\d{4}',
    r'\bLoi\s+n[°oº]?\s*[0-9][0-9\-\s]{0,20}[0-9]',
    r'\bDécision\s+n[°oº]?\s*[0-9][0-9\-\s]{0,20}[0-9]',
    r'\bDécision\s+du\s+\d{1,2}\s+[A-Za-zéèêëîïôûùâàäöü]+\s+\d{4}',
    r'\bOrdonnance\s+n[°oº]?\s*[0-9]'
]
LEGAL_START_RE = re.compile('|'.join(LEGAL_START_PATTERNS), flags=re.IGNORECASE)

@dataclass
class Block:
    start: int
    end: int
    header: str
    text: str

def split_legal_texts(full_text: str) -> List[Block]:
    starts = [m.start() for m in LEGAL_START_RE.finditer(full_text)]
    starts = sorted(set(starts))
    if not starts:
        return []
    blocks: List[Block] = []
    for i, s in enumerate(starts):
        e = starts[i+1] if i+1 < len(starts) else len(full_text)
        chunk = full_text[s:e].strip('\n')
        header = chunk.splitlines()[0] if chunk else ''
        blocks.append(Block(start=s, end=e, header=header, text=chunk))
    return blocks

def _detect_type(header: str) -> str:
    low = header.lower()
    if 'décret' in low: return 'Décret'
    if 'arrêté' in low or 'arrete' in low: return 'Arrêté'
    if low.startswith('loi') or ' loi ' in low: return 'Loi'
    if 'décision' in low or 'decision' in low: return 'Décision'
    return 'Texte'

def _extract_number(header: str) -> str:
    m = re.search(r'n[°oº]\s*([0-9][0-9\-\s]{0,20}[0-9])', header, re.IGNORECASE)
    if not m:
        return ''
    return re.sub(r'\s+', '', m.group(1)).replace('O','0').replace('o','0').replace('l','1')

TITLE_STOP_WORDS = re.compile(r'\b(D[ÉE]CR[ÈE]TE\s*:|ARR[ÊE]TE\s*:|D[ÉE]CIDE\s*:)\b', flags=re.IGNORECASE)

def _extract_dates_from_block(block_text: str) -> Tuple[str, str]:
    hijri = ''
    greg = ''
    head = '\n'.join(block_text.splitlines()[:6])  # look in first lines
    candidates = [head, block_text]
    for zone in candidates:
        m = re.search(r'du\s+([0-9]{1,2}\s+[^\(\)\d\n]{3,30}\s+[0-9]{3,4})\s*\(\s*([^\)]+)\s*\)', zone, re.IGNORECASE)
        if m:
            hijri = hijri or m.group(1).strip()
            greg = greg or (parse_french_date_to_iso(m.group(2).strip()) or '')
        else:
            mg = re.search(r'\(\s*([0-9]{1,2}\s+[A-Za-zéèêëîïôûùâàäöü]+\s+[0-9]{4})\s*\)', zone)
            if mg and not greg:
                greg = parse_french_date_to_iso(mg.group(1).strip()) or ''
    return hijri, greg

def _extract_title_from_block(header: str, block_text: str) -> str:
    line = (header or '').strip()

    after = ''
    m = re.search(r'\)\s*(.+)$', line)
    if m and len(m.group(1).strip()) > 3:
        after = m.group(1).strip()
    if not after:
        m2 = re.search(r'n[°oº]\s*[0-9][0-9\-\s]{0,20}[0-9]\s*(.+)$', line, re.IGNORECASE)
        if m2 and len(m2.group(1).strip()) > 3:
            after = m2.group(1).strip()

    body = []
    for ln in (block_text or '').splitlines()[1:12]:
        ln = ln.strip()
        if not ln:
            continue
        if TITLE_STOP_WORDS.search(ln) or re.match(r'^(Vu|Vus|Attendu|Consid[ée]rant|Chapitre|Article)\b', ln, re.IGNORECASE):
            break
        body.append(ln)
        if ln.endswith('.'):
            break

    pieces = []
    if after:
        pieces.append(after.rstrip('.'))
    if (not pieces) or len(pieces[0].split()) < 4 or re.match(r"^(approuvant|relatif|portant|modifiant|fixant|autorisant)\b", pieces[0], re.IGNORECASE):
        if body:
            sentence = ' '.join(body)
            sentence = TITLE_STOP_WORDS.split(sentence)[0]
            sentence = sentence.split(' .')[0].split('. ')[0].rstrip('.')
            if sentence and sentence.lower() not in pieces:
                pieces.append(sentence)

    title = ' '.join(pieces).strip()
    if len(title) < 10:
        title = (after or ' '.join(body)).strip().rstrip('.')
        title = TITLE_STOP_WORDS.split(title)[0].strip()
    return title

def build_description_fr(typ: str, num: str, hijri: str, greg_iso: str, title: str, block_text: str) -> str:
    parts = []
    if typ:
        parts.append(typ)
    if num:
        parts.append(f"n° {num}")
    greg = format_date_fr(greg_iso) if greg_iso else ''
    if hijri and greg:
        parts.append(f"du {hijri} ({greg})")
    elif greg:
        parts.append(f"du {greg}")
    elif hijri:
        parts.append(f"du {hijri}")

    desc = " ".join(parts).strip()
    if title:
        desc = (desc + " " + title.rstrip('.')).strip()

    money = extract_money_with_precision(block_text)
    if money:
        amt_str = fr_amount_string(money["amount"])
        unit = 'euros' if money["currency"] == 'EUR' else 'dirhams'
        if money.get("spelled"):
            desc += f", d'un montant de {money['spelled']} ({amt_str} {unit})"
        else:
            desc += f", d'un montant de {amt_str} {unit}"

    m_conclu = re.search(r"\bconclu[e]?\s+le\s+([0-9]{1,2}\s+[a-zàâçéèêëîïôûùüÿ]+(?:\s+\d{4})?)", block_text, re.IGNORECASE)
    if m_conclu:
        concl_date = ' '.join(m_conclu.group(1).split())
        if concl_date not in desc:
            desc += f", conclu le {concl_date}"

    # lender/beneficiary/project mentions (concise & factual)
    m_lender = re.search(r'\b(Banque\s+internationale\s+pour\s+la\s+reconstruction\s+et\s+le\s+d[ée]veloppement|KfW|KFW|BEI|AFD)\b', block_text, re.IGNORECASE)
    if m_lender and m_lender.group(1) not in desc:
        desc += f" avec {m_lender.group(1)}"

    m_benef = re.search(r'\b(MASEN|Moroccan Agency for Sustainable Energy|Office\s+[A-Z][A-Za-z\-\s]+)\b', block_text, re.IGNORECASE)
    if m_benef and m_benef.group(1) not in desc:
        desc += f", au profit de {m_benef.group(1)}"

    m_proj = re.search(r'Projet\s+[«\"]?\s*([^»\"\n]+)\s*[»\"]?', block_text)
    if m_proj:
        proj = m_proj.group(1).strip()
        if proj and proj not in desc:
            desc += f", pour le financement du Projet « {proj} »"

    return desc.strip()

def extract_content_details(block: str) -> Dict[str, Any]:
    cd: Dict[str, Any] = {}
    money = extract_money_with_precision(block)
    if money:
        loan = {"amount": float(money["amount"]), "currency": money["currency"]}
        m_lender = re.search(r'\b(KfW|KFW|BEI|AFD|Banque\s+internationale\s+pour\s+la\s+reconstruction\s+et\s+le\s+d[ée]veloppement|Banque\s+européenne\s+d\'investissement)\b', block, re.IGNORECASE)
        if m_lender:
            loan["lender"] = m_lender.group(1).strip()
        m_benef = re.search(r'\b(MASEN|Moroccan Agency for Sustainable Energy|Office\s+[A-Z][A-Za-z\-\s]+|Minist[eè]re\s+[A-Za-z\-\s]+)\b', block, re.IGNORECASE)
        if m_benef:
            loan["beneficiary"] = m_benef.group(1).strip()
        m_proj = re.search(r'Projet\s+[«\"]?\s*([^»\"\n]+)\s*[»\"]?', block)
        if m_proj:
            loan["project"] = m_proj.group(1).strip()
        cd["loan_guarantee"] = loan

    # IGP
    if re.search(r'(Indication\s+Géographique|IGP)', block, re.IGNORECASE):
        igp = {}
        m_prod = re.search(r'«\s*([^»]+)\s*»', block)
        if m_prod:
            igp["name"] = m_prod.group(1).strip()
        m_area = re.search(r'aire\s+g[ée]ographique[^:]*:\s*(.+?)\.', block, re.IGNORECASE | re.DOTALL)
        if m_area:
            igp["area"] = m_area.group(1).strip()
        m_cert = re.search(r'organisme\s+de\s+certification\s+et\s+de\s+contr[oô]le\s+«?\s*([^»\n]+)\s*»?', block, re.IGNORECASE)
        if m_cert:
            igp["certifier"] = m_cert.group(1).strip()
        if igp:
            cd["geographical_indication"] = igp

    return cd

def map_offset_to_page(offset: int, page_markers: Dict[int, int]) -> int:
    if not page_markers:
        return 0
    chosen = 0
    best_off = -1
    for p, off in page_markers.items():
        if off <= offset and off > best_off:
            best_off = off
            chosen = p
    return chosen

def parse_legal_text_block(block: Block, page_markers: Dict[int, int]) -> Optional[Dict[str, Any]]:
    header = block.header or ''
    typ = _detect_type(header)
    num = _extract_number(header)
    if not (typ or num):
        return None
    hijri, greg = _extract_dates_from_block(block.text)
    title = _extract_title_from_block(header, block.text)
    sigs = extract_signatories(block.text)
    cd = extract_content_details(block.text)
    description = build_description_fr(typ, num, hijri, greg, title, block.text)
    page_start = map_offset_to_page(block.start, page_markers)
    obj: Dict[str, Any] = {
        "type": typ or "",
        "number": num or "",
        "title": title or "",
        "publication_date_hijri": hijri or "",
        "publication_date_gregorian": greg or "",
        "page_start": page_start,
        "description": description,
        "signatories": sigs or []
    }
    if cd:
        obj["content_details"] = cd
    return obj

def dedupe_legal_texts(objs: List[Dict[str, Any]], blocks: List[Block]) -> List[Dict[str, Any]]:
    # Keep longest block per (type, number). If number empty, use (type, title) as fallback.
    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    lengths: Dict[Tuple[str, str], int] = {}
    for obj, blk in zip(objs, blocks):
        key = (obj.get('type',''), obj.get('number') or obj.get('title',''))
        L = len(blk.text)
        if key not in by_key or L > lengths[key]:
            by_key[key] = obj
            lengths[key] = L
        else:
            # merge signatories & content_details
            base = by_key[key]
            # signatories
            existing = {(s.get('name',''), s.get('title','')) for s in base.get('signatories', [])}
            for s in obj.get('signatories', []):
                tup = (s.get('name',''), s.get('title',''))
                if tup not in existing and s.get('name',''):
                    base.setdefault('signatories', []).append(s)
            # fill missing fields if empty
            for fld in ["title","publication_date_hijri","publication_date_gregorian"]:
                if not base.get(fld) and obj.get(fld):
                    base[fld] = obj[fld]
            # content_details union (shallow)
            if obj.get('content_details'):
                base.setdefault('content_details', {}).update(obj['content_details'])
            # choose earliest page_start if base has 0
            if (base.get('page_start') in (0,None)) and obj.get('page_start'):
                base['page_start'] = obj['page_start']
    return list(by_key.values())

# ========================= Orchestration =========================

def extract_and_parse_pdf(pdf_path: str, dpi: int = DEFAULT_DPI, cache_dir: str = DEFAULT_CACHE_DIR,
                          force_ocr: bool = False) -> Dict[str, Any]:
    full_text, pages = extract_text_pages_hybrid(pdf_path, dpi=dpi, cache_dir=cache_dir, force_ocr=force_ocr)
    full_text = _strip_controls(full_text)
    meta = parse_bulletin_metadata(full_text)
    toc = parse_table_of_contents(full_text)
    page_markers = extract_page_markers(full_text)

    blocks = split_legal_texts(full_text)
    parsed_objs: List[Dict[str, Any]] = []
    kept_blocks: List[Block] = []
    for b in blocks:
        obj = parse_legal_text_block(b, page_markers)
        if obj:
            parsed_objs.append(obj)
            kept_blocks.append(b)
    legal_texts = dedupe_legal_texts(parsed_objs, kept_blocks)

    return {
        "bulletin_metadata": meta,
        "table_of_contents": toc,
        "legal_texts": legal_texts,
    }

# ========================= CLI =========================

def to_json_safe(obj: Any) -> Any:
    """Ensure JSON compatibility (Decimal -> float)."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(x) for x in obj]
    return obj