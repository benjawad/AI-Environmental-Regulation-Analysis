# # # @title # Gpt5
# # from __future__ import annotations
# # import os
# # import re
# # import json
# # import hashlib
# # import argparse
# # import unicodedata
# # from dataclasses import dataclass
# # from typing import Any, Dict, List, Optional, Tuple
# # from concurrent.futures import ThreadPoolExecutor, as_completed
# # from decimal import Decimal, InvalidOperation

# # # Third-party (hard dependency)
# # try:
# #     import fitz  # PyMuPDF
# # except Exception as e:
# #     raise RuntimeError("PyMuPDF (fitz) is required. Install with `pip install pymupdf`.") from e

# # # OCR (optional)
# # try:
# #     import pytesseract
# #     from PIL import Image
# #     OCR_AVAILABLE = True
# # except Exception:
# #     OCR_AVAILABLE = False
# #     Image = None  # type: ignore

# # # Regex with Unicode classes (optional)
# # try:
# #     import regex as regex_mod
# # except Exception:
# #     regex_mod = None

# # # Language detection (optional)
# # try:
# #     import langid
# # except Exception:
# #     langid = None

# # # Dates (optional)
# # try:
# #     import dateparser
# # except Exception:
# #     dateparser = None

# # try:
# #     from dateutil import parser as dateutil_parser
# # except Exception:
# #     dateutil_parser = None

# # # ========================= Config =========================
# # DEFAULT_DPI = 200
# # DEFAULT_CACHE_DIR = "./ocr_cache"
# # MIN_TEXT_LENGTH_FOR_PAGEMODE = 40
# # MAX_WORKERS = max(1, (os.cpu_count() or 2) - 1)

# # MONTHS_FR = {
# #     "janvier": "01",
# #     "février": "02", "fevrier": "02",
# #     "mars": "03",
# #     "avril": "04",
# #     "mai": "05",
# #     "juin": "06",
# #     "juillet": "07",
# #     "août": "08", "aout": "08",
# #     "septembre": "09",
# #     "octobre": "10",
# #     "novembre": "11",
# #     "décembre": "12", "decembre": "12",
# # }

# # FR_MONTHS_IDX = {
# #     1: "janvier", 2: "février", 3: "mars", 4: "avril", 5: "mai", 6: "juin",
# #     7: "juillet", 8: "août", 9: "septembre", 10: "octobre", 11: "novembre", 12: "décembre"
# # }

# # def format_date_fr(iso_date: str) -> str:
# #     """ '2025-06-23' -> '23 juin 2025' """
# #     if not iso_date or not re.match(r'^\d{4}-\d{2}-\d{2}$', iso_date):
# #         return (iso_date or '').strip()
# #     y, m, d = iso_date.split('-')
# #     try:
# #         return f"{int(d)} {FR_MONTHS_IDX[int(m)]} {int(y)}"
# #     except Exception:
# #         return (iso_date or '').strip()

# # CURRENCY_MAP = {
# #     "€": "EUR",
# #     "eur": "EUR",
# #     "euro": "EUR",
# #     "euros": "EUR",
# #     "dh": "MAD",
# #     "dhs": "MAD",
# #     "mad": "MAD",
# # }

# # # ========================= Utils =========================

# # def _strip_controls(text: str) -> str:
# #     return ''.join(ch for ch in text or '' if unicodedata.category(ch)[0] != 'C' or ch in '\n\r\t')

# # def has_arabic(text: str) -> bool:
# #     if not text:
# #         return False
# #     if regex_mod:
# #         try:
# #             return bool(regex_mod.search(r'[\p{Arabic}]', text))
# #         except Exception:
# #             pass
# #     return bool(re.search(r'[\u0600-\u06FF]', text))

# # def detect_lang(text: str) -> str:
# #     if not text:
# #         return 'fr'
# #     if has_arabic(text):
# #         return 'ar'
# #     if langid:
# #         lid, conf = langid.classify(text[:4000])
# #         if lid in {'fr','ar'} and conf >= 0.65:
# #             return lid
# #     return 'fr'

# # def normalize_ocr_common(text: str) -> str:
# #     if not text:
# #         return text
# #     t = (text.replace('\u00A0', ' ')
# #             .replace('’', "'").replace('“', '"').replace('”', '"')
# #             .replace('–', '-').replace('—', '-'))
# #     # fix typical OCR digit confusions only when surrounded by digits
# #     if regex_mod:
# #         def fix_digits(m):
# #             s = m.group(0)
# #             return (s.replace('O','0').replace('o','0').replace('l','1').replace('I','1').replace('S','5'))
# #         try:
# #             t = regex_mod.sub(r'(?<=\d)[OolIS]{1,}(?=\d)', fix_digits, t)
# #             t = regex_mod.sub(r'(?:(?<=\b)|(?<=-))[OolIS0-9]{2,}(?=\b|[-/])', fix_digits, t)
# #         except Exception:
# #             pass
# #     t = re.sub(r'[ \t]{2,}', ' ', t)
# #     t = re.sub(r'\n{3,}', '\n\n', t)
# #     t = re.sub(r'(\w+)-\n(\w+)', r'\1\2', t)  # join hyphenated breaks
# #     return t

# # def normalize_plain(text: str) -> str:
# #     if not text:
# #         return text
# #     t = (text.replace('\u00A0', ' ')
# #             .replace('’', "'").replace('“', '"').replace('”', '"')
# #             .replace('–', '-').replace('—', '-'))
# #     t = re.sub(r'[ \t]{2,}', ' ', t)
# #     t = re.sub(r'\n{3,}', '\n\n', t)
# #     t = re.sub(r'(\w+)-\n(\w+)', r'\1\2', t)
# #     return t

# # def parse_french_date_to_iso(date_str: str) -> Optional[str]:
# #     if not date_str:
# #         return None
# #     s = date_str.strip().replace('1er', '1')
# #     s_low = unicodedata.normalize('NFKD', s).lower().replace(',', ' ')
# #     if dateparser:
# #         dt = dateparser.parse(s_low, languages=['fr'])
# #         if dt:
# #             return dt.strftime('%Y-%m-%d')
# #     m = re.search(r'(\d{1,2})\s+([a-zêéèîïôûùâàäöü]+)\s+(\d{4})', s_low, re.IGNORECASE)
# #     if m:
# #         d, mon, y = m.groups()
# #         mon_num = MONTHS_FR.get(mon)
# #         if mon_num:
# #             return f"{int(y):04d}-{mon_num}-{int(d):02d}"
# #     if dateutil_parser:
# #         try:
# #             dt = dateutil_parser.parse(s, dayfirst=True)
# #             return dt.strftime('%Y-%m-%d')
# #         except Exception:
# #             pass
# #     return None

# # def fr_amount_string(amount: float) -> str:
# #     """Format 578300000.0 -> '578.300.000,00'"""
# #     d = Decimal(str(amount)).quantize(Decimal('0.01'))
# #     s = f"{d:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
# #     return s

# # # ========================= Extraction (text/OCR) =========================

# # @dataclass
# # class PageResult:
# #     page_number: int
# #     text: str
# #     used_ocr: bool
# #     lang: str

# # def _render_page_to_pil(page: fitz.Page, dpi: int = DEFAULT_DPI):
# #     mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
# #     pix = page.get_pixmap(matrix=mat, alpha=False)
# #     mode = 'RGB' if pix.n < 4 else 'RGBA'
# #     return Image.frombytes(mode, [pix.width, pix.height], pix.samples)

# # def ocr_page_image(img: Image.Image, ocr_lang: str = 'fra+ara', ocr_config: str = '--psm 6') -> str: # type: ignore
# #     if not OCR_AVAILABLE:
# #         return ''
# #     return pytesseract.image_to_string(img, lang=ocr_lang, config=ocr_config)

# # def extract_text_pages_hybrid(pdf_path: str, dpi: int = DEFAULT_DPI,
# #                               cache_dir: str = DEFAULT_CACHE_DIR,
# #                               force_ocr: bool = False) -> Tuple[str, List[PageResult]]:
# #     """Text-first with OCR fallback per page; cached by file mtime + dpi"""
# #     os.makedirs(cache_dir, exist_ok=True)
# #     try:
# #         stat = os.stat(pdf_path)
# #         key_material = f"{os.path.abspath(pdf_path)}::{stat.st_mtime_ns}::dpi={dpi}"
# #     except Exception:
# #         key_material = f"{os.path.abspath(pdf_path)}::nodest"
# #     cache_key = hashlib.sha256(key_material.encode()).hexdigest()
# #     cache_file = os.path.join(cache_dir, f"{cache_key}.json")

# #     if os.path.exists(cache_file) and not force_ocr:
# #         try:
# #             with open(cache_file, 'r', encoding='utf-8') as f:
# #                 data = json.load(f)
# #             pages = [PageResult(**p) for p in data.get('pages', [])]
# #             return "\n\n".join(p.text for p in pages), pages
# #         except Exception:
# #             pass

# #     doc = fitz.open(pdf_path)
# #     n_pages = doc.page_count
# #     page_results: List[Optional[PageResult]] = [None] * n_pages
# #     pages_to_ocr: List[int] = []

# #     # Pass 1: embedded text
# #     for i in range(n_pages):
# #         page = doc.load_page(i)
# #         try:
# #             txt = page.get_text('text')
# #         except Exception:
# #             txt = ''
# #         txt = normalize_plain((txt or '').strip())
# #         if txt and len(txt) >= MIN_TEXT_LENGTH_FOR_PAGEMODE and not (force_ocr and OCR_AVAILABLE):
# #             page_results[i] = PageResult(page_number=i+1, text=txt, used_ocr=False, lang=detect_lang(txt))
# #         else:
# #             pages_to_ocr.append(i)

# #     # Pass 2: OCR (only if module available)
# #     if pages_to_ocr and OCR_AVAILABLE:
# #         def ocr_worker(idx: int) -> PageResult:
# #             p = doc.load_page(idx)
# #             img = _render_page_to_pil(p, dpi=dpi)
# #             sample = pytesseract.image_to_string(img.crop((0, 0, min(1000, img.width), min(220, img.height))),
# #                                                  lang='fra+ara', config='--psm 6')[:2000]
# #             lang_hint = 'ara' if detect_lang(sample) == 'ar' else 'fra'
# #             try:
# #                 otext = ocr_page_image(img, ocr_lang=f"{lang_hint}+fra+ara", ocr_config='--psm 6')
# #                 otext = normalize_ocr_common(otext)
# #             except Exception:
# #                 otext = ''
# #             return PageResult(page_number=idx+1, text=otext, used_ocr=True, lang=detect_lang(otext))

# #         workers = min(MAX_WORKERS, max(1, len(pages_to_ocr)))
# #         with ThreadPoolExecutor(max_workers=workers) as ex:
# #             futures = {ex.submit(ocr_worker, i): i for i in pages_to_ocr}
# #             for fut in as_completed(futures):
# #                 idx = futures[fut]
# #                 try:
# #                     page_results[idx] = fut.result()
# #                 except Exception:
# #                     page_results[idx] = PageResult(page_number=idx+1, text='', used_ocr=True, lang='fr')
# #     else:
# #         # If OCR not available, mark those pages as empty
# #         for idx in pages_to_ocr:
# #             page_results[idx] = PageResult(page_number=idx+1, text='', used_ocr=False, lang='fr')

# #     pages_final: List[PageResult] = [p if p else PageResult(page_number=i+1, text='', used_ocr=False, lang='fr')
# #                                      for i, p in enumerate(page_results)]

# #     joined = "\n\n".join(p.text for p in pages_final)
# #     try:
# #         with open(cache_file, 'w', encoding='utf-8') as f:
# #             json.dump({"pages": [p.__dict__ for p in pages_final]}, f, ensure_ascii=False)
# #     except Exception:
# #         pass
# #     return joined, pages_final

# # # ========================= Parsing =========================

# # def extract_page_markers(text: str) -> Dict[int, int]:
# #     markers = {}
# #     pattern = r'\n\s*([0-9]{2,5})\s+(?:BULLETIN(?:\s+OFFICIEL)?|النشرة)\b'
# #     for m in re.finditer(pattern, text or '', re.IGNORECASE):
# #         try:
# #             p = int(m.group(1))
# #         except Exception:
# #             continue
# #         markers[p] = m.start()
# #     return markers

# # def find_issn(text: str) -> Optional[str]:
# #     m = re.search(r'ISSN\s*[:\s]*([0-9]{4}\s*-\s*[0-9]{4})', text or '', re.IGNORECASE)
# #     return m.group(1).replace(' ', '') if m else None

# # def find_issue_and_dates(text: str) -> Tuple[str, str, str]:
# #     issue = ''
# #     hijri = ''
# #     greg = ''
# #     m_issue = re.search(r'N[°oº]\s*([0-9]{2,6})', text or '', re.IGNORECASE)
# #     if m_issue:
# #         issue = m_issue.group(1)
# #     # hijri (FR words) + (greg)
# #     if regex_mod:
# #         m = regex_mod.search(r'([0-9]{1,2}\s+[^\(\)\d\n]{3,30}\s+[0-9]{3,4})\s*\(\s*([^\)]+)\s*\)', text or '', regex_mod.IGNORECASE)
# #     else:
# #         m = re.search(r'([0-9]{1,2}\s+[A-Za-zéèêëîïôûùâàäöü\-]+?\s+[0-9]{3,4})\s*\(\s*([^\)]+)\s*\)', text or '', re.IGNORECASE)
# #     if m:
# #         hijri = m.group(1).strip()
# #         greg = parse_french_date_to_iso(m.group(2).strip()) or ''
# #     else:
# #         mg = re.search(r'\(\s*([0-9]{1,2}\s+[A-Za-zéèêëîïôûùâàäöü]+\s+[0-9]{4})\s*\)', text or '')
# #         if mg:
# #             greg = parse_french_date_to_iso(mg.group(1)) or ''
# #     return issue, hijri, greg

# # def parse_bulletin_metadata(full_text: str) -> Dict[str, str]:
# #     issue, hijri, greg = find_issue_and_dates(full_text)
# #     issn = find_issn(full_text) or ''
# #     m_year = re.search(r'([A-Z][a-zéèêàâîïôûù\- ]+année)\s+[–-]\s+N', full_text or '')
# #     pub_year_desc = m_year.group(1).strip() if m_year else ''
# #     return {
# #         "issue_number": issue or '',
# #         "publication_year_description": pub_year_desc or '',
# #         "date_hijri": hijri or '',
# #         "date_gregorian": greg or '',
# #         "issn": issn or ''
# #     }

# # # -------- TOC: robust, stateful line stitching ----------
# # def parse_table_of_contents(full_text: str) -> List[Dict[str, Any]]:
# #     lines = full_text.splitlines()
# #     # Find SOMMAIRE / TABLE DES MATIERES
# #     start_idx = None
# #     for i, ln in enumerate(lines[:1500]):
# #         if re.search(r'^\s*(SOMMAIRE|TABLE\s+DES\s+MATI[EÈ]RES)\b', ln, flags=re.IGNORECASE):
# #             start_idx = i
# #             break
# #     if start_idx is None:
# #         # Fallback: single-line dotted leaders
# #         toc = []
# #         dotted = re.compile(r'^(.*?)\s*\.{2,}\s*(\d{2,5})$')
# #         for raw in lines[:1500]:
# #             s = raw.strip()
# #             m = dotted.match(s)
# #             if m and len(m.group(1).strip(' .-')) > 10:
# #                 toc.append({"category": "", "title": m.group(1).strip(' .-'), "description": "", "page": int(m.group(2))})
# #         return toc

# #     toc_entries: List[Dict[str, Any]] = []
# #     buf: List[str] = []
# #     current_category = ""
# #     dotted = re.compile(r'(.*?)\s*\.{2,}\s*(\d{2,5})\s*$')

# #     for ln in lines[start_idx:start_idx+2500]:
# #         s = ln.strip()
# #         if not s:
# #             continue
# #         # categories
# #         if re.search(r'^(TEXTES?\s+G[EÉ]N[ÉE]RAUX|TEXTES?\s+PARTICULIERS)', s, flags=re.IGNORECASE):
# #             current_category = re.sub(r'\s+', ' ', s.upper())
# #             continue
# #         # skip page headers/footers that creep in
# #         if re.search(r'^\d{1,4}\s+(BULLETIN(?:\s+OFFICIEL)?|النشرة)\b', s, flags=re.IGNORECASE):
# #             continue
# #         m = dotted.search(s)
# #         if m:
# #             piece = m.group(1).strip(' .\t-–—')
# #             if piece:
# #                 buf.append(piece)
# #             title = ' '.join([re.sub(r'\s+', ' ', b) for b in buf]).strip()
# #             if len(title) >= 10:
# #                 try:
# #                     page = int(m.group(2))
# #                     toc_entries.append({
# #                         "category": current_category,
# #                         "title": title,
# #                         "description": "",
# #                         "page": page,
# #                     })
# #                 except Exception:
# #                     pass
# #             buf = []
# #         else:
# #             buf.append(s)

# #     # dedupe on (title, page)
# #     uniq = {}
# #     for e in toc_entries:
# #         key = (e['title'], e['page'])
# #         if key not in uniq:
# #             uniq[key] = e
# #     return list(uniq.values())

# # # -------- Money: precise, robust (FR separators) --------
# # def extract_money_with_precision(chunk: str) -> Optional[Dict[str, Any]]:
# #     """Return {'amount': float, 'currency': 'EUR'|'MAD', 'spelled'?: str} or None"""
# #     if not chunk:
# #         return None

# #     rx_spelled = re.compile(
# #         r"montant\s+de\s+([a-zàâçéèêëîïôûùüÿ\-’'\s]{8,160})\s*\(\s*([0-9][0-9\.\s\u202F,]+)\s*(€|euros?|eur|dh?s?|mad)\s*\)",
# #         re.IGNORECASE,
# #     )
# #     rx_plain = re.compile(
# #         r"\(\s*([0-9][0-9\.\s\u202F,]+)\s*(€|euros?|eur|dh?s?|mad)\s*\)"
# #         r"|([0-9][0-9\.\s\u202F,]+)\s*(€|euros?|eur|dh?s?|mad)\b",
# #         re.IGNORECASE,
# #     )

# #     spelled = None
# #     m = rx_spelled.search(chunk)
# #     if m:
# #         spelled = ' '.join(m.group(1).strip().split())
# #         num_raw = m.group(2)
# #         cur_raw = m.group(3)
# #     else:
# #         m = rx_plain.search(chunk)
# #         if not m:
# #             return None
# #         if m.group(1) and m.group(2):
# #             num_raw, cur_raw = m.group(1), m.group(2)
# #         else:
# #             num_raw, cur_raw = m.group(3), m.group(4)

# #     s = (num_raw or '').replace('\u202F', ' ').replace('\xa0', ' ').replace(' ', '')
# #     if ',' in s:
# #         s = s.replace('.', '').replace(',', '.')
# #     else:
# #         s = s.replace('.', '')

# #     try:
# #         # keep float for JSON compatibility; avoids Decimal->str fallback
# #         amount_val = float(s)
# #     except Exception:
# #         try:
# #             amount_val = float(Decimal(s))
# #         except Exception:
# #             return None

# #     cur = (cur_raw or '').lower()
# #     currency = 'EUR' if cur in ('€', 'eur', 'euro', 'euros') else ('MAD' if cur in ('dh', 'dhs', 'mad') else cur.upper())

# #     out = {"amount": amount_val, "currency": currency}
# #     if spelled:
# #         out["spelled"] = spelled
# #     return out

# # # -------- Signatories: contextual & filtered --------
# # BLOCKLIST = {
# #     "BULLETIN","ANNEXE","ARTICLE","SOMMAIRE","PAGE","TEXTES","SEEDS","INTERNATIONAL",
# #     "DÉCRÈTE","DECRETE","ARRÊTE","ARRETE","DÉCIDE","DECIDE","CHAPITRE","SECTION",
# #     "TABLE","IMPRIMERIE","OFFICIELLE","RABAT","KING","KAMAL"  # extend as needed
# # }

# # def _looks_like_name(name: str) -> bool:
# #     if not name or ' ' not in name:
# #         return False
# #     up = name.upper()
# #     # must have vowels
# #     if not re.search(r'[AEIOUY]', up):
# #         return False
# #     # 2–6 tokens
# #     parts = [p for p in up.split() if p]
# #     if not (2 <= len(parts) <= 6):
# #         return False
# #     # blocklist as whole words
# #     for tok in BLOCKLIST:
# #         if re.search(rf'\b{re.escape(tok)}\b', up):
# #             return False
# #     # avoid numbers / annex refs
# #     if re.search(r'\d', up):
# #         return False
# #     return True

# # def extract_signatories(block_text: str) -> List[Dict[str, str]]:
# #     """Strict: last 20% of text only; contextual patterns first; then conservative fallback."""
# #     if not block_text:
# #         return []
# #     L = len(block_text)
# #     tail = block_text[int(L*0.8):]

# #     results: List[Dict[str, str]] = []

# #     # Pattern A: "Pour contreseing : <Title>\n<NAME>."
# #     for m in re.finditer(r'Pour\s+contreseing\s*[:\-]?\s*([^\n,]+)\s*\n\s*([A-ZÉÈÊÂÎÏÔÛÙÄËÖÜ\'\-\s]{3,80})\.?', tail, flags=re.IGNORECASE):
# #         title = m.group(1).strip()
# #         name = ' '.join(m.group(2).split())
# #         if _looks_like_name(name):
# #             results.append({"name": name, "title": title})

# #     # Pattern B: "Fait à ..., le ...\n<Title>,\n<NAME>."
# #     for m in re.finditer(r'Fait\s+à\s+[^\n,]+,\s*le\s+[^\n]+\n\s*([A-Za-zéèêàâîïôûù\'\-\s]{3,80})\s*,\s*\n\s*([A-ZÉÈÊÂÎÏÔÛÙÄËÖÜ\'\-\s]{3,80})\.?', tail, flags=re.IGNORECASE):
# #         title = m.group(1).strip()
# #         name = ' '.join(m.group(2).split())
# #         if _looks_like_name(name):
# #             results.append({"name": name, "title": title})

# #     # Conservative fallback: stand-alone uppercase line (rarely used now)
# #     for m in re.finditer(r'\n\s*([A-ZÉÈÊÂÎÏÔÛÙÄËÖÜ\'\-\s]{3,80})\s*\.?\s*\n', tail):
# #         name = ' '.join(m.group(1).split())
# #         if _looks_like_name(name):
# #             results.append({"name": name, "title": ""})

# #     # De-duplicate
# #     seen = set()
# #     out: List[Dict[str, str]] = []
# #     for s in results:
# #         key = (s.get('name','').strip(), s.get('title','').strip())
# #         if key in seen:
# #             continue
# #         seen.add(key)
# #         out.append(s)
# #     return out

# # # -------- Legal block detection & field extraction --------
# # LEGAL_START_PATTERNS = [
# #     r'\bDécret\s+n[°oº]\s*[0-9][0-9\-\s]{0,20}[0-9]',
# #     r'\bDécret\s+du\s+\d{1,2}\s+[A-Za-zéèêëîïôûùâàäöü]+\s+\d{4}',
# #     r'\bArrêté\s+conjoint\s+n[°oº]?\s*[0-9][0-9\-\s]{0,20}[0-9]',
# #     r'\bArr[ée]t[ée]\s+n[°oº]?\s*[0-9][0-9\-\s]{0,20}[0-9]',
# #     r'\bArr[ée]t[ée]\s+du\s+\d{1,2}\s+[A-Za-zéèêëîïôûùâàäöü]+\s+\d{4}',
# #     r'\bLoi\s+n[°oº]?\s*[0-9][0-9\-\s]{0,20}[0-9]',
# #     r'\bDécision\s+n[°oº]?\s*[0-9][0-9\-\s]{0,20}[0-9]',
# #     r'\bDécision\s+du\s+\d{1,2}\s+[A-Za-zéèêëîïôûùâàäöü]+\s+\d{4}',
# #     r'\bOrdonnance\s+n[°oº]?\s*[0-9]'
# # ]
# # LEGAL_START_RE = re.compile('|'.join(LEGAL_START_PATTERNS), flags=re.IGNORECASE)

# # @dataclass
# # class Block:
# #     start: int
# #     end: int
# #     header: str
# #     text: str

# # def split_legal_texts(full_text: str) -> List[Block]:
# #     starts = [m.start() for m in LEGAL_START_RE.finditer(full_text)]
# #     starts = sorted(set(starts))
# #     if not starts:
# #         return []
# #     blocks: List[Block] = []
# #     for i, s in enumerate(starts):
# #         e = starts[i+1] if i+1 < len(starts) else len(full_text)
# #         chunk = full_text[s:e].strip('\n')
# #         header = chunk.splitlines()[0] if chunk else ''
# #         blocks.append(Block(start=s, end=e, header=header, text=chunk))
# #     return blocks

# # def _detect_type(header: str) -> str:
# #     low = header.lower()
# #     if 'décret' in low: return 'Décret'
# #     if 'arrêté' in low or 'arrete' in low: return 'Arrêté'
# #     if low.startswith('loi') or ' loi ' in low: return 'Loi'
# #     if 'décision' in low or 'decision' in low: return 'Décision'
# #     return 'Texte'

# # def _extract_number(header: str) -> str:
# #     m = re.search(r'n[°oº]\s*([0-9][0-9\-\s]{0,20}[0-9])', header, re.IGNORECASE)
# #     if not m:
# #         return ''
# #     return re.sub(r'\s+', '', m.group(1)).replace('O','0').replace('o','0').replace('l','1')

# # TITLE_STOP_WORDS = re.compile(r'\b(D[ÉE]CR[ÈE]TE\s*:|ARR[ÊE]TE\s*:|D[ÉE]CIDE\s*:)\b', flags=re.IGNORECASE)

# # def _extract_dates_from_block(block_text: str) -> Tuple[str, str]:
# #     hijri = ''
# #     greg = ''
# #     head = '\n'.join(block_text.splitlines()[:6])  # look in first lines
# #     candidates = [head, block_text]
# #     for zone in candidates:
# #         m = re.search(r'du\s+([0-9]{1,2}\s+[^\(\)\d\n]{3,30}\s+[0-9]{3,4})\s*\(\s*([^\)]+)\s*\)', zone, re.IGNORECASE)
# #         if m:
# #             hijri = hijri or m.group(1).strip()
# #             greg = greg or (parse_french_date_to_iso(m.group(2).strip()) or '')
# #         else:
# #             mg = re.search(r'\(\s*([0-9]{1,2}\s+[A-Za-zéèêëîïôûùâàäöü]+\s+[0-9]{4})\s*\)', zone)
# #             if mg and not greg:
# #                 greg = parse_french_date_to_iso(mg.group(1).strip()) or ''
# #     return hijri, greg

# # def _extract_title_from_block(header: str, block_text: str) -> str:
# #     line = (header or '').strip()

# #     after = ''
# #     m = re.search(r'\)\s*(.+)$', line)
# #     if m and len(m.group(1).strip()) > 3:
# #         after = m.group(1).strip()
# #     if not after:
# #         m2 = re.search(r'n[°oº]\s*[0-9][0-9\-\s]{0,20}[0-9]\s*(.+)$', line, re.IGNORECASE)
# #         if m2 and len(m2.group(1).strip()) > 3:
# #             after = m2.group(1).strip()

# #     body = []
# #     for ln in (block_text or '').splitlines()[1:12]:
# #         ln = ln.strip()
# #         if not ln:
# #             continue
# #         if TITLE_STOP_WORDS.search(ln) or re.match(r'^(Vu|Vus|Attendu|Consid[ée]rant|Chapitre|Article)\b', ln, re.IGNORECASE):
# #             break
# #         body.append(ln)
# #         if ln.endswith('.'):
# #             break

# #     pieces = []
# #     if after:
# #         pieces.append(after.rstrip('.'))
# #     if (not pieces) or len(pieces[0].split()) < 4 or re.match(r"^(approuvant|relatif|portant|modifiant|fixant|autorisant)\b", pieces[0], re.IGNORECASE):
# #         if body:
# #             sentence = ' '.join(body)
# #             sentence = TITLE_STOP_WORDS.split(sentence)[0]
# #             sentence = sentence.split(' .')[0].split('. ')[0].rstrip('.')
# #             if sentence and sentence.lower() not in pieces:
# #                 pieces.append(sentence)

# #     title = ' '.join(pieces).strip()
# #     if len(title) < 10:
# #         title = (after or ' '.join(body)).strip().rstrip('.')
# #         title = TITLE_STOP_WORDS.split(title)[0].strip()
# #     return title

# # def build_description_fr(typ: str, num: str, hijri: str, greg_iso: str, title: str, block_text: str) -> str:
# #     parts = []
# #     if typ:
# #         parts.append(typ)
# #     if num:
# #         parts.append(f"n° {num}")
# #     greg = format_date_fr(greg_iso) if greg_iso else ''
# #     if hijri and greg:
# #         parts.append(f"du {hijri} ({greg})")
# #     elif greg:
# #         parts.append(f"du {greg}")
# #     elif hijri:
# #         parts.append(f"du {hijri}")

# #     desc = " ".join(parts).strip()
# #     if title:
# #         desc = (desc + " " + title.rstrip('.')).strip()

# #     money = extract_money_with_precision(block_text)
# #     if money:
# #         amt_str = fr_amount_string(money["amount"])
# #         unit = 'euros' if money["currency"] == 'EUR' else 'dirhams'
# #         if money.get("spelled"):
# #             desc += f", d'un montant de {money['spelled']} ({amt_str} {unit})"
# #         else:
# #             desc += f", d'un montant de {amt_str} {unit}"

# #     m_conclu = re.search(r"\bconclu[e]?\s+le\s+([0-9]{1,2}\s+[a-zàâçéèêëîïôûùüÿ]+(?:\s+\d{4})?)", block_text, re.IGNORECASE)
# #     if m_conclu:
# #         concl_date = ' '.join(m_conclu.group(1).split())
# #         if concl_date not in desc:
# #             desc += f", conclu le {concl_date}"

# #     # lender/beneficiary/project mentions (concise & factual)
# #     m_lender = re.search(r'\b(Banque\s+internationale\s+pour\s+la\s+reconstruction\s+et\s+le\s+d[ée]veloppement|KfW|KFW|BEI|AFD)\b', block_text, re.IGNORECASE)
# #     if m_lender and m_lender.group(1) not in desc:
# #         desc += f" avec {m_lender.group(1)}"

# #     m_benef = re.search(r'\b(MASEN|Moroccan Agency for Sustainable Energy|Office\s+[A-Z][A-Za-z\-\s]+)\b', block_text, re.IGNORECASE)
# #     if m_benef and m_benef.group(1) not in desc:
# #         desc += f", au profit de {m_benef.group(1)}"

# #     m_proj = re.search(r'Projet\s+[«\"]?\s*([^»\"\n]+)\s*[»\"]?', block_text)
# #     if m_proj:
# #         proj = m_proj.group(1).strip()
# #         if proj and proj not in desc:
# #             desc += f", pour le financement du Projet « {proj} »"

# #     return desc.strip()

# # def extract_content_details(block: str) -> Dict[str, Any]:
# #     cd: Dict[str, Any] = {}
# #     money = extract_money_with_precision(block)
# #     if money:
# #         loan = {"amount": float(money["amount"]), "currency": money["currency"]}
# #         m_lender = re.search(r'\b(KfW|KFW|BEI|AFD|Banque\s+internationale\s+pour\s+la\s+reconstruction\s+et\s+le\s+d[ée]veloppement|Banque\s+européenne\s+d\'investissement)\b', block, re.IGNORECASE)
# #         if m_lender:
# #             loan["lender"] = m_lender.group(1).strip()
# #         m_benef = re.search(r'\b(MASEN|Moroccan Agency for Sustainable Energy|Office\s+[A-Z][A-Za-z\-\s]+|Minist[eè]re\s+[A-Za-z\-\s]+)\b', block, re.IGNORECASE)
# #         if m_benef:
# #             loan["beneficiary"] = m_benef.group(1).strip()
# #         m_proj = re.search(r'Projet\s+[«\"]?\s*([^»\"\n]+)\s*[»\"]?', block)
# #         if m_proj:
# #             loan["project"] = m_proj.group(1).strip()
# #         cd["loan_guarantee"] = loan

# #     # IGP
# #     if re.search(r'(Indication\s+Géographique|IGP)', block, re.IGNORECASE):
# #         igp = {}
# #         m_prod = re.search(r'«\s*([^»]+)\s*»', block)
# #         if m_prod:
# #             igp["name"] = m_prod.group(1).strip()
# #         m_area = re.search(r'aire\s+g[ée]ographique[^:]*:\s*(.+?)\.', block, re.IGNORECASE | re.DOTALL)
# #         if m_area:
# #             igp["area"] = m_area.group(1).strip()
# #         m_cert = re.search(r'organisme\s+de\s+certification\s+et\s+de\s+contr[oô]le\s+«?\s*([^»\n]+)\s*»?', block, re.IGNORECASE)
# #         if m_cert:
# #             igp["certifier"] = m_cert.group(1).strip()
# #         if igp:
# #             cd["geographical_indication"] = igp

# #     return cd

# # def map_offset_to_page(offset: int, page_markers: Dict[int, int]) -> int:
# #     if not page_markers:
# #         return 0
# #     chosen = 0
# #     best_off = -1
# #     for p, off in page_markers.items():
# #         if off <= offset and off > best_off:
# #             best_off = off
# #             chosen = p
# #     return chosen

# # def parse_legal_text_block(block: Block, page_markers: Dict[int, int]) -> Optional[Dict[str, Any]]:
# #     header = block.header or ''
# #     typ = _detect_type(header)
# #     num = _extract_number(header)
# #     if not (typ or num):
# #         return None
# #     hijri, greg = _extract_dates_from_block(block.text)
# #     title = _extract_title_from_block(header, block.text)
# #     sigs = extract_signatories(block.text)
# #     cd = extract_content_details(block.text)
# #     description = build_description_fr(typ, num, hijri, greg, title, block.text)
# #     page_start = map_offset_to_page(block.start, page_markers)
# #     obj: Dict[str, Any] = {
# #         "type": typ or "",
# #         "number": num or "",
# #         "title": title or "",
# #         "publication_date_hijri": hijri or "",
# #         "publication_date_gregorian": greg or "",
# #         "page_start": page_start,
# #         "description": description,
# #         "signatories": sigs or []
# #     }
# #     if cd:
# #         obj["content_details"] = cd
# #     return obj

# # def dedupe_legal_texts(objs: List[Dict[str, Any]], blocks: List[Block]) -> List[Dict[str, Any]]:
# #     # Keep longest block per (type, number). If number empty, use (type, title) as fallback.
# #     by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
# #     lengths: Dict[Tuple[str, str], int] = {}
# #     for obj, blk in zip(objs, blocks):
# #         key = (obj.get('type',''), obj.get('number') or obj.get('title',''))
# #         L = len(blk.text)
# #         if key not in by_key or L > lengths[key]:
# #             by_key[key] = obj
# #             lengths[key] = L
# #         else:
# #             # merge signatories & content_details
# #             base = by_key[key]
# #             # signatories
# #             existing = {(s.get('name',''), s.get('title','')) for s in base.get('signatories', [])}
# #             for s in obj.get('signatories', []):
# #                 tup = (s.get('name',''), s.get('title',''))
# #                 if tup not in existing and s.get('name',''):
# #                     base.setdefault('signatories', []).append(s)
# #             # fill missing fields if empty
# #             for fld in ["title","publication_date_hijri","publication_date_gregorian"]:
# #                 if not base.get(fld) and obj.get(fld):
# #                     base[fld] = obj[fld]
# #             # content_details union (shallow)
# #             if obj.get('content_details'):
# #                 base.setdefault('content_details', {}).update(obj['content_details'])
# #             # choose earliest page_start if base has 0
# #             if (base.get('page_start') in (0,None)) and obj.get('page_start'):
# #                 base['page_start'] = obj['page_start']
# #     return list(by_key.values())

# # # ========================= Orchestration =========================

# # def extract_and_parse_pdf(pdf_path: str, dpi: int = DEFAULT_DPI, cache_dir: str = DEFAULT_CACHE_DIR,
# #                           force_ocr: bool = False) -> Dict[str, Any]:
# #     full_text, pages = extract_text_pages_hybrid(pdf_path, dpi=dpi, cache_dir=cache_dir, force_ocr=force_ocr)
# #     full_text = _strip_controls(full_text)
# #     meta = parse_bulletin_metadata(full_text)
# #     toc = parse_table_of_contents(full_text)
# #     page_markers = extract_page_markers(full_text)

# #     blocks = split_legal_texts(full_text)
# #     parsed_objs: List[Dict[str, Any]] = []
# #     kept_blocks: List[Block] = []
# #     for b in blocks:
# #         obj = parse_legal_text_block(b, page_markers)
# #         if obj:
# #             parsed_objs.append(obj)
# #             kept_blocks.append(b)
# #     legal_texts = dedupe_legal_texts(parsed_objs, kept_blocks)

# #     return {
# #         "bulletin_metadata": meta,
# #         "table_of_contents": toc,
# #         "legal_texts": legal_texts,
# #     }

# # # ========================= CLI =========================

# # def _to_json_safe(obj: Any) -> Any:
# #     """Ensure JSON compatibility (Decimal -> float)."""
# #     if isinstance(obj, Decimal):
# #         return float(obj)
# #     if isinstance(obj, dict):
# #         return {k: _to_json_safe(v) for k, v in obj.items()}
# #     if isinstance(obj, list):
# #         return [_to_json_safe(x) for x in obj]
# #     return obj


# """
# Unified Bulletin Officiel parser focusing on TEXTES GÉNÉRAUX and
# TEXTES PARTICULIERS.

# This module parses Moroccan Bulletin Officiel (BO) PDF files and
# extracts complete legal texts (dahirs, decrets, arretés, decisions)
# appearing in the categories “TEXTES GÉNÉRAUX” and
# “TEXTES PARTICULIERS”.  It deliberately skips the cover, sommaire
# (table of contents) and any sections labelled “AVIS ET
# COMMUNICATIONS”.  Each document is returned with metadata (issue
# number, publication dates, title, reference number/date, page range,
# preamble visas, enacting clause, articles, and signatures) in a
# structured JSON format.

# Key characteristics
# -------------------

# * **Hybrid extraction** – Each page is extracted using PyMuPDF.  If
#   the textual extraction yields fewer than ``MIN_TEXT_LENGTH``
#   characters, the parser falls back to OCR (via Tesseract) to
#   recover the content.
# * **Header detection with validation** – Candidate headers must begin
#   with a recognised document type (Dahir, Décret, Arrêté,
#   Décision).  A header is only accepted if an enacting clause
#   (PROMULGUE, DÉCRÈTE, ARRÊTE, DÉCIDE) appears within
#   ``VALIDATION_LOOKAHEAD`` characters downstream.  This guards
#   against spurious matches in the sommaire.
# * **Non‑greedy segmentation** – Documents extend from one accepted
#   header to the next.  Hard stops include subsequent category
#   banners or signature lines, preventing the merging of multiple
#   documents.
# * **Full article extraction** – Articles are delineated by
#   recognised headings.  Unlike earlier implementations, this parser
#   never truncates article content; it keeps all lines between
#   headings, only discarding obvious noise (page numbers, footers).  If
#   a document lacks articles but contains a valid enacting clause, it
#   is still retained.
# * **Category awareness** – Only documents appearing under
#   “TEXTES GÉNÉRAUX” or “TEXTES PARTICULIERS” are kept.  The parser
#   determines the category of each document based on the most recent
#   banner preceding it.  Documents outside these categories are
#   ignored.

# Usage
# -----

# The module can be executed as a script from the command line:

# ```
# python bo_parser_final.py /path/to/BO_7422_fr.pdf -o result.json
# ```

# If the ``-o``/``--output`` argument is omitted, the JSON result will
# be printed to standard output.  The output structure matches the
# examples provided in the project brief.

# This file does not depend on any external APIs; it uses only
# PyMuPDF and, optionally, Tesseract for OCR.  If pytesseract is not
# installed, OCR is skipped and the parser relies solely on the text
# extraction provided by PyMuPDF.
# """

# from __future__ import annotations

# import os
# import re
# import unicodedata
# from dataclasses import dataclass
# from typing import Any, Dict, Iterable, List, Optional, Tuple

# try:
#     import fitz  # PyMuPDF
# except Exception as e:
#     raise RuntimeError(
#         "PyMuPDF (fitz) is required to run this parser. Install it via 'pip install pymupdf'."
#     ) from e

# try:
#     from PIL import Image  # type: ignore
#     import pytesseract  # type: ignore
#     OCR_AVAILABLE = True
# except Exception:
#     OCR_AVAILABLE = False
#     Image = None  # type: ignore

# # -----------------------------------------------------------------------------
# # Configuration constants
# # -----------------------------------------------------------------------------

# # Minimum characters required from a page before skipping OCR.  Pages with
# # shorter extracted text are passed through pytesseract if available.
# MIN_TEXT_LENGTH = 80

# # Lookahead window (in characters) to validate that an enacting clause exists
# # after a candidate header.  A generous window helps avoid rejecting
# # decisions whose enacting clauses appear several pages later.
# VALIDATION_LOOKAHEAD = 20000

# # Regular expression for section banners.  Only TEXTES GÉNÉRAUX and TEXTES
# # PARTICULIERS are retained; AVIS ET COMMUNICATIONS serves as a stop marker.
# SECTION_BANNER_RE = re.compile(
#     r"(?mi)^\s*(TEXTES?\s+G[ÉE]N[ÉE]RAUX|TEXTES?\s+PARTICULIERS|AVIS\s+ET\s+COMMUNICATIONS)\s*$"
# )

# # Header pattern: recognise Dahir, Décret, Arrêté, Décision (with or without
# # accents).  The pattern anchors at the start of a line and allows an
# # optional qualifier (e.g. “conjoint”).
# DOC_HEADER_RE = re.compile(
#     r"(?mi)^(?:Dahir|Décret|Decret|Arr[éeê]t[éeê]|Arrete|Décision|Decision)(?:\s+conjoint)?\b"
# )

# # Enacting clauses: detect PROMULGUE, DÉCRÈTE, ARRÊTE, DÉCIDE (case-insensitive).
# # The parser uppercases the clause and appends a colon when returning it.
# ENACTING_CLAUSE_RE = re.compile(
#     r"(?mi)^(PROMULGUE\b|D[ÉE]CR[ÈE]TE\b|ARR[ÊE]TE\b|D[ÉE]CIDE\b)\s*[:：]?\s*$"
# )

# # Article headings: recognise full and abbreviated forms.
# ARTICLE_HEADER_RE = re.compile(
#     r"(?mi)^(ARTICLE\s+PREMIER|ARTICLE\s+UNIQUE|ARTICLE\s+\d+|ART\.\s*\d+|ART\.\s*PREMIER|ART\.\s*UNIQUE)\b"
# )

# # Mapping of French month names to numeric representations for date parsing.
# MONTHS_FR = {
#     "janvier": "01",
#     "février": "02",
#     "fevrier": "02",
#     "mars": "03",
#     "avril": "04",
#     "mai": "05",
#     "juin": "06",
#     "juillet": "07",
#     "août": "08",
#     "aout": "08",
#     "septembre": "09",
#     "octobre": "10",
#     "novembre": "11",
#     "décembre": "12",
#     "decembre": "12",
# }


# def normalise_text(text: str) -> str:
#     """Normalise punctuation and spacing in extracted text."""
#     if not text:
#         return ""
#     t = (
#         text.replace("\u00A0", " ")
#         .replace("’", "'")
#         .replace("“", '"').replace("”", '"')
#         .replace("–", "-").replace("—", "-")
#     )
#     # collapse multiple spaces
#     t = re.sub(r"[ \t]{2,}", " ", t)
#     # collapse multiple blank lines
#     t = re.sub(r"\n{3,}", "\n\n", t)
#     # rejoin hyphenated words broken at line endings
#     t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
#     return t


# def parse_french_date_to_iso(text: str) -> Optional[str]:
#     """Parse a French date into ISO YYYY-MM-DD, if possible."""
#     if not text:
#         return None
#     s = text.strip().replace("1er", "1")
#     s_low = unicodedata.normalize("NFKD", s).lower().replace(",", " ")
#     m = re.search(r"(\d{1,2})\s+([a-zéèêëàâîïôûùç\-]+)\s+(\d{4})", s_low)
#     if m:
#         day, month, year = m.groups()
#         month_num = MONTHS_FR.get(month)
#         if month_num:
#             return f"{int(year):04d}-{month_num}-{int(day):02d}"
#     return None


# @dataclass
# class PageText:
#     page_index: int
#     text: str
#     used_ocr: bool


# def extract_text_pages(pdf_path: str) -> List[PageText]:
#     """Extract text from each page with optional OCR fallback."""
#     doc = fitz.open(pdf_path)
#     pages: List[PageText] = []
#     for i, page in enumerate(doc):
#         text = ""
#         used_ocr = False
#         try:
#             text = page.get_text("text")
#         except Exception:
#             text = ""
#         text = normalise_text(text or "").strip()
#         if len(text) < MIN_TEXT_LENGTH and OCR_AVAILABLE:
#             try:
#                 pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72.0, 300 / 72.0), alpha=False)
#                 mode = 'RGB' if pix.n < 4 else 'RGBA'
#                 img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)  # type: ignore
#                 ocr_text = pytesseract.image_to_string(img, lang="fra+ara", config="--psm 6")
#                 text = normalise_text(ocr_text or "").strip()
#                 used_ocr = True
#             except Exception:
#                 # fallback: keep whatever we had
#                 pass
#         pages.append(PageText(page_index=i, text=text, used_ocr=used_ocr))
#     return pages


# def join_pages_with_offsets(
#     pages: List[PageText],
# ) -> Tuple[str, List[Tuple[int, str]], Dict[int, int]]:
#     """Concatenate page texts with synthetic delimiters and track offsets."""
#     parts: List[str] = []
#     line_offsets: List[Tuple[int, str]] = []
#     page_markers: Dict[int, int] = {}
#     cursor = 0
#     for i, page in enumerate(pages):
#         delim = f"\n\n<<<BO_PAGE_{i + 1}_DELIM>>>\n\n"
#         parts.append(delim)
#         cursor += len(delim)
#         page_markers[i + 1] = cursor
#         for ln in page.text.splitlines(True):
#             stripped = ln.rstrip("\n")
#             parts.append(ln)
#             line_offsets.append((cursor, stripped))
#             cursor += len(ln)
#         if not page.text.endswith("\n"):
#             parts.append("\n")
#             line_offsets.append((cursor, ""))
#             cursor += 1
#     full_text = "".join(parts)
#     return full_text, line_offsets, page_markers


# def find_section_boundaries(lines_with_offsets: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
#     """Return a list of (offset, banner_text) for each category banner."""
#     banners: List[Tuple[int, str]] = []
#     for off, line in lines_with_offsets:
#         m = SECTION_BANNER_RE.match(line.strip())
#         if m:
#             banners.append((off, m.group(1).upper()))
#     return banners


# def validate_header(full_text: str, pos: int) -> bool:
#     """Check if an enacting clause appears within the validation window."""
#     window = full_text[pos : pos + VALIDATION_LOOKAHEAD]
#     return bool(ENACTING_CLAUSE_RE.search(window))


# def find_document_starts(lines_with_offsets: List[Tuple[int, str]], full_text: str) -> List[int]:
#     """Identify candidate document start offsets."""
#     starts: List[int] = []
#     for off, line in lines_with_offsets:
#         stripped = line.strip()
#         if not stripped:
#             continue
#         if SECTION_BANNER_RE.match(stripped):
#             starts.append(off)
#             continue
#         if ENACTING_CLAUSE_RE.match(stripped):
#             continue
#         if DOC_HEADER_RE.match(stripped):
#             if validate_header(full_text, off):
#                 starts.append(off)
#             continue
#     return sorted(set(starts))


# def prune_document_starts(full_text: str, starts: List[int]) -> List[int]:
#     """Collapse candidate starts into one representative per document."""
#     pruned: List[int] = []
#     i = 0
#     n = len(starts)
#     while i < n:
#         s = starts[i]
#         # Skip banners
#         hdr_line = full_text[s : full_text.find("\n", s) if full_text.find("\n", s) != -1 else s + 200].strip()
#         if SECTION_BANNER_RE.match(hdr_line):
#             i += 1
#             continue
#         m = ENACTING_CLAUSE_RE.search(full_text, s)
#         if m:
#             clause_pos = m.start()
#             j = i + 1
#             while j < n and starts[j] < clause_pos:
#                 j += 1
#             candidates: List[int] = []
#             for k in range(i, j):
#                 off = starts[k]
#                 line = full_text[off : full_text.find("\n", off) if full_text.find("\n", off) != -1 else off + 200].strip()
#                 if SECTION_BANNER_RE.match(line) or ENACTING_CLAUSE_RE.match(line):
#                     continue
#                 candidates.append(off)
#             chosen: Optional[int] = None
#             for off in reversed(candidates):
#                 doc_t = detect_document_type(
#                     full_text[off : full_text.find("\n", off) if full_text.find("\n", off) != -1 else off + 200]
#                 )
#                 if doc_t not in {"dahir", "texte"}:
#                     chosen = off
#                     break
#             if chosen is None and candidates:
#                 chosen = candidates[-1]
#             if chosen is not None:
#                 pruned.append(chosen)
#             i = j
#         else:
#             pruned.append(s)
#             i += 1
#     return sorted(set(pruned))


# def cut_document_chunks(
#     full_text: str,
#     starts: List[int],
#     section_banners: List[Tuple[int, str]],
# ) -> List[Tuple[int, int]]:
#     """Produce (start, end) offsets for document chunks."""
#     chunks: List[Tuple[int, int]] = []
#     n = len(starts)
#     for i, s in enumerate(starts):
#         line = full_text[s : full_text.find("\n", s) if full_text.find("\n", s) != -1 else s + 200].strip()
#         if SECTION_BANNER_RE.match(line):
#             continue
#         e_default = starts[i + 1] if i + 1 < n else len(full_text)
#         window = full_text[s:e_default]
#         rel_banner_end: Optional[int] = None
#         for b_off, _ in section_banners:
#             if b_off > s and b_off < e_default:
#                 rel = b_off - s
#                 if rel > 80:
#                     rel_banner_end = rel
#                     break
#         rel_sig_end: Optional[int] = None
#         for m in re.finditer(
#             r"(?mi)^(?:Fait\s+à\s+[^\n,]+,|[A-ZÉÈ][^,\n]{1,40},)\s+le\s+[^\n]+$",
#             window,
#         ):
#             if m.start() > 80:
#                 rel_sig_end = m.start()
#                 break
#         candidates = [rel for rel in (rel_banner_end, rel_sig_end) if rel is not None]
#         e = s + min(candidates) if candidates else e_default
#         chunks.append((s, e))
#     return chunks


# def get_category_for_offset(
#     section_banners: List[Tuple[int, str]], offset: int
# ) -> Tuple[str, str]:
#     """Return (raw_category, normalised_category) for the offset."""
#     category_raw = ""
#     for off, cat in section_banners:
#         if off <= offset:
#             category_raw = cat
#         else:
#             break
#     if not category_raw:
#         return "", ""
#     cat_up = category_raw.upper()
#     if "GÉNÉRAUX" in cat_up or "GENERAUX" in cat_up:
#         return category_raw, "general_texts"
#     if "PARTICULIERS" in cat_up:
#         return category_raw, "particular_texts"
#     return "", ""


# def detect_document_type(header_line: str) -> str:
#     """Infer the document type from a header line."""
#     h_norm = unicodedata.normalize("NFKD", header_line or "").encode("ASCII", "ignore").decode().lower()
#     if "dahir" in h_norm:
#         return "dahir"
#     if "decret" in h_norm:
#         return "decret"
#     if "arrete" in h_norm:
#         return "arrete"
#     if "decision" in h_norm:
#         return "decision"
#     return "texte"


# # Regular expressions for extracting references and dates
# NUM_RE = re.compile(r"n[°ºo]\s*([0-9][0-9A-Za-z\-/.]*)", re.IGNORECASE)
# DATE_PAREN_RE = re.compile(r"\(([^)]+)\)")
# DATE_DU_RE = re.compile(r"\bdu\s+([0-9]{1,2}\s+[A-Za-zéèêàâîïôûùç\-]+\s+\d{4})", re.IGNORECASE)


# def extract_reference(header: str, after_lines: List[str]) -> Tuple[str, str]:
#     """Extract the reference number and date from the header and neighbouring lines."""
#     search_zone = header + "\n" + "\n".join(after_lines[:3])
#     num = ""
#     date_iso = ""
#     m = NUM_RE.search(search_zone)
#     if m:
#         num = m.group(1).strip()
#     m2 = DATE_PAREN_RE.search(search_zone)
#     if m2:
#         date_iso = parse_french_date_to_iso(m2.group(1).strip()) or ""
#     if not date_iso:
#         m3 = DATE_DU_RE.search(search_zone)
#         if m3:
#             date_iso = parse_french_date_to_iso(m3.group(1).strip()) or ""
#     return num, date_iso


# def extract_title(header: str, body_lines: List[str]) -> str:
#     """Derive a title either from the header remainder or from the body."""
#     m = re.search(r"n[°ºo]\s*[0-9][0-9A-Za-z\-/.]*\s*(.+)$", header, re.IGNORECASE)
#     if m:
#         candidate = m.group(1).strip().rstrip('.')
#         if len(candidate.split()) >= 3:
#             return candidate
#     for ln in body_lines[:20]:
#         s = ln.strip()
#         if not s:
#             continue
#         if re.match(r"(?i)^(Vu|Vus|Vu la|Vu le|Après|Apres|Considérant|Attendu)\b", s):
#             break
#         if ENACTING_CLAUSE_RE.match(s):
#             break
#         if len(s.split()) >= 3:
#             return s.rstrip('.')
#     return header.strip()


# def extract_promulgating_authority(body_lines: List[str]) -> str:
#     """Extract the promulgating authority (all caps lines)."""
#     for ln in body_lines[:15]:
#         s = ln.strip()
#         if not s:
#             continue
#         if len(s) > 6 and s.isupper() and not any(c.isdigit() for c in s):
#             return s.rstrip(",")
#     return ""


# def extract_preamble(body_lines: List[str]) -> List[Dict[str, str]]:
#     """Extract preamble entries (visa, adoption, recital)."""
#     items: List[Dict[str, str]] = []
#     for s in body_lines[:250]:
#         s_strip = s.strip()
#         if not s_strip:
#             continue
#         if ENACTING_CLAUSE_RE.match(s_strip):
#             break
#         if re.match(r"(?i)^(Vu|Vus|Vu la|Vu le)\b", s_strip):
#             ref = None
#             m = re.search(r"\b(loi|dahir|décret|decret|arr[êe]té|arrete|décision|decision)\s+n[°ºo]\s*([0-9][0-9A-Za-z\-/.]*)", s_strip, re.IGNORECASE)
#             if m:
#                 kind = m.group(1).lower().replace("décret", "decret").replace("arrêté", "arrete").replace("décision", "decision")
#                 num = m.group(2)
#                 ref = f"{kind.title()} n° {num}"
#             items.append({"type": "visa", "text": s_strip, "reference_to": ref})
#             continue
#         if re.match(r"(?i)^(Après|Apres|Après avis|Apres avis|Après délibération|Apres deliberation)\b", s_strip):
#             items.append({"type": "adoption", "text": s_strip})
#             continue
#         if re.match(r"(?i)^(Consid[ée]rant|Attendu)\b", s_strip):
#             items.append({"type": "recital", "text": s_strip})
#             continue
#     return items


# def extract_enacting_clause(body_lines: List[str]) -> str:
#     """Return the enacting clause (uppercase with colon) if found."""
#     for s in body_lines:
#         m = ENACTING_CLAUSE_RE.match(s.strip())
#         if m:
#             return m.group(1).upper() + " :"
#     return ""


# def is_noise_line(line: str) -> bool:
#     """Heuristic to discard page numbers and boilerplate from articles."""
#     s = line.strip()
#     if not s:
#         return True
#     if re.match(r"(?i)^(BULLETIN\s+OFFICIEL|ISSN|ROYAUME|Pages?|Page)\b", s):
#         return True
#     if re.match(r"^\d{1,4}$", s):
#         return True
#     if re.match(r"^\*{3,}$", s):
#         return True
#     return False


# def extract_articles_full(doc_text: str) -> List[Dict[str, Any]]:
#     """Extract a list of articles with all their content."""
#     articles: List[Dict[str, Any]] = []
#     headers = [m for m in ARTICLE_HEADER_RE.finditer(doc_text)]
#     if not headers:
#         return articles
#     positions = [m.start() for m in headers] + [len(doc_text)]
#     for idx, m in enumerate(headers):
#         start = m.start()
#         end = positions[idx + 1]
#         block = doc_text[start:end]
#         lines = [ln.strip() for ln in block.splitlines() if not is_noise_line(ln)]
#         if not lines:
#             continue
#         head = lines[0].rstrip('.')
#         body = lines[1:] if len(lines) > 1 else []
#         articles.append({"article_number": head, "paragraphs": body})
#     return articles


# def extract_signatures(doc_text: str) -> List[Dict[str, str]]:
#     """Extract signature information from the tail of the document."""
#     tail = doc_text[-3000:]
#     out: List[Dict[str, str]] = []
#     for m in re.finditer(
#         r"(?mi)^(?P<place>[A-ZÉÈ][A-Za-zÉÈÊÂÎÏÔÛÙÄËÖÜ'\- ]{1,40}),\s+le\s+(?P<date>[^\n]+)$",
#         tail,
#     ):
#         place = m.group("place").strip()
#         date_iso = parse_french_date_to_iso(m.group("date").strip()) or m.group("date").strip()
#         seg = tail[m.end() : m.end() + 200]
#         n = re.search(r"(?m)^[A-ZÉÈÊÂÎÏÔÛÙÄËÖÜ'\-\s]{3,80}\.?​*\s*$", seg)
#         if n:
#             name = " ".join(n.group(0).split())
#             out.append({"signed_at": place, "signed_on": date_iso, "signer_name": name, "signer_title": ""})
#     for m in re.finditer(
#         r"(?mi)^Fait\s+à\s+(?P<place>[^\n,]+),\s+le\s+(?P<date>[^\n]+)$",
#         tail,
#     ):
#         place = m.group("place").strip()
#         date_iso = parse_french_date_to_iso(m.group("date").strip()) or m.group("date").strip()
#         seg = tail[m.end() : m.end() + 200]
#         n = re.search(r"(?m)^[A-ZÉÈÊÂÎÏÔÛÙÄËÖÜ'\-\s]{3,80}\.?​*\s*$", seg)
#         if n:
#             name = " ".join(n.group(0).split())
#             out.append({"signed_at": place, "signed_on": date_iso, "signer_name": name, "signer_title": ""})
#     for m in re.finditer(
#         r"(?mi)^Pour\s+contreseing\s*[:：]?\s*([^\n]+)\n\s*([A-ZÉÈÊÂÎÏÔÛÙÄËÖÜ'\-\s]{3,80})",
#         tail,
#     ):
#         title = m.group(1).strip()
#         name = " ".join(m.group(2).split())
#         out.append({"signed_at": "", "signed_on": "", "signer_name": name, "signer_title": title})
#     seen = set()
#     uniq: List[Dict[str, str]] = []
#     for s in out:
#         key = (s.get("signer_name", ""), s.get("signer_title", ""), s.get("signed_on", ""))
#         if key not in seen:
#             seen.add(key)
#             uniq.append(s)
#     return uniq


# def extract_and_parse_pdf(pdf_path: str) -> Dict[str, Any]:
#     """Parse a single BO PDF into a structured JSON result."""
#     pages = extract_text_pages(pdf_path)
#     full_text, lines_with_offsets, page_markers = join_pages_with_offsets(pages)
#     section_banners = find_section_boundaries(lines_with_offsets)
#     starts = find_document_starts(lines_with_offsets, full_text)
#     doc_starts = prune_document_starts(full_text, starts)
#     if doc_starts:
#         first_doc = doc_starts[0]
#         section_banners = [(off, cat) for off, cat in section_banners if off >= first_doc]
#     chunks = cut_document_chunks(full_text, doc_starts, section_banners)
#     bulletin_number = ""
#     greg, hijri = "", ""
#     head = "\n".join([line for _, line in lines_with_offsets[:300]])
#     m_bn = re.search(r"\bN[°ºo]\s*([0-9]{3,6})\b", head, re.IGNORECASE)
#     if m_bn:
#         bulletin_number = m_bn.group(1)
#     m_date = re.search(r"([0-9]{1,2}\s+[^()\d\n]{3,30}\s+[0-9]{3,4})\s*\(\s*([^)]+)\)", head, re.IGNORECASE)
#     if m_date:
#         hijri = m_date.group(1).strip()
#         greg = parse_french_date_to_iso(m_date.group(2).strip()) or ""
#     else:
#         m2 = re.search(r"\(\s*([0-9]{1,2}\s+[A-Za-zéèêàâîïôûùç\-]+\s+[0-9]{4})\s*\)", head)
#         if m2:
#             greg = parse_french_date_to_iso(m2.group(1).strip()) or ""
#     def offset_to_page(off: int) -> int:
#         if not page_markers:
#             return 0
#         best_page = 0
#         best_offset = -1
#         for p, o in page_markers.items():
#             if o <= off and o > best_offset:
#                 best_offset = o
#                 best_page = p
#         return best_page
#     content: List[Dict[str, Any]] = []
#     for idx, (s, e) in enumerate(chunks):
#         chunk_text = full_text[s:e]
#         lines = chunk_text.splitlines()
#         if not lines:
#             continue
#         header = lines[0].strip()
#         if SECTION_BANNER_RE.match(header):
#             continue
#         cat_raw, cat_norm = get_category_for_offset(section_banners, s)
#         if not cat_raw or not cat_norm:
#             cat_raw, cat_norm = "TEXTES GENERAUX", "general_texts"
#         doc_type = detect_document_type(header)
#         if doc_type == "texte":
#             for probe in lines[:6]:
#                 dt = detect_document_type(probe.strip())
#                 if dt != "texte":
#                     doc_type = dt
#                     break
#         num, date_iso = extract_reference(header, lines[1:])
#         title = extract_title(header, lines[1:])
#         authority = extract_promulgating_authority(lines[1:])
#         preamble = extract_preamble(lines[1:])
#         enact_clause = extract_enacting_clause(lines[1:])
#         articles = extract_articles_full(chunk_text)
#         signatures = extract_signatures(chunk_text)
#         p_start = offset_to_page(s)
#         p_end = offset_to_page(e)
#         if p_end < p_start:
#             p_end = p_start
#         if not articles and not enact_clause and doc_type == "texte":
#             continue
#         entry_id = f"BO-{bulletin_number or 'XXXX'}-P{p_start}-{idx + 1}"
#         entry = {
#             "entry_id": entry_id,
#             "category_raw": cat_raw,
#             "category_normalized": cat_norm,
#             "sub_category": None,
#             "document_type": doc_type,
#             "title": title or header,
#             "reference": {"type": doc_type, "number": num, "date": date_iso},
#             "pages": {"start": p_start, "end": p_end},
#             "legal_references": {
#                 "based_on": [],
#                 "modifies": [],
#                 "abrogates": [],
#                 "effective_date": "",
#             },
#             "document_structure": {
#                 "promulgating_authority": authority,
#                 "preamble": preamble,
#                 "enacting_clause": enact_clause,
#                 "chapters": [
#                     {
#                         "chapter_number": "Chapitre",
#                         "chapter_title": None,
#                         "articles": articles,
#                     }
#                 ],
#                 "signatures": signatures,
#                 "annexes": [],
#             },
#         }
#         content.append(entry)
#     content.sort(key=lambda d: (d["pages"]["start"], d["entry_id"]))
#     return {
#         "bulletin_number": bulletin_number,
#         "publication_date": {"gregorian": greg, "hijri": hijri},
#         "source_file": os.path.basename(pdf_path),
#         "content": content,
#     }
import fitz  # PyMuPDF
import pandas as pd
import re
import os
import io
import numpy as np
from PIL import Image
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TableValidationConfig:
    """Configuration pour la validation des tableaux"""
    max_columns: int = 12
    min_rows: int = 2
    max_null_percentage: float = 0.4
    min_content_ratio: float = 0.3


class PDFParser:
    """
    Parser PDF de production robuste avec LayoutLMv3/LayoutXLM.
    This is the version with the correct __init__ signature.
    """
    POLLUTANT_MAPPING = {
        'SO2': ['dioxyde de soufre', 'so2', 'sulphur dioxide', 'anhydride sulfureux'],
        'NOx': ["oxydes d'azote", 'nox', 'nitrogen oxides', "monoxyde d'azote", "dioxyde d'azote"],
        'COV': ['composés organiques volatils', 'cov', 'volatile organic compounds', 'voc'],
        'PM10': ['particules pm10', 'pm10', 'particulate matter 10', 'poussières pm10'],
        'PM2.5': ['particules pm2.5', 'pm2.5', 'particules fines'],
        'Hg': ['mercure', 'mercury', 'hg'],
        'Pb': ['plomb', 'lead', 'pb'],
        'Cd': ['cadmium', 'cd'],
        'O3': ['ozone', 'o3'],
        'CO': ['monoxyde de carbone', 'co', 'carbon monoxide'],
        'Benzène': ['benzène', 'benzene', 'c6h6'],
        'H2S': ["sulfure d'hydrogène", 'h2s', 'hydrogen sulfide'],
        'NH3': ['ammoniac', 'nh3', 'ammonia'],
        'Fluorures': ['fluorures', 'fluorides', 'hf'],
        'Chlorures': ['chlorures', 'chlorides', 'hcl']
    }

    def __init__(
        self,
        pdf_path: str,
        config: Optional[TableValidationConfig] = None,
        layout_model_name: Optional[str] = None,
        layout_device: Optional[str] = None
    ):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Fichier introuvable: {pdf_path}")

        self.pdf_path = pdf_path
        self.filename = os.path.basename(pdf_path)
        self.doc = fitz.open(pdf_path)
        self.config = config or TableValidationConfig()

        # LayoutLM model (lazy-loaded)
        self.layout_model = None
        self.layout_processor = None
        self.layout_id2label: Dict[int, str] = {}
        self.layout_model_name = layout_model_name
        self.layout_device = layout_device
        self._torch = None
        self._device = "cpu"

        # Analyse préliminaire
        self.doc_analysis = self._analyze_document_structure()
        self.doc_type = self._classify_document()
        self.metadata = self._extract_metadata()

    # ---------------- Core analysis helpers ----------------

    def _analyze_document_structure(self) -> Dict[str, Any]:
        """Analyse simplifiée de la structure du document."""
        analysis = {
            'total_pages': len(self.doc),
            'scanned_pages': 0, 'text_pages': 0, 'table_pages': 0,
            'mixed_pages': 0, 'avg_text_density': 0.0, 'has_images': False
        }
        text_densities = []
        for page_num in range(min(5, len(self.doc))):
            page = self.doc.load_page(page_num)
            text = page.get_text()
            area = page.rect.width * page.rect.height or 1
            text_density = len(text.strip()) / area * 10000
            text_densities.append(text_density)
            has_images = bool(page.get_images())
            if has_images:
                analysis['has_images'] = True
            if len(text.strip()) < 200 and has_images:
                analysis['scanned_pages'] += 1
            elif text_density > 5:
                analysis['text_pages'] += 1
            else:
                analysis['mixed_pages'] += 1
        analysis['avg_text_density'] = float(np.mean(text_densities)) if text_densities else 0.0
        return analysis

    def _classify_document(self) -> str:
        filename_lower = self.filename.lower()
        filename_patterns = [
            (r'valeurs?.*limites?.*générales?.*atmosphérique', 'vlg_atmospherique'),
            (r'valeurs?.*limites?.*générales?.*liquide', 'vlg_liquide'),
            (r'valeurs?.*limites?.*sectorielles?.*ciment', 'vls_ciment'),
            (r'valeurs?.*limites?.*sectorielles?.*céramique', 'vls_ceramique'),
            (r'valeurs?.*limites?.*sectorielles?', 'vls_autre'),
            (r'normes?.*qualité.*air', 'normes_air'),
            (r'normes?.*qualité.*eau', 'normes_eau'),
            (r'seuils?.*information.*alerte', 'seuils'),
            (r'décret|decret', 'decret'),
            (r'lettre.*royale', 'lettre_royale'),
            (r'irrigation', 'irrigation')
        ]
        for pattern, doc_type in filename_patterns:
            if re.search(pattern, filename_lower):
                return doc_type
        content_keywords = self._extract_content_keywords()
        if 'valeur' in content_keywords and 'limite' in content_keywords:
            if 'atmosphérique' in content_keywords or 'air' in content_keywords:
                return 'vlg_atmospherique'
            elif 'liquide' in content_keywords or 'eau' in content_keywords:
                return 'vlg_liquide'
        return 'autre'

    def _extract_content_keywords(self) -> List[str]:
        keywords = []
        for page_num in range(min(3, len(self.doc))):
            page = self.doc.load_page(page_num)
            text = page.get_text().lower()
            key_terms = [
                'valeur', 'limite', 'norme', 'seuil', 'décret', 'atmosphérique', 'air',
                'liquide', 'eau', 'ciment', 'céramique', 'textile', 'sucre',
                'pollution', 'émission', 'rejet', 'qualité'
            ]
            for term in key_terms:
                if term in text:
                    keywords.append(term)
        return keywords

    def _extract_metadata(self) -> Dict[str, Any]:
        meta = self.doc.metadata
        return {
            "title": meta.get("title", ""),
            "author": meta.get("author", ""),
            "creation_date": self._parse_pdf_date(meta.get("creationDate")),
            "modification_date": self._parse_pdf_date(meta.get("modDate")),
            "keywords": meta.get("keywords", ""),
            "pages": len(self.doc),
            "file_size": os.path.getsize(self.pdf_path),
            "analysis": self.doc_analysis
        }

    def _parse_pdf_date(self, date_str: Optional[str]) -> str:
        if not date_str:
            return ""
        try:
            if date_str.startswith("D:") and len(date_str) >= 10:
                return f"{date_str[2:6]}-{date_str[6:8]}-{date_str[8:10]}"
        except Exception:
            pass
        return date_str

    def _extract_structured_text(self, page) -> Dict[str, Any]:
        blocks = page.get_text("blocks")
        structured_content = {"title_text": "", "body_text": "", "headers": [], "paragraphs": []}

        def looks_like_header(t: str) -> bool:
            t_norm = t.strip()
            if not t_norm or len(t_norm) > 120:
                return False
            cap_ratio = sum(1 for c in t_norm if c.isupper()) / max(1, len(t_norm))
            keywords = ['article', 'chapitre', 'section', 'annexe', 'tableau']
            if cap_ratio > 0.4:
                return True
            return any(k in t_norm.lower() for k in keywords)

        for block in blocks:
            if len(block) >= 5:
                text = block[4].strip()
                if not text:
                    continue
                if looks_like_header(text) and len(text) < 100:
                    structured_content["headers"].append(text)
                    if not structured_content["title_text"]:
                        structured_content["title_text"] = text
                elif len(text) > 30:
                    structured_content["paragraphs"].append(text)
                    structured_content["body_text"] += text + "\n"
        return structured_content

    # ---------------- Model loading and layout detection ----------------

    def _ensure_layout_model(self) -> bool:
        # only try once per document
        if getattr(self, "_layout_load_attempted", False):
            return self.layout_model is not None
        self._layout_load_attempted = True

        try:
            # Lazy import already present at module-level, but keep for isolation
            from transformers import AutoProcessor, AutoModelForTokenClassification  # type: ignore
            import torch  # type: ignore
            self._torch = torch

            model_name = (
                self.layout_model_name
                or os.getenv("LAYOUTLM_MODEL_NAME")
                or "HYPJUDY/layoutlmv3-base-finetuned-publaynet"
            )
            hf_token = (
                getattr(self, "layout_hf_token", None)
                or os.getenv("HUGGINGFACE_TOKEN")
                or os.getenv("HF_TOKEN")
            )
            requested = (self.layout_device or os.getenv("LAYOUT_DEVICE", "cpu")).lower()

            # Choose processor source:
            # - If local dir lacks preprocessor_config.json, use a safe base processor.
            processor_id = model_name
            if os.path.isdir(model_name):
                has_preproc = os.path.isfile(os.path.join(model_name, "preprocessor_config.json"))
                if not has_preproc:
                    processor_id = "microsoft/layoutlmv3-base"

            # Load processor with OCR
            proc_kwargs = {"apply_ocr": True}
            try:
                self.layout_processor = AutoProcessor.from_pretrained(
                    processor_id, token=hf_token, **proc_kwargs
                ) if hf_token else AutoProcessor.from_pretrained(processor_id, **proc_kwargs)
            except TypeError:
                self.layout_processor = AutoProcessor.from_pretrained(
                    processor_id, use_auth_token=hf_token, **proc_kwargs
                )
            except Exception as e_proc:
                base_proc_id = "microsoft/layoutlmv3-base"
                logger.warning(
                    "Processor load failed for %s (%s). Falling back to %s.",
                    processor_id, e_proc, base_proc_id
                )
                self.layout_processor = AutoProcessor.from_pretrained(
                    base_proc_id, token=hf_token, **proc_kwargs
                ) if hf_token else AutoProcessor.from_pretrained(base_proc_id, **proc_kwargs)

            # Load token-classification model
            model_candidates = [
                model_name,
                os.getenv("LAYOUTLM_FALLBACK_REMOTE", "HYPJUDY/layoutlmv3-base-finetuned-publaynet"),
                "nielsr/layoutlmv3-finetuned-doclaynet",
            ]
            last_err = None
            self.layout_model = None
            for cand in model_candidates:
                try:
                    if hf_token:
                        self.layout_model = AutoModelForTokenClassification.from_pretrained(cand, token=hf_token)
                    else:
                        self.layout_model = AutoModelForTokenClassification.from_pretrained(cand)
                    model_name = cand
                    break
                except TypeError:
                    try:
                        self.layout_model = AutoModelForTokenClassification.from_pretrained(
                            cand, use_auth_token=hf_token
                        )
                        model_name = cand
                        break
                    except Exception as e_model:
                        last_err = e_model
                except Exception as e_model:
                    last_err = e_model

            if self.layout_model is None:
                raise last_err or RuntimeError("Aucun modèle LayoutLM valide n'a pu être chargé.")

            # Device selection
            if requested.startswith("cuda") and torch.cuda.is_available():
                self._device = requested
            elif requested in {"mps", "mps:0"} and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
            self.layout_model.to(self._device)

            # id2label
            self.layout_id2label = dict(getattr(self.layout_model.config, "id2label", {}))
            logger.info(
                "Layout model loaded: %s on %s. Labels: %s",
                model_name, self._device, list(self.layout_id2label.values())
            )
            self._layout_max_length = int(os.getenv("LAYOUT_MAX_TOKENS", "512"))
            self._layout_stride = int(os.getenv("LAYOUT_STRIDE", "128"))
            return True

        except Exception as e:
            self._layout_last_error = str(e)
            logger.warning(
                "Impossible de charger LayoutLMv3/LayoutXLM (%s) - fallback sur extraction basique.",
                e
            )
            self.layout_model = None
            self.layout_processor = None
            self.layout_id2label = {}
            self._torch = None
            self._device = "cpu"
            return False

    def _normalize_layout_label(self, raw_label: str) -> str:
        rl = (raw_label or "").strip().lower()
        if '-' in rl and rl[0] in {'b', 'i', 's', 'e'}:
            rl = rl.split('-', 1)[1]
        if any(k in rl for k in ['table', 'tab']):
            return 'table'
        if any(k in rl for k in ['title', 'header', 'heading', 'section', 'h1', 'h2', 'h3']):
            return 'title'
        return 'paragraph'

    def _run_layoutlm_layout_detection(self, page_image: Image.Image) -> List[Dict[str, Any]]:
        if not self._ensure_layout_model():
            w, h = page_image.size
            return [{'box': [0, 0, w, h], 'label': 'paragraph', 'confidence': 0.4, 'text': ""}]
        torch = self._torch
        assert torch is not None
        try:
            processor = self.layout_processor
            model = self.layout_model
            assert processor is not None and model is not None

            encoding = processor(images=page_image, return_tensors="pt")
            encoding = {k: (v.to(self._device) if hasattr(v, "to") else v) for k, v in encoding.items()}
            with torch.no_grad():
                outputs = model(**encoding)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            pred_ids = probs.argmax(axis=-1)

            input_ids = encoding.get("input_ids")
            if input_ids is None:
                # Some processors may not expose input_ids in image-only mode; fallback
                w, h = page_image.size
                return [{'box': [0, 0, w, h], 'label': 'paragraph', 'confidence': 0.4, 'text': ""}]
            input_ids = input_ids[0].cpu().numpy()
            tokens = processor.tokenizer.convert_ids_to_tokens(input_ids.tolist())
            bboxes = encoding["bbox"][0].cpu().numpy()
            attn = encoding.get("attention_mask")
            if attn is not None:
                attn = attn[0].cpu().numpy()
            img_w, img_h = page_image.size

            token_items: List[Dict[str, Any]] = []
            for i in range(len(tokens)):
                if attn is not None and attn[i] == 0:
                    continue
                box = bboxes[i].tolist() if i < len(bboxes) else [0, 0, 0, 0]
                if box == [0, 0, 0, 0]:
                    continue
                raw_label = self.layout_id2label.get(int(pred_ids[i]), str(int(pred_ids[i])))
                label = self._normalize_layout_label(raw_label)
                score = float(probs[i, pred_ids[i]])
                x0 = int(round(box[0] / 1000.0 * img_w))
                y0 = int(round(box[1] / 1000.0 * img_h))
                x1 = int(round(box[2] / 1000.0 * img_w))
                y1 = int(round(box[3] / 1000.0 * img_h))
                x0 = max(0, min(x0, img_w - 1))
                y0 = max(0, min(y0, img_h - 1))
                x1 = max(x0 + 1, min(x1, img_w))
                y1 = max(y0 + 1, min(y1, img_h))
                tok = tokens[i]
                token_items.append({
                    "text": tok, "bbox": [x0, y0, x1, y1],
                    "label": label, "score": score
                })
            blocks = self._group_tokens_into_blocks(token_items, page_image.size)
            return blocks
        except Exception as e:
            logger.warning(f"Échec LayoutLM inference: {e}")
            w, h = page_image.size
            return [{'box': [0, 0, w, h], 'label': 'paragraph', 'confidence': 0.4, 'text': ""}]

    def _group_tokens_into_blocks(self, tokens: List[Dict[str, Any]], img_size: tuple) -> List[Dict[str, Any]]:
        if not tokens:
            return []
        heights = [t["bbox"][3] - t["bbox"][1] for t in tokens]
        median_h = float(np.median([h for h in heights if h > 0])) if heights else 10.0
        line_thr = max(4.0, median_h * 0.7)
        cluster_thr = max(6.0, median_h * 1.2)

        by_label: Dict[str, List[Dict[str, Any]]] = {"title": [], "paragraph": [], "table": []}
        for t in tokens:
            lbl = t["label"] if t["label"] in by_label else "paragraph"
            by_label[lbl].append(t)

        blocks: List[Dict[str, Any]] = []
        for lbl, toks in by_label.items():
            if not toks:
                continue
            clusters: List[Dict[str, Any]] = []
            for t in sorted(toks, key=lambda x: (x["bbox"][1], x["bbox"][0])):
                placed = False
                for cl in clusters:
                    if self._token_close_to_cluster(t, cl["bbox"], cluster_thr):
                        cl["tokens"].append(t)
                        cl["bbox"] = self._merge_boxes(cl["bbox"], t["bbox"])
                        placed = True
                        break
                if not placed:
                    clusters.append({"tokens": [t], "bbox": t["bbox"][:]})
            for cl in clusters:
                cl_tokens = sorted(cl["tokens"], key=lambda x: (self._y_center(x["bbox"]), x["bbox"][0]))
                text = self._compose_block_text(cl_tokens, lbl, line_thr, table_mode=(lbl == "table"))
                conf = float(np.mean([tt["score"] for tt in cl_tokens])) if cl_tokens else 0.0
                blocks.append({"box": cl["bbox"], "label": lbl, "confidence": conf, "text": text})
        blocks = self._sort_detections_reading_order(blocks)
        return blocks

    def _merge_boxes(self, a: List[int], b: List[int]) -> List[int]:
        return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]

    def _y_center(self, box: List[int]) -> float:
        return (box[1] + box[3]) / 2.0

    def _token_close_to_cluster(self, token: Dict[str, Any], cl_box: List[int], thr: float) -> bool:
        tb = token["bbox"]
        if not (tb[2] < cl_box[0] or tb[0] > cl_box[2] or tb[3] < cl_box[1] or tb[1] > cl_box[3]):
            return True
        vdist = max(0, max(cl_box[1] - tb[3], tb[1] - cl_box[3]))
        hdist = max(0, max(cl_box[0] - tb[2], tb[0] - cl_box[2]))
        return vdist <= thr and hdist <= thr * 2

    def _normalize_token_text(self, tok: str):
        if not tok:
            return "", False
        if tok.startswith("##"):
            return tok[2:], True
        if tok.startswith(" "):
            return tok[1:], False
        return tok, False

    def _compose_block_text(self, tokens: List[Dict[str, Any]], label: str, line_thr: float, table_mode: bool) -> str:
        if not tokens:
            return ""
        lines: List[List[Dict[str, Any]]] = []
        current_line: List[Dict[str, Any]] = []
        current_y: Optional[float] = None
        for t in tokens:
            y = self._y_center(t["bbox"])
            if current_line and current_y is not None and abs(y - current_y) > line_thr:
                lines.append(sorted(current_line, key=lambda x: x["bbox"][0]))
                current_line = [t]
                current_y = y
            else:
                if not current_line:
                    current_y = y
                current_line.append(t)
        if current_line:
            lines.append(sorted(current_line, key=lambda x: x["bbox"][0]))

        out_lines: List[str] = []
        for ln in lines:
            if not ln:
                continue
            widths = [tt["bbox"][2] - tt["bbox"][0] for tt in ln]
            median_w = float(np.median(widths)) if widths else 10.0
            parts: List[str] = []
            prev_box: Optional[List[int]] = None
            prev_join_no_space = False
            for tt in ln:
                text_raw = tt["text"]
                text_norm, join_no_space = self._normalize_token_text(text_raw)
                if not text_norm:
                    continue
                delim = ""
                if parts:
                    if prev_join_no_space or join_no_space:
                        delim = ""
                    else:
                        if table_mode and prev_box is not None:
                            gap = tt["bbox"][0] - prev_box[2]
                            if gap > max(12, 1.5 * median_w):
                                delim = " | "
                            else:
                                delim = " "
                        else:
                            delim = " "
                parts.append(delim + text_norm)
                prev_box = tt["bbox"]
                prev_join_no_space = join_no_space
            out_lines.append("".join(parts))
        return "\n".join(out_lines).strip()

    def _parse_table_from_text(self, text: str) -> Optional[pd.DataFrame]:
        if not text or len(text.strip()) < 10:
            return None
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if len(lines) < 2:
            return None
        count_pipe = sum(1 for ln in lines if '|' in ln)
        count_tab = sum(1 for ln in lines if '\t' in ln)
        sep = r'\|' if count_pipe >= max(2, len(lines) // 3) else r'\t' if count_tab >= max(2, len(lines) // 3) else r'\s{2,}'
        token_rows: List[List[str]] = []
        for ln in lines:
            if sep == r'\|':
                ln = re.sub(r'^\|', '', ln)
                ln = re.sub(r'\|$', '', ln)
            cols = [c.strip() for c in re.split(sep, ln) if c.strip() != ""]
            if len(cols) == 0:
                continue
            token_rows.append(cols)
        if len(token_rows) < 2:
            return None
        max_cols = max(len(r) for r in token_rows)
        if max_cols == 1:
            return None
        norm_rows = [r + [""] * (max_cols - len(r)) if len(r) < max_cols else r for r in token_rows]
        header_candidates = norm_rows[0]
        num_in_first = sum(1 for v in header_candidates if re.search(r'\d', v))
        header_is_texty = num_in_first <= len(header_candidates) / 3.0
        try:
            if header_is_texty:
                df = pd.DataFrame(norm_rows[1:], columns=[self._normalize_header_cell(h) for h in header_candidates])
            else:
                df = pd.DataFrame(norm_rows, columns=[f"col_{i+1}" for i in range(max_cols)])
        except Exception:
            df = pd.DataFrame(norm_rows, columns=[f"col_{i+1}" for i in range(max_cols)])
        return df

    def _normalize_header_cell(self, s: str) -> str:
        return re.sub(r'\s+', ' ', (s or '').strip()) or "col"

    def _process_merged_cells(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        if len(df_copy.columns) > 0:
            df_copy.iloc[:, 0] = df_copy.iloc[:, 0].ffill()
        if self._is_standards_table(df_copy):
            df_copy = self._process_standards_table(df_copy)
        return df_copy

    def _is_standards_table(self, df: pd.DataFrame) -> bool:
        keywords = ['polluant', 'limite', 'valeur', 'unité', 'µg/m³', 'mg/l']
        text_content = ' '.join([str(col) for col in df.columns]).lower()
        return any(keyword in text_content for keyword in keywords)

    def _process_standards_table(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = df.copy()
        for col_idx in range(len(df_processed.columns)):
            df_processed.iloc[:, col_idx] = df_processed.iloc[:, col_idx].ffill()
        return df_processed

    def _validate_table_strict(self, df: pd.DataFrame, page) -> bool:
        if len(df.columns) > self.config.max_columns:
            return False
        if len(df) < self.config.min_rows:
            return False
        null_ratio = df.isnull().sum().sum() / max(1, df.size)
        if null_ratio > self.config.max_null_percentage:
            return False
        text_cells = sum(1 for v in df.values.ravel() if pd.notna(v) and str(v).strip())
        content_ratio = text_cells / max(1, df.size)
        if content_ratio < self.config.min_content_ratio:
            return False
        if self._has_incoherent_columns(df):
            return False
        return True

    def _has_incoherent_columns(self, df: pd.DataFrame) -> bool:
        for col in df.columns:
            values = df[col].dropna().astype(str)
            if len(values) > 0:
                short_values = sum(1 for v in values if len(v.strip()) < 3)
                if short_values / len(values) > 0.8:
                    return True
        return False

    def _calculate_table_confidence(self, df: pd.DataFrame) -> float:
        score = 1.0
        null_ratio = df.isnull().sum().sum() / max(1, df.size)
        score -= null_ratio * 0.5
        for col in df.columns:
            values = df[col].dropna().astype(str)
            if len(values) > 0:
                numeric_count = sum(1 for v in values if re.fullmatch(r'[+-]?[\d\s.,]+', v))
                if numeric_count / len(values) > 0.7:
                    score += 0.1
        return max(0.0, min(1.0, score))

    def _sort_detections_reading_order(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(detections, key=lambda d: (d['box'][1] // 25, d['box'][0]))

    # ---------------- NLP-ish extraction helpers ----------------

    def _detect_pollutants_enhanced(self, text: str) -> Dict[str, Dict[str, Any]]:
        found_pollutants = {}
        text_lower = text.lower()
        for code, names in self.POLLUTANT_MAPPING.items():
            for name in names:
                if name in text_lower:
                    context = self._extract_context(text_lower, name)
                    found_pollutants[code] = {
                        "name": name,
                        "matched_term": name,
                        "context": context,
                        "has_limit_value": self._has_associated_limit(context)
                    }
                    break
        return found_pollutants

    def _extract_context(self, text: str, term: str, window: int = 100) -> str:
        pos = text.find(term)
        if pos == -1:
            return ""
        start = max(0, pos - window)
        end = min(len(text), pos + len(term) + window)
        return text[start:end].strip()

    def _has_associated_limit(self, context: str) -> bool:
        limit_pattern = r'\d+[\d\s,.]*\s*(µg/m³|mg/m³|mg/l|µg/l|ppm|ppb)'
        return bool(re.search(limit_pattern, context))

    def _extract_limit_values_enhanced(self, text: str) -> List[Dict[str, Any]]:
        patterns = [
            r'(\d+(?:[,\.\s]\d+)*)\s*(µg/m³|mg/m³|mg/l|µg/l|ng/m³|ppm|ppb|°C|%)',
            r'(\d+(?:[,\.\s]\d+)*)\s*(microgrammes?|milligrammes?|nanogrammes?)',
            r'(\d+(?:[,\.\s]\d+)*)\s*(?:µg|mg|ng|ppm|ppb)'
        ]
        found_values = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                value_str = match[0].replace(' ', '').replace(',', '.')
                unit = match[1] if len(match) > 1 else 'unité_non_spécifiée'
                try:
                    numeric_value = float(value_str)
                    found_values.append({
                        "raw_value": match[0],
                        "numeric_value": numeric_value,
                        "unit": unit,
                        "context": self._extract_context(text, match[0])
                    })
                except ValueError:
                    continue
        return found_values

    # ---------------- Page processing and orchestration ----------------

    def _process_page_with_strategy(self, page_num: int) -> Dict[str, Any]:
        page = self.doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes())).convert("RGB")
        page_data = {
            "page_number": page_num + 1,
            "dimensions": {"width": page.rect.width, "height": page.rect.height},
            "strategy_used": "layoutlm_layout",
            "content": {},
            "confidence": 0.0
        }

        detections = self._run_layoutlm_layout_detection(img)

        elements: List[Dict[str, Any]] = []
        headers: List[str] = []
        paragraphs: List[str] = []
        tables: List[Dict[str, Any]] = []

        for det in detections:
            x0, y0, x1, y1 = det['box']
            label = det['label']
            text = det.get('text', '') or ''
            conf = float(det.get('confidence', 0.0))

            if label == 'table':
                df = self._parse_table_from_text(text)
                if df is not None and self._validate_table_strict(df, page):
                    df_clean = self._process_merged_cells(df)
                    table_conf = self._calculate_table_confidence(df_clean)
                    tables.append({
                        "table_id": len(tables),
                        "bbox": [x0, y0, x1, y1],
                        "header": df_clean.columns.tolist(),
                        "rows": df_clean.fillna("").values.tolist(),
                        "shape": df_clean.shape,
                        "confidence": float(max(0.0, min(1.0, (conf + table_conf) / 2.0)))
                    })
                elements.append({"label": "table", "bbox": [x0, y0, x1, y1], "confidence": conf, "text": text})
            elif label == 'title':
                if text:
                    headers.append(text)
                elements.append({"label": "title", "bbox": [x0, y0, x1, y1], "confidence": conf, "text": text})
            else:
                if text:
                    paragraphs.append(text)
                elements.append({"label": "paragraph", "bbox": [x0, y0, x1, y1], "confidence": conf, "text": text})

        if not elements or (len(''.join([e.get('text', '') for e in elements]).strip()) == 0):
            logger.debug("LayoutLM n'a pas fourni de contenu exploitable - fallback bloc texte.")
            text_content = self._extract_structured_text(page)
            full_text_fb = text_content.get("body_text", "").strip()
            page_data["content"] = {
                "elements": [],
                "headers": text_content.get("headers", []),
                "paragraphs": text_content.get("paragraphs", []),
                "tables": [],
                "text": full_text_fb,
                "pollutants": self._detect_pollutants_enhanced(full_text_fb),
                "limit_values": self._extract_limit_values_enhanced(full_text_fb)
            }
            page_data["confidence"] = 0.55 if full_text_fb else 0.1
            return page_data

        aggregated_text_parts = [
            ("\n".join(headers) if headers else ""),
            ("\n".join(paragraphs) if paragraphs else "")
        ]
        aggregated_text = "\n".join(aggregated_text_parts).strip()
        page_data["content"] = {
            "elements": elements,
            "headers": headers,
            "paragraphs": paragraphs,
            "tables": tables,
            "text": aggregated_text,
            "pollutants": self._detect_pollutants_enhanced(aggregated_text),
            "limit_values": self._extract_limit_values_enhanced(aggregated_text)
        }
        det_conf_avg = float(np.mean([e.get("confidence", 0.0) for e in elements])) if elements else 0.0
        table_conf_avg = float(np.mean([t.get("confidence", 0.0) for t in tables])) if tables else det_conf_avg
        richness_bonus = 0.05 if (headers or paragraphs) and tables else 0.0
        page_conf = max(0.0, min(1.0, 0.6 * det_conf_avg + 0.4 * table_conf_avg + richness_bonus))
        page_data["confidence"] = page_conf
        return page_data

    def parse(self) -> Dict[str, Any]:
        logger.info(f"Début analyse: {self.filename}")
        doc_data = {
            "metadata": self.metadata,
            "document_type": self.doc_type,
            "filename": self.filename,
            "analysis_summary": {
                "total_pages": len(self.doc),
                "strategies_used": {},
                "avg_confidence": 0.0,
                "processing_errors": 0
            },
            "pages": []
        }
        confidences = []
        strategies: Dict[str, int] = {}
        errors = 0
        for page_num in range(len(self.doc)):
            try:
                page_data = self._process_page_with_strategy(page_num)
                doc_data["pages"].append(page_data)
                confidences.append(page_data.get("confidence", 0.0))
                strategy = page_data.get("strategy_used", "layoutlm_layout")
                strategies[strategy] = strategies.get(strategy, 0) + 1
            except Exception as e:
                logger.error(f"Erreur page {page_num + 1}: {str(e)}")
                doc_data["pages"].append({
                    "page_number": page_num + 1,
                    "error": str(e),
                    "strategy_used": "error",
                    "confidence": 0.0
                })
                errors += 1

        doc_data["analysis_summary"].update({
            "strategies_used": strategies,
            "avg_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "processing_errors": errors
        })
        doc_data["global_analysis"] = self._analyze_document_globally(doc_data)
        logger.info(f"Analyse terminée - Confiance: {doc_data['analysis_summary']['avg_confidence']:.2f}")
        return doc_data

    def _analyze_document_globally(self, doc_data: Dict[str, Any]) -> Dict[str, Any]:
        global_pollutants: Dict[str, Dict[str, Any]] = {}
        global_limits: List[Dict[str, Any]] = []
        for page in doc_data["pages"]:
            if "content" in page and isinstance(page["content"], dict):
                if "pollutants" in page["content"]:
                    for code, info in page["content"]["pollutants"].items():
                        if code not in global_pollutants:
                            global_pollutants[code] = {
                                "name": info["name"],
                                "pages": [page["page_number"]],
                                "contexts": [info.get("context", "")],
                                "has_limits": info.get("has_limit_value", False)
                            }
                        else:
                            global_pollutants[code]["pages"].append(page["page_number"])
                            global_pollutants[code]["contexts"].append(info.get("context", ""))
                if "limit_values" in page["content"]:
                    for limit in page["content"]["limit_values"]:
                        limit["page"] = page["page_number"]
                        global_limits.append(limit)
        return {
            "pollutants_summary": global_pollutants,
            "limit_values_summary": global_limits,
            "document_quality": self._assess_document_quality(doc_data),
            "extraction_recommendations": self._generate_recommendations(doc_data)
        }

    def _assess_document_quality(self, doc_data: Dict[str, Any]) -> Dict[str, Any]:
        total_pages = len(doc_data["pages"])
        successful_pages = sum(1 for p in doc_data["pages"] if p.get("confidence", 0) > 0.5)
        quality_score = successful_pages / total_pages if total_pages > 0 else 0
        return {
            "overall_score": quality_score,
            "successful_pages": successful_pages,
            "total_pages": total_pages,
            "quality_level": "high" if quality_score > 0.8 else "medium" if quality_score > 0.5 else "low"
        }

    def _generate_recommendations(self, doc_data: Dict[str, Any]) -> List[str]:
        recommendations = []
        strategies = doc_data["analysis_summary"]["strategies_used"]
        errors = doc_data["analysis_summary"]["processing_errors"]
        if strategies.get("layoutlm_layout", 0) == 0:
            recommendations.append("Aucune page traitée par LayoutLM - vérifier le modèle (nom/poids) ou l'environnement.")
        if errors > 0:
            recommendations.append(f"{errors} pages ont échoué - vérifier la qualité du PDF")
        if doc_data["analysis_summary"]["avg_confidence"] < 0.6:
            recommendations.append("Confiance faible - révision manuelle recommandée")
        return recommendations


class DocumentConfigs:
    """Configurations prédéfinies pour différents types de documents"""
    @staticmethod
    def get_config(doc_type: str) -> TableValidationConfig:
        configs = {
            "vlg_atmospherique": TableValidationConfig(max_columns=8, min_rows=3, max_null_percentage=0.3, min_content_ratio=0.4),
            "vlg_liquide": TableValidationConfig(max_columns=10, min_rows=5, max_null_percentage=0.2, min_content_ratio=0.5),
            "normes_air": TableValidationConfig(max_columns=6, min_rows=4, max_null_percentage=0.1, min_content_ratio=0.6),
            "vls_ciment": TableValidationConfig(max_columns=12, min_rows=3, max_null_percentage=0.4, min_content_ratio=0.3),
            "default": TableValidationConfig()
        }
        return configs.get(doc_type, configs["default"])


def process_pdf_batch(
    pdf_files: List[str],
    config: Optional[TableValidationConfig] = None,
    layout_model_name: Optional[str] = None,
    layout_device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Traite un batch de fichiers PDF avec le parser LayoutLM et des rapports détaillés.
    """
    results: Dict[str, Any] = {}
    processing_summary = {
        "total_files": len(pdf_files), "successful": 0, "failed": 0,
        "avg_confidence": 0.0, "processing_time": 0.0,
        "model_used": layout_model_name, "device_used": layout_device
    }

    import time
    import traceback
    start_time = time.time()

    for pdf_path in pdf_files:
        try:
            logger.info(f"Traitement: {pdf_path} sur {layout_device}")
            if not os.path.exists(pdf_path):
                logger.error(f"Fichier introuvable: {pdf_path}")
                results[pdf_path] = {"error": "Fichier introuvable"}
                processing_summary["failed"] += 1
                continue
            parser = PDFParser(
                pdf_path,
                config=config,
                layout_model_name=layout_model_name,
                layout_device=layout_device
            )
            result = parser.parse()
            results[pdf_path] = result
            processing_summary["successful"] += 1
            if "analysis_summary" in result and "avg_confidence" in result["analysis_summary"]:
                processing_summary["avg_confidence"] += result["analysis_summary"]["avg_confidence"]
            logger.info(f"✓ Succès: {os.path.basename(pdf_path)} - Confiance: {result['analysis_summary']['avg_confidence']:.2f}")
        except Exception as e:
            logger.error(f"✗ Erreur critique: {pdf_path} - {str(e)}")
            results[pdf_path] = {"error": str(e), "traceback": traceback.format_exc()}
            processing_summary["failed"] += 1

    processing_summary["processing_time"] = time.time() - start_time
    if processing_summary["successful"] > 0:
        processing_summary["avg_confidence"] /= processing_summary["successful"]
    return {"results": results, "summary": processing_summary}


    # === Public API wrappers expected by views/templates ===
from typing import Optional, List, Dict, Any, Union
import os, tempfile

__all__ = [
    "TableValidationConfig",
    "PDFParser",
    "DocumentConfigs",
    "process_pdf_batch",
    "extract_and_parse_pdf",
    "extract_and_parse_pdf_from_upload",
    "extract_and_parse_pdf_batch",
]

def extract_and_parse_pdf(
    pdf_path: str,
    *,
    layout_model_name: Optional[str] = None,
    layout_device: Optional[str] = None,  # e.g. "cpu", "cuda:0"
    table_config: Optional[TableValidationConfig] = None,
) -> Dict[str, Any]:
    """
    Drop-in wrapper used by views.py:
      from app.services.legal_register.pdf_parser import extract_and_parse_pdf

    Returns the same dict as PDFParser.parse().
    """
    if not pdf_path or not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF introuvable: {pdf_path}")

    # First pass init (classifies document in __init__)
    parser = PDFParser(
        pdf_path=pdf_path,
        config=table_config or TableValidationConfig(),
        layout_model_name=layout_model_name,
        layout_device=layout_device,
    )

    # If caller didn't force a config, specialize it using detected doc_type
    if table_config is None:
        try:
            parser.config = DocumentConfigs.get_config(parser.doc_type)
        except Exception:
            # Safe fallback: keep the generic config if anything goes wrong
            pass

    return parser.parse()


def extract_and_parse_pdf_from_upload(
    uploaded_file: Union[bytes, "UploadedFile"],
    *,
    filename: Optional[str] = None,
    layout_model_name: Optional[str] = None,
    layout_device: Optional[str] = None,
    table_config: Optional[TableValidationConfig] = None,
    cleanup: bool = True,
) -> Dict[str, Any]:
    """
    Helper for Django file uploads. Accepts either raw bytes or a Django UploadedFile.
    Saves to a temp .pdf then calls extract_and_parse_pdf.
    """
    # Write the content to a temp file
    suffix = ".pdf"
    if filename:
        # keep .pdf suffix if present, else ensure .pdf
        base, ext = os.path.splitext(filename)
        suffix = ext if ext.lower() == ".pdf" else ".pdf"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name

    try:
        if hasattr(uploaded_file, "chunks"):  # Django UploadedFile
            for chunk in uploaded_file.chunks():
                tmp.write(chunk)
        elif isinstance(uploaded_file, (bytes, bytearray)):
            tmp.write(uploaded_file)
        else:
            raise TypeError("uploaded_file doit être bytes ou UploadedFile")
        tmp.flush()
        tmp.close()

        result = extract_and_parse_pdf(
            tmp_path,
            layout_model_name=layout_model_name,
            layout_device=layout_device,
            table_config=table_config,
        )
        # annotate filename if provided
        if filename:
            result.setdefault("metadata", {}).setdefault("source_filename", filename)
        return result
    finally:
        if cleanup:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def extract_and_parse_pdf_batch(
    pdf_paths: List[str],
    *,
    layout_model_name: Optional[str] = None,
    layout_device: Optional[str] = None,
    table_config: Optional[TableValidationConfig] = None,
) -> Dict[str, Any]:
    """
    Optional: batch wrapper matching the extract_* naming your code expects.
    Forwards to process_pdf_batch with the same behavior you already had.
    """
    return process_pdf_batch(
        pdf_paths,
        config=table_config,
        layout_model_name=layout_model_name,
        layout_device=layout_device,
    )
