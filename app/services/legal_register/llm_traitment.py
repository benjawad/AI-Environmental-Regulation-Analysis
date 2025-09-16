from __future__ import annotations

import json, os, re, time, uuid, unicodedata, difflib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from django.conf import settings
from django.utils.module_loading import import_string

# Session key used by PDF download
LEGAL_REGISTER_TABLE_DATA_KEY = "legal_register_table_structured"

# Where we also write results.json for debugging
JSON_OUTPUT_DIR = Path(getattr(settings, "MEDIA_ROOT", "media")).joinpath("json").resolve()
JSON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------
# Fixed business categories (must match your PDF headers)
# --------------------------------------------------------------------------------
FIXED_CATEGORIES: Tuple[str, ...] = (
    "Solid Waste",
    "Water and Liquid discharges",
    "Noise and vibrations",
    "Air",
    "Energy",
    "General Environmental & Sustainability Regulations",
)

# --------------------------------------------------------------------------------
# ----------------------------- General helpers ----------------------------------
# --------------------------------------------------------------------------------

def _to_plain_dict(x):
    if hasattr(x, "model_dump") and callable(x.model_dump):
        return x.model_dump()
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

def _normalize_text(s: str) -> str:
    s = _to_str(s).lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"[^a-z0-9\s\-_/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# --------------------------------------------------------------------------------
# Dates & row mapping
# --------------------------------------------------------------------------------

def _extract_dates_from_row_and_doc(row: Dict[str, Any], doc: Dict[str, Any]) -> str:
    """
    Prefer entry.reference.date (ISO) if provided in row['date'].
    Else fall back to document-level publication_date (gregorian/hijri).
    """
    g = (
        row.get("date")
        or _extract_nested(doc, "publication_date.gregorian")
        or _extract_nested(doc, "metadata.date_gregorian", "bulletin_metadata.date_gregorian")
        or ""
    )
    h = (
        _extract_nested(doc, "publication_date.hijri")
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
    11 cells required by your PDF:
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

def _build_table_structured_sorted(rows_flat: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups = _build_table_structured(rows_flat)
    order = {name: i for i, name in enumerate(FIXED_CATEGORIES)}
    groups.sort(key=lambda g: (order.get(g["category_title"], 999), g["category_title"]))
    return groups

# --------------------------------------------------------------------------------
# Relevance utils
# --------------------------------------------------------------------------------

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

# --------------------------------------------------------------------------------
# Category mapping (heuristics)
# --------------------------------------------------------------------------------

def _heuristic_category(text: str) -> Optional[str]:
    t = _normalize_text(text)
    if re.search(r"\b(dechet|dechets|waste|landfill|decharge|ordure|recycl|compost|inciner|dangerous waste|hazardous)\b", t):
        return "Solid Waste"
    if re.search(r"\b(eau|eaux|water|liquid|effluent|wastewater|deversement|rejet|assainissement|sewer|drainage|discharge)\b", t):
        return "Water and Liquid discharges"
    if re.search(r"\b(bruit|noise|acoustic|acoustique|vibration|vibrations)\b", t):
        return "Noise and vibrations"
    if re.search(r"\b(air|atmospher|emission|gaz|polluant|pm10|pm2|nox|so2|co2|chimique atmos|qualite de l air)\b", t):
        return "Air"
    if re.search(r"\b(energie|énergie|energy|efficiency|rendement|fuel|electric|consommation|renewable|pv|solar)\b", t):
        return "Energy"
    if re.search(r"\b(environnemen|environment|sustainab|loi cadre|framework|general)\b", t):
        return "General Environmental & Sustainability Regulations"
    return None

# --------------------------------------------------------------------------------
# JSON parsing helpers for LLM replies
# --------------------------------------------------------------------------------

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

# --------------------------------------------------------------------------------
# LLM client plumbing
# --------------------------------------------------------------------------------

class _NoopClient:
    def generate_json(self, prompt: str) -> str:
        return "{}"

def _build_trainer(api_key: Optional[str]):
    """
    Returns object with .client.generate_json(prompt)->str
    Wire with settings.LLM_JSON_PROVIDER = "myproj.ai.json_provider:JsonClient"
    """
    provider_path = getattr(settings, "LLM_JSON_PROVIDER", "")
    client = _NoopClient()
    if provider_path:
        try:
            ClientCls = import_string(provider_path)
            client = ClientCls(api_key=api_key)  # type: ignore
        except Exception:
            pass
    return type("Trainer", (), {"client": client})()

# --------------------------------------------------------------------------------
# LLM prompts & field normalisation
# --------------------------------------------------------------------------------

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
\"\"\"{(legal_req or '').strip()[:1500]}\"\"\"""".strip()

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
- Legal requirement: {base_row.get('legal_requirement_raw','')[:1000]}
- Date: {base_row.get('date','')}
""".strip()

def _normalize_phase_en(s: str) -> str:
    if not s:
        return ""
    s0 = str(s).strip().lower()
    if any(k in s0 for k in ["design", "engineering", "feasibility", "basic design", "detailed design", "study"]):
        return "Design"
    if any(k in s0 for k in ["construct", "construction", "build", "erection", "site works", "civil works"]):
        return "Construction"
    if any(k in s0 for k in ["operate", "operation", "operational", "operations"]):
        return "Operation"
    if any(k in s0 for k in ["decommission", "decommissioning", "closure", "dismantling"]):
        return "Decommissioning"
    if "commission" in s0:
        return "Construction"
    return s.strip().capitalize() if len(s0) <= 20 else ""

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

def _llm_classify_category(trainer, project_desc: str, legal_req: str, type_hint: str) -> Optional[str]:
    gen = getattr(getattr(trainer, "client", None), "generate_json", None)
    if not callable(gen):
        return None
    raw = gen(_build_llm_classify_prompt(project_desc, legal_req, type_hint))
    data = _try_parse_json_str(raw) or {}
    cat = _to_str(data.get("category", "")).strip()
    return cat if cat in FIXED_CATEGORIES else None

def _llm_fill_fields_for_row(trainer, project_desc: str, base_row: Dict[str, Any]) -> Dict[str, Any]:
    gen = getattr(getattr(trainer, "client", None), "generate_json", None)
    if not callable(gen):
        return {}
    prompt = _build_llm_row_prompt(
        project_desc,
        base_row.get("fixed_category") or base_row.get("category_title") or "",
        base_row.get("type_hint") or "",
        base_row.get("legal_requirement_raw") or "",
        base_row.get("date") or "",
    )
    raw = gen(prompt)
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

# --------------------------------------------------------------------------------
# **Parser output adapter**: turn your parsed BO JSON into base rows
# --------------------------------------------------------------------------------

def _first_article_snippet(entry: Dict[str, Any]) -> str:
    try:
        ds = entry.get("document_structure") or {}
        preamble = ds.get("preamble") or []
        for item in preamble:
            txt = _to_str(item.get("text", "")).strip()
            if txt and len(txt.split()) >= 4:
                return txt[:400]
        chapters = ds.get("chapters") or []
        for ch in chapters:
            arts = (ch or {}).get("articles") or []
            if not arts:
                continue
            paras = (arts[0] or {}).get("paragraphs") or []
            if paras:
                joined = " ".join(p.strip() for p in paras if p)
                if joined:
                    return joined[:400]
    except Exception:
        pass
    return ""

def base_rows_from_parsed(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    content = doc.get("content") or []
    if not isinstance(content, list):
        return rows
    for entry in content:
        if not isinstance(entry, dict):
            continue
        doc_type = _to_str(entry.get("document_type", ""))
        legal_req = _to_str(entry.get("title", ""))
        ref = entry.get("reference") or {}
        ref_date = _to_str(ref.get("date", ""))
        ref_num = _to_str(ref.get("number", ""))

        legal_raw = legal_req
        suffix = []
        if ref_num:
            suffix.append(f"n° {ref_num}")
        if ref_date:
            suffix.append(ref_date)
        if suffix:
            legal_raw = f"{legal_req} ({', '.join(suffix)})"

        desc_raw = _first_article_snippet(entry)
        type_hint = doc_type
        cat_hint = _to_str(entry.get("category_raw", ""))

        rows.append({
            "category_title": cat_hint or "General Environmental & Sustainability Regulations",
            "type_hint": type_hint,
            "legal_requirement_raw": legal_raw,
            "description_raw": desc_raw,
            "date": _extract_dates_from_row_and_doc({"date": ref_date}, doc),
            "jurisdiction": "National",
            # EN placeholders
            "phase": "", "activity_aspect": "", "impacts": "",
            "description": "", "task": "", "responsibility": "", "comments": "",
        })
    return rows

# --------------------------------------------------------------------------------
# ------------------------- History integration ----------------------------------
# --------------------------------------------------------------------------------

def _norm_law_key(text: str) -> str:
    t = _normalize_text(text)
    if not t:
        return ""
    m = re.search(r"\b(law|loi|decree|decret|arr[ée]t[ée]|arrete|dahir|order|joint order)\b.*?(?:n[o°º]?\s*)?([0-9][0-9\-/.]{1,20})", t)
    if m:
        kind = m.group(1)
        kind = {"loi":"law","decret":"decree","arrete":"arrete","arrêté":"arrete"}.get(kind, kind)
        num = re.sub(r"[^0-9\-/.]", "", m.group(2))
        return f"{kind}|{num}"
    brand = ""
    if "ifc" in t: brand = "ifc"
    elif "who" in t: brand = "who"
    elif "ocp" in t: brand = "ocp"
    if brand:
        year = ""
        y = re.search(r"\b(19|20)\d{2}\b", t)
        if y: year = y.group(0)
        topic = ""
        for tok in ("wastewater","water","air","noise","hazardous","energy","emissions","waste"):
            if tok in t:
                topic = tok; break
        return "|".join([x for x in (brand, topic, year) if x])
    return ""

def _history_category_hint(prev_row: Dict[str, Any]) -> Optional[str]:
    pitch = " ".join([
        _to_str(prev_row.get("activity_aspect","")),
        _to_str(prev_row.get("impacts","")),
        _to_str(prev_row.get("type","")),
        _to_str(prev_row.get("comments","")),
    ])
    return _heuristic_category(pitch)

def _history_prefill(prev_row: Dict[str, Any]) -> Dict[str, str]:
    return {
        "phase": _normalize_phase_en(_to_str(prev_row.get("phase",""))),
        "activity_aspect": _to_str(prev_row.get("activity_aspect","")),
        "impacts": _to_str(prev_row.get("impacts","")),
        "description": _to_str(prev_row.get("description","")),
        "task": _to_str(prev_row.get("task","")),
        "responsibility": _to_str(prev_row.get("responsibility","")),
        "comments": _to_str(prev_row.get("comments","")),
    }

def _build_history_index(history_rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    idx: Dict[str, List[Dict[str, Any]]] = {}
    for r in history_rows or []:
        key = _norm_law_key(_to_str(r.get("legal_requirement","")))
        if not key:
            continue
        idx.setdefault(key, []).append(r)
    return idx

def _best_history_match(history_idx: Dict[str, List[Dict[str, Any]]], text: str) -> Optional[Dict[str, Any]]:
    key = _norm_law_key(text)
    if key and key in history_idx:
        rows = history_idx[key]
        def score(rr):
            s = 0
            if rr.get("description"): s += 3
            if rr.get("task"): s += 2
            if rr.get("responsibility"): s += 1
            if rr.get("date_iso") or rr.get("date"): s += 1
            return -s
        return sorted(rows, key=score)[0]
    if history_idx:
        candidates = [(k, difflib.SequenceMatcher(None, k, _normalize_text(text)).ratio()) for k in history_idx.keys()]
        candidates.sort(key=lambda x: x[1], reverse=True)
        if candidates and candidates[0][1] >= 0.75:
            return history_idx[candidates[0][0]][0]
    return None

# --------------------------------------------------------------------------------
# ------------------------- Public orchestration ---------------------------------
# --------------------------------------------------------------------------------

def run_llm_analysis(
    *,
    description: str,
    parsed_docs: List[Dict[str, Any]],
    controls: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns (table_structured, diagnostics)
    - table_structured: [{category_title, rows: [[11 cells], ...]}, ...]
    - diagnostics: stats for logs/UI
    """
    controls = controls or {}
    use_llm_fill = bool(controls.get("use_llm_fill", True))
    relevance_mode = (controls.get("relevance_mode") or "hybrid").lower()
    relevance_threshold = float(controls.get("relevance_threshold", 0.12))
    relevance_gray_margin = float(controls.get("relevance_gray_margin", 0.06))
    relevance_keep_top_k = int(controls.get("relevance_keep_top_k", 30))
    max_llm_relevance = int(controls.get("max_llm_relevance", 10))
    max_llm_classify = int(controls.get("max_llm_classify", 20))
    max_llm_rows = int(controls.get("max_llm_rows", 25))
    api_key = controls.get("api_key")

    # --- History (optional)
    history_rows: List[Dict[str, Any]] = controls.get("history_rows") or []
    history_json_files: List[str] = controls.get("history_json_files") or []
    for path in history_json_files:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                arr = json.load(fh)
                if isinstance(arr, list):
                    history_rows.extend([r for r in arr if isinstance(r, dict)])
        except Exception:
            pass
    history_idx = _build_history_index(history_rows)
    history_boost = float(controls.get("history_relevance_boost", 0.08))
    prefer_history = bool(controls.get("prefer_history_fields", True))
    merge_strategy = controls.get("merge_strategy", "history_then_llm")

    # --- Build base rows from all parsed docs
    all_base_rows: List[Dict[str, Any]] = []
    for item in parsed_docs:
        doc = item.get("parsed") if isinstance(item, dict) else None
        if isinstance(doc, dict):
            all_base_rows.extend(base_rows_from_parsed(doc))

    # --- Init trainer if needed
    trainer_for_classify = None
    if use_llm_fill or relevance_mode in ("llm", "hybrid"):
        try:
            trainer_for_classify = _build_trainer(api_key)
        except Exception:
            trainer_for_classify = None

    # --- Category mapping (history → heuristic → LLM)
    llm_classify_calls = 0
    for r in all_base_rows:
        hist = _best_history_match(history_idx, r.get("legal_requirement_raw",""))
        cat = _history_category_hint(hist) if hist else None
        if not cat:
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
        r["_history_match"] = hist or None

    # --- Relevance (history boost)
    scored = []
    for r in all_base_rows:
        s = _heuristic_relevance_score(description, {
            "category_title": r.get("fixed_category",""),
            "type": r.get("type_hint",""),
            "legal_requirement_raw": r.get("legal_requirement_raw",""),
            "description_raw": r.get("description_raw",""),
        })
        if r.get("_history_match") is not None:
            s = min(1.0, s + history_boost)
        scored.append((s, r))
    scored.sort(key=lambda x: x[0], reverse=True)

    # --- Selection
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
            gen = getattr(getattr(trainer_for_classify, "client", None), "generate_json", None)
            if callable(gen):
                prompt = f'Reply ONLY JSON {{"relevant": true|false}}\nProject description:\n"{description[:1200]}"\nLegal requirement:\n"{r.get("legal_requirement_raw","")[:1200]}"'
                ans = gen(prompt)
                data = _try_parse_json_str(ans) or {}
                if bool(data.get("relevant", False)):
                    selected.append(r)
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
                gen = getattr(getattr(trainer_for_classify, "client", None), "generate_json", None)
                if callable(gen):
                    prompt = f'Reply ONLY JSON {{"relevant": true|false}}\nProject description:\n"{description[:1200]}"\nLegal requirement:\n"{r.get("legal_requirement_raw","")[:1200]}"'
                    ans = gen(prompt)
                    data = _try_parse_json_str(ans) or {}
                    if bool(data.get("relevant", False)):
                        selected.append(r)
                    llm_relevance_calls += 1
        if not selected and relevance_keep_top_k > 0:
            selected = [r for _, r in scored[:relevance_keep_top_k]]

    # --- Fill (history prefill, LLM for blanks)
    enriched_rows: List[Dict[str, Any]] = []
    llm_fill_calls = 0

    for base in selected:
        filled = dict(base)
        fixed_cat = base.get("fixed_category") or "General Environmental & Sustainability Regulations"
        filled["category_title"] = fixed_cat
        filled["type"] = fixed_cat

        hist = base.get("_history_match")
        if prefer_history and isinstance(hist, dict):
            for k, v in _history_prefill(hist).items():
                if v and not filled.get(k):
                    filled[k] = v
            if not filled.get("date"):
                d_iso = _to_str(hist.get("date_iso",""))
                d = _to_str(hist.get("date",""))
                filled["date"] = d_iso or d or filled.get("date","")
            if not filled.get("legal_requirement"):
                prev_title = _to_str(hist.get("legal_requirement",""))
                filled["legal_requirement"] = prev_title or _to_str(base.get("legal_requirement_raw",""))

        do_llm = use_llm_fill and trainer_for_classify is not None and llm_fill_calls < max_llm_rows

        if merge_strategy == "history_then_llm" and do_llm:
            sugg = _llm_fill_fields_for_row(trainer_for_classify, description, base)
            for k in ("phase","activity_aspect","impacts","description","task","responsibility","comments"):
                if not _to_str(filled.get(k,"")).strip() and _to_str(sugg.get(k,"")).strip():
                    filled[k] = _to_str(sugg[k])
            en_req = _to_str(sugg.get("legal_requirement_en","")).strip()
            if en_req:
                filled["legal_requirement"] = en_req
            elif not filled.get("legal_requirement"):
                filled["legal_requirement"] = _to_str(base.get("legal_requirement_raw",""))
            llm_fill_calls += 1

        elif do_llm:
            sugg = _llm_fill_fields_for_row(trainer_for_classify, description, base)
            for k in ("phase","activity_aspect","impacts","description","task","responsibility","comments"):
                if _to_str(sugg.get(k,"")).strip():
                    filled[k] = _to_str(sugg[k])
            en_req = _to_str(sugg.get("legal_requirement_en","")).strip()
            if en_req:
                filled["legal_requirement"] = en_req
            llm_fill_calls += 1
            if isinstance(hist, dict):
                for k, v in _history_prefill(hist).items():
                    if v and not filled.get(k):
                        filled[k] = v
                if not filled.get("legal_requirement"):
                    filled["legal_requirement"] = _to_str(base.get("legal_requirement_raw",""))

        else:
            if not filled.get("legal_requirement"):
                filled["legal_requirement"] = _to_str(base.get("legal_requirement_raw",""))

        enriched_rows.append(filled)

    table_structured = _build_table_structured_sorted(enriched_rows)

    diagnostics = {
        "total_base": len(all_base_rows),
        "selected": len(selected),
        "llm_classify_calls": llm_classify_calls,
        "llm_relevance_calls": llm_relevance_calls,
        "llm_fill_calls": llm_fill_calls,
        "categories": len(table_structured),
        "rows": sum(len(c["rows"]) for c in table_structured),
    }
    return table_structured, diagnostics
