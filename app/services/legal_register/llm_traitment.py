# services/gemini_training.py
# pip install google-generativeai pydantic

from __future__ import annotations

import os
import re
import json
import time
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ValidationError, validator, root_validator
import google.generativeai as genai

try:
    # Adjust if your actual module path differs
    from services.pdf_parser import (
        BOPdfParsingService,
        extract_text_pages_hybrid,
    )
    _HAS_PARSER = True
except Exception:
    _HAS_PARSER = False


# =========================
# Pydantic Schema (Strict)
# =========================

class BulletinMetadata(BaseModel):
    issue_number: str
    publication_year_description: str
    date_hijri: str
    date_gregorian: str  
    issn: str


class TOCItem(BaseModel):
    category: str
    title: str
    description: str = ""
    page: int

    @validator("page", pre=True)
    def _coerce_page(cls, v):
        if v is None or v == "":
            return 0
        try:
            return int(v)
        except Exception:
            m = re.search(r"\d+", str(v))
            return int(m.group()) if m else 0


class Signatory(BaseModel):
    name: str
    title: str = ""


class LoanGuarantee(BaseModel):
    amount: float
    currency: str


class ContentDetails(BaseModel):
    loan_guarantee: Optional[LoanGuarantee] = None


class LegalText(BaseModel):
    type: str
    number: str
    title: str
    publication_date_hijri: str = ""
    publication_date_gregorian: str = ""
    page_start: int
    description: str = ""
    signatories: List[Signatory] = []
    content_details: Optional[ContentDetails] = None

    @validator("page_start", pre=True)
    def _coerce_page_start(cls, v):
        if v is None or v == "":
            return 0
        try:
            return int(v)
        except Exception:
            m = re.search(r"\d+", str(v))
            return int(m.group()) if m else 0


class BulletinIssue(BaseModel):
    bulletin_metadata: BulletinMetadata
    table_of_contents: List[TOCItem]
    legal_texts: List[LegalText]

    @root_validator(pre=True)
    def _ensure_lists(cls, values):
        values["table_of_contents"] = values.get("table_of_contents") or []
        values["legal_texts"] = values.get("legal_texts") or []
        return values


# =========================
# Gemini Client (JSON mode)
# =========================

class GeminiClient:
    """
    Minimal Gemini wrapper forcing JSON output with retry/backoff.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-flash-latest",
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 40,
        max_retries: int = 5,
        initial_wait: float = 2.0,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not provided. Pass api_key or set env GEMINI_API_KEY.")
        genai.configure(api_key=self.api_key)

        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "temperature": float(temperature),
                "top_p": float(top_p),
                "top_k": int(top_k),
                "response_mime_type": "application/json",
            },
        )
        self.max_retries = int(max_retries)
        self.initial_wait = float(initial_wait)

    def generate_json(self, prompt: str) -> str:
        wait = self.initial_wait
        for attempt in range(self.max_retries):
            try:
                resp = self.model.generate_content(prompt)
                return resp.text
            except Exception as e:
                msg = str(e)
                if (("429" in msg) or ("rate" in msg.lower())) and attempt < self.max_retries - 1:
                    time.sleep(wait)
                    wait *= 2
                    continue
                raise


# =========================
# Few-shot Memory (ICL)
# =========================

class FewShotMemory:
    """
    Holds (input_text, target_json) pairs for in-context learning ("training" via prompt).
    """

    def __init__(self, limit_chars: int = 180_000):
        self.examples: List[Tuple[str, Dict[str, Any]]] = []
        self.limit_chars = int(limit_chars)

    def add(self, input_text: str, target_json: Dict[str, Any]) -> None:
        self.examples.append((input_text, target_json))

    def extend(self, pairs: List[Tuple[str, Dict[str, Any]]]) -> None:
        self.examples.extend(pairs)

    def clear(self) -> None:
        self.examples.clear()

    def save(self, path: str) -> None:
        data = [{"input_text": t, "target_json": j} for (t, j) in self.examples]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.examples = [(d["input_text"], d["target_json"]) for d in data]

    def build_block(self, max_chars: Optional[int] = None) -> str:
        max_chars = int(max_chars or self.limit_chars)
        parts: List[str] = []
        total = 0
        for i, (inp, tgt) in enumerate(self.examples, start=1):
            block = (
                f"Exemple {i}:\n"
                f"Texte source:\n\"\"\"{inp.strip()[:6000]}\"\"\"\n\n"
                f"JSON attendu:\n{json.dumps(tgt, ensure_ascii=False)}\n"
            )
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)
        return "\n".join(parts)


# =========================
# Trainer/Refiner Class
# =========================

class GeminiBulletinTrainer:
    """
    - Train (few-shot ICL) using your parser outputs as labels
    - Predict full JSON from raw text
    - Refine/repair an existing parsed JSON using Gemini
    """

    def __init__(self, client: GeminiClient, language: str = "fr", max_repair_rounds: int = 2):
        self.client = client
        self.language = language
        self.max_repair_rounds = max_repair_rounds
        self.memory = FewShotMemory()

    # -------- Public API --------

    def add_example(self, input_text: str, target_json: Dict[str, Any]) -> None:
        self.memory.add(input_text, target_json)

    def add_examples_from_pdfs(self, pdf_paths: List[str]) -> None:
        """
        Build (input_text, target_json) pairs from PDFs using your parser.
        Requires services/pdf_parser.py availability.
        """
        if not _HAS_PARSER:
            raise RuntimeError("Parser not available. Ensure services/pdf_parser.py is importable.")
        svc = BOPdfParsingService()
        for p in pdf_paths:
            # Raw text
            raw_text, _ = extract_text_pages_hybrid(p)
            # Parsed JSON (target)
            target = svc.parse_pdf(p)
            self.add_example(raw_text, target)

    def predict_from_text(self, raw_text: str, extra_context: Optional[str] = None) -> BulletinIssue:
        """
        Generate full JSON strictly from raw text using few-shot ICL.
        """
        prompt = self._build_prompt_for_full_extraction(raw_text, extra_context)
        raw = self.client.generate_json(prompt)
        data = self._try_parse_json(raw)

        if data is not None:
            try:
                return BulletinIssue.parse_obj(data)
            except ValidationError as ve:
                last_error = str(ve)
        else:
            last_error = "Initial JSON parse failed."

        # Repair loop
        for _ in range(self.max_repair_rounds):
            repair_prompt = self._build_repair_prompt(raw, last_error)
            raw = self.client.generate_json(repair_prompt)
            data = self._try_parse_json(raw)
            if data is not None:
                try:
                    return BulletinIssue.parse_obj(data)
                except ValidationError as ve:
                    last_error = str(ve)
                    continue

        data = self._lenient_parse(raw)
        if data is not None:
            return BulletinIssue.parse_obj(data)
        raise ValueError("Unable to obtain valid BulletinIssue JSON after repair attempts.")

    def refine_parsed_json(
        self,
        parsed_json: Dict[str, Any],
        raw_text: Optional[str] = None,
        strategy: str = "fill_missing",  # "fill_missing" | "rewrite"
        extra_context: Optional[str] = None,
    ) -> BulletinIssue:
        """
        Use Gemini to validate/repair the parser's JSON.
        - fill_missing: keep parser fields; only fill ""/null and add missing objects
        - rewrite: let Gemini return a full JSON; we accept it as is (still validated)
        """
        # Validate the base first (so we know the structure)
        try:
            base = BulletinIssue.parse_obj(parsed_json).dict()
        except ValidationError:
            # If base invalid, fallback to rewrite
            strategy = "rewrite"

        if strategy == "rewrite":
            prompt = self._build_prompt_for_rewrite(raw_text, extra_context)
            raw = self.client.generate_json(prompt)
            data = self._try_parse_json(raw)
            return self._finalize_or_repair(data, raw)

        # Fill-missing strategy
        prompt = self._build_prompt_for_fill_missing(base, raw_text, extra_context)
        raw = self.client.generate_json(prompt)
        data = self._try_parse_json(raw)
        if data is None:
            return self._finalize_or_repair(None, raw)

        # Merge suggestion into base (only fill missing/empty)
        merged = self._merge_fill_missing(base, data)
        try:
            return BulletinIssue.parse_obj(merged)
        except ValidationError:
            # If merge makes it invalid, try repairing suggestion alone
            return self._finalize_or_repair(data, raw)

    # -------- Internals: Prompt Builders --------

    def _schema_block(self) -> str:
        return """
Schéma JSON cible (types stricts) :
{
  "bulletin_metadata": {
    "issue_number": "string",
    "publication_year_description": "string",
    "date_hijri": "string",
    "date_gregorian": "YYYY-MM-DD",
    "issn": "string"
  },
  "table_of_contents": [
    {
      "category": "string",
      "title": "string",
      "description": "string",
      "page": 0
    }
  ],
  "legal_texts": [
    {
      "type": "string",
      "number": "string",
      "title": "string",
      "publication_date_hijri": "string",
      "publication_date_gregorian": "YYYY-MM-DD",
      "page_start": 0,
      "description": "string",
      "signatories": [{"name": "string", "title": "string"}],
      "content_details": {
        "loan_guarantee": {"amount": 0.0, "currency": "string"}
      }
    }
  ]
}
""".strip()

    def _rules_block(self) -> str:
        return """
Règles impératives:
- Retourne UNIQUEMENT un JSON valide, sans texte avant/après, sans balises Markdown.
- Respecte EXACTEMENT les clés et la structure du schéma.
- Types stricts: page/page_start = int; amount = float.
- Conserve la casse et les accents d'origine.
- Si une information est absente, renvoie "" ou null (sous-objet optionnel).
- N'invente pas de contenu; n'ajoute pas de champs hors schéma.
""".strip()

    def _build_prompt_for_full_extraction(self, text: str, extra_context: Optional[str]) -> str:
        examples = self.memory.build_block()
        return f"""
Tu es un assistant d'extraction (langue: {self.language}). À partir du texte source,
produis STRICTEMENT le JSON suivant.

{self._schema_block()}

{self._rules_block()}

{f'Contexte additionnel: {extra_context}' if extra_context else ''}

{('Exemples (few-shot):\n' + examples) if examples else ''}

Texte source:
\"\"\"{(text or '').strip()[:200000]}\"\"\"
""".strip()

    def _build_prompt_for_fill_missing(self, base_json: Dict[str, Any], text: Optional[str], extra_context: Optional[str]) -> str:
        examples = self.memory.build_block()
        return f"""
Tu dois COMPLÉTER un JSON partiellement rempli. Règles:
- Ne modifie pas les champs déjà renseignés (conserve leur valeur).
- Remplis uniquement les champs vides "" ou null ou les listes/objets manquants.
- Retourne le JSON COMPLET (toutes les clés présentes), strictement conforme au schéma.
- Pas de texte hors JSON, pas de Markdown.

{self._schema_block()}

{self._rules_block()}

{f'Contexte additionnel: {extra_context}' if extra_context else ''}

{('Exemples (few-shot):\n' + examples) if examples else ''}

JSON de départ (à compléter sans altérer les valeurs existantes):
{json.dumps(base_json, ensure_ascii=False)}

{f'Texte source pour vérification:\n\"\"\"{(text or "").strip()[:120000]}\"\"\"' if text else ''}
""".strip()

    def _build_prompt_for_rewrite(self, text: Optional[str], extra_context: Optional[str]) -> str:
        examples = self.memory.build_block()
        return f"""
Produit STRICTEMENT le JSON complet à partir du texte source. Respecte le schéma et les types.

{self._schema_block()}

{self._rules_block()}

{f'Contexte additionnel: {extra_context}' if extra_context else ''}

{('Exemples (few-shot):\n' + examples) if examples else ''}

Texte source:
\"\"\"{(text or '').strip()[:200000]}\"\"\"
""".strip()

    def _build_repair_prompt(self, last_output: str, validation_error: str) -> str:
        return f"""
Le JSON suivant est invalide. Corrige-le pour respecter STRICTEMENT le schéma demandé.
RENVOIE UNIQUEMENT du JSON valide, sans texte autour.

Erreur:
{validation_error}

JSON à corriger:
{self._strip_code_fences(last_output)}
""".strip()

    # -------- Internals: JSON helpers --------

    def _finalize_or_repair(self, data: Optional[Dict[str, Any]], raw: str) -> BulletinIssue:
        last_error = "Invalid or empty JSON"
        if data is not None:
            try:
                return BulletinIssue.parse_obj(data)
            except ValidationError as ve:
                last_error = str(ve)

        for _ in range(self.max_repair_rounds):
            repair_prompt = self._build_repair_prompt(raw, last_error)
            raw = self.client.generate_json(repair_prompt)
            data = self._try_parse_json(raw)
            if data is not None:
                try:
                    return BulletinIssue.parse_obj(data)
                except ValidationError as ve:
                    last_error = str(ve)
                    continue

        data = self._lenient_parse(raw)
        if data is not None:
            return BulletinIssue.parse_obj(data)
        raise ValueError("Unable to obtain valid BulletinIssue JSON after repair attempts.")

    @staticmethod
    def _strip_code_fences(s: str) -> str:
        s = s.strip()
        if s.startswith("```"):
            s = re.sub(r"^```[a-zA-Z]*\n", "", s)
            s = re.sub(r"\n```$", "", s)
        return s.strip()

    def _extract_json_block(self, s: str) -> Optional[str]:
        s = self._strip_code_fences(s)
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

    def _try_parse_json(self, s: str) -> Optional[Dict[str, Any]]:
        s = self._strip_code_fences(s)
        if not s.strip().startswith("{"):
            block = self._extract_json_block(s)
            if block:
                s = block
        try:
            return json.loads(s)
        except Exception:
            return None

    def _lenient_parse(self, s: str) -> Optional[Dict[str, Any]]:
        s = self._strip_code_fences(s)
        block = self._extract_json_block(s) or s
        # Remove trailing commas
        block = re.sub(r",(\s*[}```])", r"\1", block)
        # Normalize quotes
        block = block.replace("“", '"').replace("”", '"').replace("’", "'")
        try:
            return json.loads(block)
        except Exception:
            return None

    def _merge_fill_missing(self, base: Dict[str, Any], sug: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge suggestion into base: only fill empty ""/null or missing keys. Lists merged by union.
        """
        def merge(a: Any, b: Any) -> Any:
            if isinstance(a, dict) and isinstance(b, dict):
                out = dict(a)
                for k, vb in b.items():
                    va = out.get(k, None)
                    if va in ("", None):
                        out[k] = vb
                    elif va is None and vb is not None:
                        out[k] = vb
                    elif k not in out:
                        out[k] = vb
                    else:
                        out[k] = merge(va, vb)
                return out
            if isinstance(a, list) and isinstance(b, list):
                # union by JSON string repr to avoid dup dicts
                seen = set()
                out_list = []
                for item in a + b:
                    key = json.dumps(item, sort_keys=True, ensure_ascii=False)
                    if key not in seen:
                        seen.add(key)
                        out_list.append(item)
                return out_list
            # if base has meaningful value, keep it; else take b
            if (a in ("", None)) and (b not in ("", None)):
                return b
            return a
        return merge(base, sug)


# =========================
# Quick usage helper (optional)
# =========================

def make_trainer(api_key: Optional[str] = None) -> GeminiBulletinTrainer:
    client = GeminiClient(api_key=api_key, model_name="gemini-1.5-flash-latest", temperature=0.2)
    return GeminiBulletinTrainer(client)