import json
import logging
import re
import time
from typing import Dict, List, Any, Tuple

import pandas as pd
from django.conf import settings

import google.generativeai as genai

logger = logging.getLogger(__name__)


class CommitmentRegisterAnalyzer:
    """
    Analyzer that completes commitment rows using Gemini and a local knowledge base.
    - Enforces non-empty project_description at init.
    - Always uses the project description in prompts (even if no relevant docs are found).
    - Emits data that matches exactly your 3-level MultiIndex column structure and your PDF generator.
    """

    # This structure MUST match the expected order/shape used by your PDF generator (31 columns total).
    _COLUMNS: List[Tuple[str, str, str]] = [
        ('Commitment Register Overview', 'Register Identifier', ''),
        ('Commitment Register Overview', 'Commitment Identifier', ''),
        ('Commitment Register Overview', 'Commitment or Obligation', ''),
        ('Commitment Register Overview', 'Description', ''),
        ('Commitment Register Overview', 'Project Phase', ''),
        ('Commitment Management', 'Potential Impact on Scope?', ''),
        ('Commitment Management', 'Status', ''),
        ('Commitment Management', 'Commitment Deadline', ''),
        ('Commitment Management', 'First Lead', ''),
        ('Commitment Management', 'Second Lead', ''),
        ('Commitment Management', 'Third Lead', ''),
        ('Commitment Management', 'Primary Commitment Documentation', ''),
        ('Commitment Management', 'Impact or Hazard Addressed', ''),
        ('Commitment Management', 'Approving Agencies', ''),
        ('Commitment Management', 'Other Stakeholders', ''),
        ('Commitment Management', 'Affected Areas or Processes', 'Preparation/construction'),
        ('Commitment Management', 'Affected Areas or Processes', 'Operation'),
        ('Commitment Management', 'Affected Areas or Processes', 'Input Management'),
        ('Commitment Management', 'Affected Areas or Processes', 'Discharge management'),
        ('Commitment Management', 'Affected Areas or Processes', 'Off-Sites'),
        ('Commitment Management', 'Affected Areas or Processes', 'Other'),
        ('Commitment Management', 'Affected Areas or Processes', 'Fungibility'),
        ('Commitment Management', 'Impact', 'CAPEX'),
        ('Commitment Management', 'Impact', 'OPEX'),
        ('Commitment Management', 'Impact', 'Health & Safety'),
        ('Commitment Management', 'Impact', 'Social'),
        ('Commitment Management', 'Impact', 'Economic'),
        ('Commitment Management', 'Impact', 'Environmental'),
        ('Commitment Management', 'Impact', 'Regulatory'),
        ('Commitment Management', 'Comments', ''),
        ('Commitment Management', 'Requires Change Order?', ''),
    ]

    # JSON keys we want the model to return. These map to cells we set later.
    EXPECTED_KEYS = [
        "Impact or Hazard Addressed",
        "Approving Agencies",
        "Comments",

        "Affected_Preparation_Construction",
        "Affected_Operation",
        "Affected_Input_Management",
        "Affected_Discharge_Management",
        "Affected_Off_Sites",
        "Affected_Other",
        "Affected_Fungibility",

        "Impact_CAPEX",
        "Impact_OPEX",
        "Impact_Health_Safety",
        "Impact_Social",
        "Impact_Economic",
        "Impact_Environmental",
        "Impact_Regulatory",
    ]

    def __init__(self, project_description: str):
        """Initialize with API key and project description (required)."""
        if not project_description or not project_description.strip():
            raise ValueError("project_description is required and cannot be empty.")

        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            logger.info("Gemini API configured successfully.")
        except Exception as e:
            raise ValueError(f"Error configuring Gemini API: {e}")

        self.project_description = project_description.strip()
        self.columns = pd.MultiIndex.from_tuples(self._COLUMNS)

    # Expose structure for DataFrame creation
    def _get_column_structure(self) -> List[Tuple[str, str, str]]:
        return self._COLUMNS

    @staticmethod
    def _norm_text(s: Any) -> str:
        try:
            return (s or "").strip()
        except Exception:
            return ""

    def find_relevant_documents(self, commitment_description: str, knowledge_base: List[Dict[str, str]]) -> str:
        """
        Find KB snippets relevant to the given commitment description using simple keyword heuristics.
        The analyzer ALWAYS runs even if no docs are found.
        """
        cd = self._norm_text(commitment_description).lower()
        if not cd:
            return ""

        # keep words length >=4, including some French diacritics
        keywords = set(re.findall(r'\b[a-zA-Zçéàèùâêîôûæœ\d]{4,}\b', cd))
        relevant_texts = []
        for doc in knowledge_base or []:
            fn = self._norm_text(doc.get('filename')).lower()
            content = self._norm_text(doc.get('content')).lower()
            if not content:
                continue
            if any(k in fn for k in keywords) or any(k in content for k in keywords):
                # cap each doc content to avoid oversize prompts
                snippet = doc.get('content', '')[:6000]
                relevant_texts.append(
                    f"--- START OF RELEVANT DOCUMENT ({doc.get('filename', 'unknown')}) ---\n"
                    f"{snippet}\n--- END OF DOCUMENT ---"
                )
        return "\n\n".join(relevant_texts)

    @staticmethod
    def _clean_gemini_json_text(txt: str) -> str:
        cleaned = (txt or "").strip()
        # Remove common code-fence wrappers
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return cleaned.strip()

    def _normalize_llm_result(self, raw: Dict[str, Any]) -> Dict[str, str]:
        """Ensure all expected keys exist and are normalized as strings."""
        result: Dict[str, str] = {}
        for k in self.EXPECTED_KEYS:
            v = raw.get(k, "")
            if isinstance(v, bool):
                v = "x" if v else ""
            result[k] = self._norm_text(v)
        # Force allowed marks to 'x' or ''
        for phase_key in [
            "Affected_Preparation_Construction", "Affected_Operation", "Affected_Input_Management",
            "Affected_Discharge_Management", "Affected_Off_Sites", "Affected_Other", "Affected_Fungibility",
            "Impact_CAPEX", "Impact_OPEX", "Impact_Health_Safety", "Impact_Social",
            "Impact_Economic", "Impact_Environmental", "Impact_Regulatory"
        ]:
            val = result.get(phase_key, "")
            val = val.lower().strip()
            result[phase_key] = "x" if val in {"x", "yes", "true", "1", "checked"} else ""
        return result

    def call_gemini(self, commitment_row: pd.Series, relevant_texts: str) -> Dict[str, str]:
        """Call Gemini to complete one row. Always includes project description in the prompt."""
        # Use a lightweight model by default; swap to pro if you need higher quality.
        model = genai.GenerativeModel('gemini-1.5-flash')

        commitment_desc = self._norm_text(commitment_row[('Commitment Register Overview', 'Description', '')])
        commitment_id = self._norm_text(commitment_row[('Commitment Register Overview', 'Commitment Identifier', '')])

        # Ensure we always have something for legal evidence
        legal_evidence = relevant_texts if relevant_texts else (
            "No specific legal documents were matched. Rely on Moroccan environmental and project compliance best practices."
        )

        # Keep the prompt short and deterministic
        prompt = f"""
You are an expert Moroccan environmental & project compliance analyst.

Use ALL THREE sources to complete EXACTLY the following JSON schema. JSON ONLY in the response.

SOURCE 1: PROJECT DESCRIPTION
{self.project_description}

SOURCE 2: COMMITMENT CONTEXT
- Commitment ID: "{commitment_id}"
- Commitment Description: "{commitment_desc}"

SOURCE 3: LEGAL EVIDENCE (Moroccan regulations, decrees, guidance)
{legal_evidence}

TASK
1) Analyze how the commitment relates to project phases and compliance objectives.
2) Fill ALL keys below. For phase/impact flags, ONLY use "x" or "" (empty string).
3) Be concise, factual, and avoid hallucinating agencies or laws not implicated by the context.

OUTPUT (JSON ONLY):
{{
  "Impact or Hazard Addressed": "Short statement of the risk or objective",
  "Approving Agencies": "Comma-separated list (e.g., 'Ministry of Energy Transition, ABH, Local Authorities')",
  "Comments": "1-2 sentences linking the commitment to the project phase and compliance logic",

  "Affected_Preparation_Construction": "x or ''",
  "Affected_Operation": "x or ''",
  "Affected_Input_Management": "x or ''",
  "Affected_Discharge_Management": "x or ''",
  "Affected_Off_Sites": "x or ''",
  "Affected_Other": "x or ''",
  "Affected_Fungibility": "x or ''",

  "Impact_CAPEX": "x or ''",
  "Impact_OPEX": "x or ''",
  "Impact_Health_Safety": "x or ''",
  "Impact_Social": "x or ''",
  "Impact_Economic": "x or ''",
  "Impact_Environmental": "x or ''",
  "Impact_Regulatory": "x or ''"
}}
        """.strip()

        max_retries = 3
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = model.generate_content(prompt)
                if not resp or not getattr(resp, "text", ""):
                    logger.warning("Empty Gemini response for '%s' (attempt %s)", commitment_id, attempt)
                    time.sleep(1)
                    continue

                cleaned = self._clean_gemini_json_text(resp.text)
                raw = json.loads(cleaned)
                return self._normalize_llm_result(raw)

            except json.JSONDecodeError as e:
                last_err = e
                logger.warning("JSON decode error for '%s' (attempt %s): %s", commitment_id, attempt, e)
                logger.debug("Raw response: %r", resp.text if resp else None)
                time.sleep(1)
            except Exception as e:
                last_err = e
                logger.warning("Unexpected error for '%s' (attempt %s): %s", commitment_id, attempt, e)
                time.sleep(1)

        logger.error("All Gemini attempts failed for '%s'. Last error: %s", commitment_id, last_err)
        # Return empty structure (keeps your DataFrame stable)
        return {k: "" for k in self.EXPECTED_KEYS}

    def analyze_commitments(self, commitments_data: List[List[str]], knowledge_base: List[Dict[str, str]]) -> pd.DataFrame:
        """
        Main loop to analyze commitments with full schema. Always runs analysis even if no KB docs match.
        :param commitments_data: List of rows matching self.columns shape (we set ID+Description before call).
        :param knowledge_base: List of {'filename','content'} items.
        :return: Final DataFrame with the same MultiIndex columns.
        """
        if not isinstance(commitments_data, list) or not commitments_data:
            raise ValueError("commitments_data must be a non-empty list of rows.")

        df_initial = pd.DataFrame(commitments_data, columns=self.columns)
        df_final = df_initial.copy()

        for index, row in df_initial.iterrows():
            commitment_description = self._norm_text(row[('Commitment Register Overview', 'Description', '')])
            commitment_id = self._norm_text(row[('Commitment Register Overview', 'Commitment Identifier', '')])

            relevant_texts = self.find_relevant_documents(commitment_description, knowledge_base)

            # ALWAYS analyze (even if relevant_texts is empty)
            extracted = self.call_gemini(row, relevant_texts)
            time.sleep(0.5)  # Be kind to the API; adjust as needed

            # Map extracted fields to DataFrame
            df_final.loc[index, ('Commitment Management', 'Impact or Hazard Addressed', '')] = extracted.get("Impact or Hazard Addressed", "")
            df_final.loc[index, ('Commitment Management', 'Approving Agencies', '')] = extracted.get("Approving Agencies", "")
            df_final.loc[index, ('Commitment Management', 'Comments', '')] = extracted.get("Comments", "")

            df_final.loc[index, ('Commitment Management', 'Affected Areas or Processes', 'Preparation/construction')] = extracted.get("Affected_Preparation_Construction", "")
            df_final.loc[index, ('Commitment Management', 'Affected Areas or Processes', 'Operation')] = extracted.get("Affected_Operation", "")
            df_final.loc[index, ('Commitment Management', 'Affected Areas or Processes', 'Input Management')] = extracted.get("Affected_Input_Management", "")
            df_final.loc[index, ('Commitment Management', 'Affected Areas or Processes', 'Discharge management')] = extracted.get("Affected_Discharge_Management", "")
            df_final.loc[index, ('Commitment Management', 'Affected Areas or Processes', 'Off-Sites')] = extracted.get("Affected_Off_Sites", "")
            df_final.loc[index, ('Commitment Management', 'Affected Areas or Processes', 'Other')] = extracted.get("Affected_Other", "")
            df_final.loc[index, ('Commitment Management', 'Affected Areas or Processes', 'Fungibility')] = extracted.get("Affected_Fungibility", "")

            df_final.loc[index, ('Commitment Management', 'Impact', 'CAPEX')] = extracted.get("Impact_CAPEX", "")
            df_final.loc[index, ('Commitment Management', 'Impact', 'OPEX')] = extracted.get("Impact_OPEX", "")
            df_final.loc[index, ('Commitment Management', 'Impact', 'Health & Safety')] = extracted.get("Impact_Health_Safety", "")
            df_final.loc[index, ('Commitment Management', 'Impact', 'Social')] = extracted.get("Impact_Social", "")
            df_final.loc[index, ('Commitment Management', 'Impact', 'Economic')] = extracted.get("Impact_Economic", "")
            df_final.loc[index, ('Commitment Management', 'Impact', 'Environmental')] = extracted.get("Impact_Environmental", "")
            df_final.loc[index, ('Commitment Management', 'Impact', 'Regulatory')] = extracted.get("Impact_Regulatory", "")

        return df_final
