from transformers import AutoProcessor, AutoModelForTokenClassification
import torch
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
            r'(\d+(?:[,.\s]\d+)*)\s*(µg/m³|mg/m³|mg/l|µg/l|ng/m³|ppm|ppb|°C|%)',
            r'(\d+(?:[,.\s]\d+)*)\s*(microgrammes?|milligrammes?|nanogrammes?)',
            r'(\d+(?:[,.\s]\d+)*)\s*(?:µg|mg|ng|ppm|ppb)'
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