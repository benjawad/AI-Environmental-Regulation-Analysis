import fitz  # PyMuPDF
import pandas as pd
import re
import os
import io
import traceback
import numpy as np
from PIL import Image
import pytesseract
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Configuration du logging
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
    Parser PDF de production robuste qui adresse les problèmes identifiés :
    - Stratégie multi-approche pour les différents types de documents
    - Gestion robuste des mises en page complexes
    - OCR intégré avec détection automatique
    - Validation stricte des tableaux pour éviter les hallucinations
    - Classification intelligente des documents
    """

    # Configurations pour la détection des polluants (enrichie)
    POLLUTANT_MAPPING = {
        'SO2': ['dioxyde de soufre', 'so2', 'sulphur dioxide', 'anhydride sulfureux'],
        'NOx': ['oxydes d\'azote', 'nox', 'nitrogen oxides', 'monoxyde d\'azote', 'dioxyde d\'azote'],
        'COV': ['composés organiques volatils', 'cov', 'volatile organic compounds', 'voc'],
        'PM10': ['particules pm10', 'pm10', 'particulate matter 10', 'poussières pm10'],
        'PM2.5': ['particules pm2.5', 'pm2.5', 'particules fines'],
        'Hg': ['mercure', 'mercury', 'hg'],
        'Pb': ['plomb', 'lead', 'pb'],
        'Cd': ['cadmium', 'cd'],
        'O3': ['ozone', 'o3'],
        'CO': ['monoxyde de carbone', 'co', 'carbon monoxide'],
        'Benzène': ['benzène', 'benzene', 'c6h6'],
        'H2S': ['sulfure d\'hydrogène', 'h2s', 'hydrogen sulfide'],
        'NH3': ['ammoniac', 'nh3', 'ammonia'],
        'Fluorures': ['fluorures', 'fluorides', 'hf'],
        'Chlorures': ['chlorures', 'chlorides', 'hcl']
    }

    def __init__(self, pdf_path: str, config: Optional[TableValidationConfig] = None):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Fichier introuvable: {pdf_path}")

        self.pdf_path = pdf_path
        self.filename = os.path.basename(pdf_path)
        self.doc = fitz.open(pdf_path)
        self.config = config or TableValidationConfig()

        # Analyse préliminaire du document
        self.doc_analysis = self._analyze_document_structure()
        self.doc_type = self._classify_document()
        self.metadata = self._extract_metadata()

    def _analyze_document_structure(self) -> Dict[str, Any]:
        """Analyse la structure générale du document"""
        analysis = {
            'total_pages': len(self.doc),
            'scanned_pages': 0,
            'text_pages': 0,
            'table_pages': 0,
            'mixed_pages': 0,
            'avg_text_density': 0,
            'has_images': False
        }

        text_densities = []

        for page_num in range(min(5, len(self.doc))):  # Analyse des 5 premières pages
            page = self.doc.load_page(page_num)

            # Analyse du texte
            text = page.get_text()
            text_density = len(text.strip()) / (page.rect.width * page.rect.height) * 10000
            text_densities.append(text_density)

            # Détection du type de page
            if self._is_scanned_page(page):
                analysis['scanned_pages'] += 1
            elif self._has_complex_tables(page):
                analysis['table_pages'] += 1
            elif text_density > 5:
                analysis['text_pages'] += 1
            else:
                analysis['mixed_pages'] += 1

            # Détection d'images
            if page.get_images():
                analysis['has_images'] = True

        analysis['avg_text_density'] = np.mean(text_densities) if text_densities else 0
        return analysis

    def _classify_document(self) -> str:
        """Classification intelligente basée sur le nom et le contenu"""
        filename_lower = self.filename.lower()

        # Classification par nom de fichier
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

        # Classification par contenu (analyse des premières pages)
        content_keywords = self._extract_content_keywords()
        if 'valeur' in content_keywords and 'limite' in content_keywords:
            if 'atmosphérique' in content_keywords or 'air' in content_keywords:
                return 'vlg_atmospherique'
            elif 'liquide' in content_keywords or 'eau' in content_keywords:
                return 'vlg_liquide'

        return 'autre'

    def _extract_content_keywords(self) -> List[str]:
        """Extrait les mots-clés des premières pages pour la classification"""
        keywords = []
        for page_num in range(min(3, len(self.doc))):
            page = self.doc.load_page(page_num)
            text = page.get_text().lower()

            # Extraction de mots-clés pertinents
            key_terms = [
                'valeur', 'limite', 'norme', 'seuil', 'décret',
                'atmosphérique', 'air', 'liquide', 'eau',
                'ciment', 'céramique', 'textile', 'sucre',
                'pollution', 'émission', 'rejet', 'qualité'
            ]

            for term in key_terms:
                if term in text:
                    keywords.append(term)

        return keywords

    def _extract_metadata(self) -> Dict[str, Any]:
        """Extraction enrichie des métadonnées"""
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
        """Convertit les dates PDF en format ISO"""
        if not date_str:
            return ""
        try:
            if date_str.startswith("D:") and len(date_str) >= 10:
                return f"{date_str[2:6]}-{date_str[6:8]}-{date_str[8:10]}"
        except Exception:
            pass
        return date_str

    def _is_scanned_page(self, page) -> bool:
        """Détection améliorée des pages scannées"""
        # 1. Densité de texte très faible
        text = page.get_text()
        if len(text.strip()) > 500:  # Seuil plus élevé
            return False

        # 2. Présence d'images de grande taille
        images = page.get_images()
        if not images:
            return False

        for img in images:
            # Vérification des dimensions et de la résolution
            if img[2] > 400 and img[3] > 400:  # Largeur et hauteur minimales
                return True

        # 3. Rapport texte/surface très faible
        text_density = len(text.strip()) / (page.rect.width * page.rect.height) * 10000
        return text_density < 2

    def _has_complex_tables(self, page) -> bool:
        """Détecte si une page contient des tableaux complexes"""
        try:
            tables = page.find_tables()
            if not tables:
                return False

            # Vérification de la complexité des tableaux
            for table in tables:
                df = table.to_pandas()
                if len(df.columns) > 3 and len(df) > 3:
                    return True
            return False
        except Exception:
            return False

    def _ocr_page(self, page) -> str:
        """OCR robuste avec gestion d'erreurs"""
        try:
            # Extraction avec haute résolution
            pix = page.get_pixmap(dpi=300)
            img_data = pix.tobytes()

            if not img_data:
                return ""

            img = Image.open(io.BytesIO(img_data))

            # Configuration OCR optimisée
            custom_config = r'--oem 3 --psm 6'

            # Essai avec différentes langues
            languages = ['fra', 'ara+fra', 'eng']

            for lang in languages:
                try:
                    text = pytesseract.image_to_string(
                        img,
                        lang=lang,
                        config=custom_config
                    )
                    if text.strip() and len(text.strip()) > 20:
                        return self._clean_text(text)
                except Exception:
                    continue

            # Dernier recours sans spécification de langue
            text = pytesseract.image_to_string(img, config=custom_config)
            return self._clean_text(text)

        except Exception as e:
            logger.warning(f"Échec OCR page: {str(e)}")
            return ""

    def _clean_text(self, text: str) -> str:
        """Nettoyage approfondi du texte"""
        if not text:
            return ""

        # Remplacement des caractères problématiques
        replacements = {
            '\xad': '',      # Soft hyphen
            '\uf0b7': '•',   # Bullet
            'ﬁ': 'fi',
            'ﬂ': 'fl',
            '\u200b': '',    # Zero-width space
            '\u202f': ' ',   # Narrow no-break space
            '\ufeff': '',    # BOM
        }

        for char, replacement in replacements.items():
            text = text.replace(char, replacement)

        # Normalisation des espaces
        text = re.sub(r'\s+', ' ', text).strip()

        # Suppression des lignes très courtes (souvent du bruit OCR)
        lines = text.split('\n')
        cleaned_lines = [line for line in lines if len(line.strip()) > 2]

        return '\n'.join(cleaned_lines)

    def _extract_tables_robust(self, page) -> List[Dict[str, Any]]:
        """Extraction robuste des tableaux avec validation stricte"""
        tables = []

        try:
            # Tentative d'extraction des tableaux
            found_tables = page.find_tables(
                strategy="lines_strict",  # Plus strict pour éviter les hallucinations
                snap_tolerance=3.0
            )

            if not found_tables:
                # Tentative avec une stratégie alternative
                found_tables = page.find_tables(strategy="explicit")

            for i, table in enumerate(found_tables):
                try:
                    df = table.to_pandas()

                    # Validation stricte du tableau
                    if not self._validate_table_strict(df, page):
                        logger.debug(f"Tableau {i} rejeté - validation échoué")
                        continue

                    # Nettoyage et traitement des cellules fusionnées
                    df_cleaned = self._process_merged_cells(df)

                    # Structure finale du tableau
                    table_data = {
                        "table_id": i,
                        "bbox": table.bbox,
                        "header": df_cleaned.columns.tolist(),
                        "rows": df_cleaned.fillna("").values.tolist(),
                        "shape": df_cleaned.shape,
                        "confidence": self._calculate_table_confidence(df_cleaned)
                    }

                    tables.append(table_data)

                except Exception as e:
                    logger.warning(f"Erreur traitement tableau {i}: {str(e)}")
                    continue

        except Exception as e:
            logger.warning(f"Erreur extraction tableaux: {str(e)}")

        return tables

    def _validate_table_strict(self, df: pd.DataFrame, page) -> bool:
        """Validation stricte pour éviter les hallucinations de tableaux"""
        # 1. Vérification des dimensions
        if len(df.columns) > self.config.max_columns:
            logger.debug(f"Trop de colonnes: {len(df.columns)}")
            return False

        if len(df) < self.config.min_rows:
            logger.debug(f"Pas assez de lignes: {len(df)}")
            return False

        # 2. Vérification du taux de cellules vides
        null_ratio = df.isnull().sum().sum() / df.size
        if null_ratio > self.config.max_null_percentage:
            logger.debug(f"Trop de cellules vides: {null_ratio:.2%}")
            return False

        # 3. Vérification du contenu significatif
        text_cells = 0
        total_cells = df.size

        for col in df.columns:
            for value in df[col]:
                if pd.notna(value) and str(value).strip():
                    text_cells += 1

        content_ratio = text_cells / total_cells
        if content_ratio < self.config.min_content_ratio:
            logger.debug(f"Ratio de contenu trop faible: {content_ratio:.2%}")
            return False

        # 4. Vérification de la cohérence des colonnes
        if self._has_incoherent_columns(df):
            logger.debug("Colonnes incohérentes détectées")
            return False

        return True

    def _has_incoherent_columns(self, df: pd.DataFrame) -> bool:
        """Détecte les colonnes incohérentes (signe d'hallucination)"""
        for col in df.columns:
            # Vérification si une colonne contient principalement des fragments
            values = df[col].dropna().astype(str)
            if len(values) > 0:
                # Si plus de 80% des valeurs font moins de 3 caractères, c'est suspect
                short_values = sum(1 for v in values if len(v.strip()) < 3)
                if short_values / len(values) > 0.8:
                    return True

        return False

    def _process_merged_cells(self, df: pd.DataFrame) -> pd.DataFrame:
        """Traitement intelligent des cellules fusionnées"""
        df_copy = df.copy()

        # Forward fill pour les colonnes d'en-tête (première colonne généralement)
        if len(df_copy.columns) > 0:
            df_copy.iloc[:, 0] = df_copy.iloc[:, 0].ffill()

        # Traitement spécifique pour les tableaux de normes
        if self._is_standards_table(df_copy):
            df_copy = self._process_standards_table(df_copy)

        return df_copy

    def _is_standards_table(self, df: pd.DataFrame) -> bool:
        """Détecte si c'est un tableau de normes/valeurs limites"""
        # Recherche de mots-clés typiques
        keywords = ['polluant', 'limite', 'valeur', 'unité', 'µg/m³', 'mg/l']
        text_content = ' '.join([str(col) for col in df.columns]).lower()

        return any(keyword in text_content for keyword in keywords)

    def _process_standards_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Traitement spécialisé pour les tableaux de normes"""
        # Logic spécifique pour les tableaux de valeurs limites
        df_processed = df.copy()

        # Propagation des valeurs dans les cellules fusionnées
        for col_idx in range(len(df_processed.columns)):
            df_processed.iloc[:, col_idx] = df_processed.iloc[:, col_idx].ffill()

        return df_processed

    def _calculate_table_confidence(self, df: pd.DataFrame) -> float:
        """Calcule un score de confiance pour le tableau"""
        score = 1.0

        # Pénalité pour les cellules vides
        null_ratio = df.isnull().sum().sum() / df.size
        score -= null_ratio * 0.5

        # Bonus pour la cohérence des types de données
        for col in df.columns:
            values = df[col].dropna()
            if len(values) > 0:
                # Vérification de la cohérence des types
                numeric_count = sum(1 for v in values if str(v).replace('.', '').replace(',', '').isdigit())
                if numeric_count / len(values) > 0.7:  # Colonne majoritairement numérique
                    score += 0.1

        return max(0.0, min(1.0, score))

    def _extract_structured_text(self, page) -> Dict[str, Any]:
        """Extraction de texte structuré sans bruit excessif"""
        # Utilisation d'une approche par blocs plutôt que par span
        blocks = page.get_text("blocks")

        structured_content = {
            "title_text": "",
            "body_text": "",
            "headers": [],
            "paragraphs": []
        }

        for block in blocks:
            if len(block) >= 5:  # Structure valide
                text = block[4].strip()
                if not text:
                    continue

                # Classification basique du contenu
                if self._is_title_text(text, block):
                    structured_content["headers"].append(text)
                    if not structured_content["title_text"]:
                        structured_content["title_text"] = text
                elif len(text) > 50:  # Paragraphe substantiel
                    structured_content["paragraphs"].append(text)
                    structured_content["body_text"] += text + "\n"

        return structured_content

    def _is_title_text(self, text: str, block: tuple) -> bool:
        """Détecte si un texte est un titre"""
        # Heuristiques simples pour détecter les titres
        if len(text) > 100:  # Trop long pour être un titre
            return False

        if text.isupper() or text.istitle():
            return True

        # Détection basée sur des mots-clés
        title_keywords = ['article', 'chapitre', 'section', 'annexe', 'tableau']
        return any(keyword in text.lower() for keyword in title_keywords)

    def _detect_pollutants_enhanced(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Détection améliorée des polluants avec contexte"""
        found_pollutants = {}
        text_lower = text.lower()

        for code, names in self.POLLUTANT_MAPPING.items():
            for name in names:
                if name in text_lower:
                    # Extraction du contexte autour du polluant
                    context = self._extract_context(text_lower, name)

                    found_pollutants[code] = {
                        "name": name,
                        "matched_term": name,
                        "context": context,
                        "has_limit_value": self._has_associated_limit(context)
                    }
                    break  # Un seul match par polluant

        return found_pollutants

    def _extract_context(self, text: str, term: str, window: int = 100) -> str:
        """Extrait le contexte autour d'un terme"""
        pos = text.find(term)
        if pos == -1:
            return ""

        start = max(0, pos - window)
        end = min(len(text), pos + len(term) + window)

        return text[start:end].strip()

    def _has_associated_limit(self, context: str) -> bool:
        """Vérifie si le contexte contient une valeur limite"""
        # Recherche de patterns numériques avec unités
        limit_pattern = r'\d+[\d\s,.]*\s*(µg/m³|mg/m³|mg/l|µg/l|ppm|ppb)'
        return bool(re.search(limit_pattern, context))

    def _extract_limit_values_enhanced(self, text: str) -> List[Dict[str, Any]]:
        """Extraction améliorée des valeurs limites"""
        # Pattern plus sophistiqué pour les valeurs limites
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

    def _process_page_with_strategy(self, page_num: int) -> Dict[str, Any]:
        """Traite une page avec la stratégie appropriée"""
        page = self.doc.load_page(page_num)

        page_data = {
            "page_number": page_num + 1,
            "dimensions": {"width": page.rect.width, "height": page.rect.height},
            "strategy_used": "",
            "content": {},
            "confidence": 0.0
        }

        # Stratégie 1: Page scannée -> OCR
        if self._is_scanned_page(page):
            page_data["strategy_used"] = "ocr"
            ocr_text = self._ocr_page(page)

            if ocr_text:
                page_data["content"] = {
                    "text": ocr_text,
                    "pollutants": self._detect_pollutants_enhanced(ocr_text),
                    "limit_values": self._extract_limit_values_enhanced(ocr_text)
                }
                page_data["confidence"] = 0.6  # OCR moins fiable
            else:
                page_data["content"] = {"error": "Échec OCR"}
                page_data["confidence"] = 0.0

        # Stratégie 2: Page avec tableaux complexes
        elif self._has_complex_tables(page):
            page_data["strategy_used"] = "table_extraction"

            tables = self._extract_tables_robust(page)
            text_content = self._extract_structured_text(page)

            full_text = text_content["body_text"]

            page_data["content"] = {
                "tables": tables,
                "text_structure": text_content,
                "pollutants": self._detect_pollutants_enhanced(full_text),
                "limit_values": self._extract_limit_values_enhanced(full_text)
            }

            # Calcul de confiance basé sur les tableaux
            if tables:
                avg_confidence = np.mean([t["confidence"] for t in tables])
                page_data["confidence"] = avg_confidence
            else:
                page_data["confidence"] = 0.3

        # Stratégie 3: Page textuelle standard
        else:
            page_data["strategy_used"] = "text_extraction"
            text_content = self._extract_structured_text(page)
            full_text = text_content["body_text"]

            if full_text.strip():
                page_data["content"] = {
                    "text_structure": text_content,
                    "pollutants": self._detect_pollutants_enhanced(full_text),
                    "limit_values": self._extract_limit_values_enhanced(full_text)
                }
                page_data["confidence"] = 0.9
            else:
                page_data["content"] = {"error": "Aucun contenu textuel significatif"}
                page_data["confidence"] = 0.1

        return page_data

    def parse(self) -> Dict[str, Any]:
        """Analyse complète du document avec stratégies adaptatives"""
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

        # Traitement des pages
        confidences = []
        strategies = {}
        errors = 0

        for page_num in range(len(self.doc)):
            try:
                page_data = self._process_page_with_strategy(page_num)
                doc_data["pages"].append(page_data)

                # Collecte des statistiques
                confidences.append(page_data["confidence"])
                strategy = page_data["strategy_used"]
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

        # Résumé de l'analyse
        doc_data["analysis_summary"].update({
            "strategies_used": strategies,
            "avg_confidence": np.mean(confidences) if confidences else 0.0,
            "processing_errors": errors
        })

        # Analyse globale des polluants
        doc_data["global_analysis"] = self._analyze_document_globally(doc_data)

        logger.info(f"Analyse terminée - Confiance: {doc_data['analysis_summary']['avg_confidence']:.2f}")

        return doc_data

    def _analyze_document_globally(self, doc_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse globale du document"""
        global_pollutants = {}
        global_limits = []

        for page in doc_data["pages"]:
            if "content" in page and isinstance(page["content"], dict):
                # Agrégation des polluants
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

                # Agrégation des valeurs limites
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
        """Évalue la qualité de l'extraction"""
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
        """Génère des recommandations pour améliorer l'extraction"""
        recommendations = []

        # Analyse des stratégies utilisées
        strategies = doc_data["analysis_summary"]["strategies_used"]
        errors = doc_data["analysis_summary"]["processing_errors"]

        if strategies.get("ocr", 0) > strategies.get("text_extraction", 0):
            recommendations.append("Document principalement scanné - considérer une version native si disponible")

        if errors > 0:
            recommendations.append(f"{errors} pages ont échoué - vérifier la qualité du PDF")

        if doc_data["analysis_summary"]["avg_confidence"] < 0.6:
            recommendations.append("Confiance faible - révision manuelle recommandée")

        return recommendations


def process_pdf_batch(pdf_files: List[str], config: Optional[TableValidationConfig] = None) -> Dict[str, Any]:
    """Traite un batch de fichiers PDF avec rapports détaillés"""
    results = {}
    processing_summary = {
        "total_files": len(pdf_files),
        "successful": 0,
        "failed": 0,
        "avg_confidence": 0.0,
        "processing_time": 0.0
    }

    import time
    start_time = time.time()

    for pdf_path in pdf_files:
        try:
            logger.info(f"Traitement: {pdf_path}")

            # Vérification de l'existence du fichier
            if not os.path.exists(pdf_path):
                logger.error(f"Fichier introuvable: {pdf_path}")
                results[pdf_path] = {"error": "Fichier introuvable"}
                processing_summary["failed"] += 1
                continue

            # Traitement du PDF
            parser = PDFParser(pdf_path, config)
            result = parser.parse()
            results[pdf_path] = result

            # Mise à jour des statistiques
            processing_summary["successful"] += 1
            processing_summary["avg_confidence"] += result["analysis_summary"]["avg_confidence"]

            # Affichage des résultats (via logs)
            doc_type = result["document_type"]
            strategies = result["analysis_summary"]["strategies_used"]
            confidence = result["analysis_summary"]["avg_confidence"]
            pollutants_count = len(result["global_analysis"]["pollutants_summary"])

            logger.info(f"✓ Succès: {result['metadata']['pages']} pages")
            logger.info(f"  Type: {doc_type}")
            logger.info(f"  Stratégies: {strategies}")
            logger.info(f"  Confiance: {confidence:.2f}")
            logger.info(f"  Polluants: {pollutants_count}")

            # Affichage des recommandations
            recommendations = result["global_analysis"]["extraction_recommendations"]
            if recommendations:
                logger.warning("  Recommandations:")
                for rec in recommendations:
                    logger.warning(f"    - {rec}")

        except Exception as e:
            logger.error(f"✗ Erreur critique: {pdf_path} - {str(e)}")
            results[pdf_path] = {"error": str(e), "traceback": traceback.format_exc()}
            processing_summary["failed"] += 1

    # Finalisation des statistiques
    processing_summary["processing_time"] = time.time() - start_time
    if processing_summary["successful"] > 0:
        processing_summary["avg_confidence"] /= processing_summary["successful"]

    return {
        "results": results,
        "summary": processing_summary
    }


def validate_parsing_setup():
    """Valide la configuration de l'environnement de parsing"""
    validation_results = {
        "tesseract_available": False,
        "tesseract_languages": [],
        "pymupdf_version": None,
        "pandas_version": None,
        "recommendations": []
    }

    # Vérification de Tesseract
    try:
        _ = pytesseract.get_tesseract_version()
        validation_results["tesseract_available"] = True

        # Test des langues disponibles
        try:
            langs = pytesseract.get_languages(config='')
            validation_results["tesseract_languages"] = langs
        except Exception:
            validation_results["tesseract_languages"] = ["eng"]  # Par défaut

    except Exception:
        validation_results["recommendations"].append(
            "Tesseract non trouvé - installer avec: apt-get install tesseract-ocr tesseract-ocr-fra"
        )

    # Vérification des versions des dépendances
    try:
        validation_results["pymupdf_version"] = fitz.__version__
        validation_results["pandas_version"] = pd.__version__
    except Exception:
        pass

    # Recommandations additionnelles
    if "fra" not in validation_results["tesseract_languages"]:
        validation_results["recommendations"].append(
            "Pack français non trouvé - installer avec: apt-get install tesseract-ocr-fra"
        )

    return validation_results


# Configuration d'exemple pour différents types de documents
class DocumentConfigs:
    """Configurations prédéfinies pour différents types de documents"""

    @staticmethod
    def get_config(doc_type: str) -> TableValidationConfig:
        """Retourne une configuration optimisée selon le type de document"""
        configs = {
            "vlg_atmospherique": TableValidationConfig(
                max_columns=8,
                min_rows=3,
                max_null_percentage=0.3,
                min_content_ratio=0.4
            ),
            "vlg_liquide": TableValidationConfig(
                max_columns=10,
                min_rows=5,
                max_null_percentage=0.2,
                min_content_ratio=0.5
            ),
            "normes_air": TableValidationConfig(
                max_columns=6,
                min_rows=4,
                max_null_percentage=0.1,
                min_content_ratio=0.6
            ),
            "vls_ciment": TableValidationConfig(
                max_columns=12,
                min_rows=3,
                max_null_percentage=0.4,
                min_content_ratio=0.3
            ),
            "default": TableValidationConfig()
        }

        return configs.get(doc_type, configs["default"])
