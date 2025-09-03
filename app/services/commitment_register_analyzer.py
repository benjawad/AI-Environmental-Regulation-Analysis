import logging
import pandas as pd
import json
import google.generativeai as genai
import re
import time
from django.conf import settings
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CommitmentRegisterAnalyzer:
    def __init__(self, project_description):
        """Initialize with API key and project description."""
        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            print("Gemini API configured successfully.")
        except Exception as e:
            raise ValueError(f"Error configuring Gemini API: {e}")

        self.project_description = project_description
        self.columns = pd.MultiIndex.from_tuples(self._get_column_structure())

    def _get_column_structure(self):
        """Return the 3-level hierarchical column structure (identical to standalone script)."""
        return [
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
            ('Commitment Management', 'Requires Change Order?', '')
        ]

    def find_relevant_documents(self, commitment_description, knowledge_base):
        """Find documents relevant to the commitment description."""
        commitment_description = commitment_description.lower()
        keywords = set(re.findall(r'\b[a-zA-Zçéàèùâêîôûæœ\d]{4,}\b', commitment_description))
        relevant_texts = []
        
        for doc in knowledge_base:
            if any(keyword in doc['filename'].lower() for keyword in keywords) or \
               any(keyword in doc['content'] for keyword in keywords):
                relevant_texts.append(f"--- START OF RELEVANT DOCUMENT ({doc['filename']}) ---\n{doc['content']}\n--- END OF DOCUMENT ---\n")
        
        return "\n".join(relevant_texts)
    
    def call_gemini(self, commitment_row, relevant_texts):
        """Call Gemini model to complete a row with a detailed, structured prompt."""
        model = genai.GenerativeModel('gemini-1.5-flash')

        commitment_desc = commitment_row[('Commitment Register Overview', 'Description', '')]
        commitment_id = commitment_row[('Commitment Register Overview', 'Commitment Identifier', '')]

        # Enhanced prompt with more specific instructions and fallback logic
        prompt = f"""
        You are an expert Moroccan environmental and project management compliance analyst.
        Your task is to complete a row in a project's Commitment Register by synthesizing information from three sources: the project's description, the specific commitment, and relevant legal documents.

        **SOURCE 1: PROJECT DESCRIPTION**
        {self.project_description}

        **SOURCE 2: COMMITMENT CONTEXT**
        - Commitment ID: "{commitment_id}"
        - Commitment Description: "{commitment_desc}"

        **SOURCE 3: LEGAL EVIDENCE (Relevant Moroccan regulations and laws)**
        {relevant_texts if relevant_texts else "No specific legal documents found. Use general knowledge of Moroccan environmental regulations."}

        **TASK:**
        Based on ALL THREE sources provided, analyze how the commitment relates to the project phase, its objectives, and the legal requirements. Fill in ALL fields below. If specific information is not available, provide reasonable inferences based on the commitment type and context. Be concise and accurate. Output ONLY a valid JSON object, with no other text or markdown.

        **OUTPUT FORMAT (JSON ONLY):**
        {{
        "Impact or Hazard Addressed": "Identify the specific risk or hazard this commitment addresses. Example: 'Risk of air pollution from emissions exceeding legal limits during the operational phase.'",
        "Approving Agencies": "List the relevant government bodies. Example: 'Ministry of Energy Transition, Authorities coordinated by the Customer'",
        "Comments": "Provide a brief analysis connecting the commitment to the law and project phase. Example: 'As the project is in the FEED phase, this commitment ensures compliance with Law 13-03 is designed in from the start.'",
        "Affected_Preparation_Construction": "Enter 'x' if this phase is affected, otherwise ''",
        "Affected_Operation": "Enter 'x' if this phase is affected, otherwise ''",
        "Affected_Discharge_Management": "Enter 'x' if discharge management is affected, otherwise ''",
        "Impact_Health_Safety": "Enter 'x' if health and safety is impacted, otherwise ''",
        "Impact_Environmental": "Enter 'x' if environmental impact exists, otherwise ''",
        "Impact_Regulatory": "Enter 'x' if regulatory compliance is involved, otherwise ''"
        }}
        """
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                if not response or not response.text:
                    tqdm.write(f"Empty response from Gemini for '{commitment_id}' (attempt {attempt + 1})")
                    continue
                    
                # Clean the response to ensure it's valid JSON
                cleaned_response = response.text.strip()
                # Remove markdown code blocks
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response[7:]
                if cleaned_response.startswith('```'):
                    cleaned_response = cleaned_response[3:]
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-3]
                cleaned_response = cleaned_response.strip()
                
                # Try to parse JSON
                result = json.loads(cleaned_response)
                
                # Validate that we have the expected fields
                required_fields = [
                    "Impact or Hazard Addressed", "Approving Agencies", "Comments",
                    "Affected_Preparation_Construction", "Affected_Operation", "Affected_Discharge_Management",
                    "Impact_Health_Safety", "Impact_Environmental", "Impact_Regulatory"
                ]
                
                # Fill missing fields with empty strings
                for field in required_fields:
                    if field not in result:
                        result[field] = ""
                
                return result
                
            except json.JSONDecodeError as e:
                tqdm.write(f"JSON decode error for '{commitment_id}' (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    tqdm.write(f"Raw response: {response.text[:200]}...")
                    time.sleep(1)  # Wait before retry
                    continue
                else:
                    tqdm.write(f"Final attempt failed. Raw response: {response.text}")
                    
            except Exception as e:
                tqdm.write(f"Unexpected error for '{commitment_id}' (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
        
        # If all retries failed, return a default structure
        tqdm.write(f"All attempts failed for '{commitment_id}'. Using default values.")
        return {
            "Impact or Hazard Addressed": "",
            "Approving Agencies": "",
            "Comments": "",
            "Affected_Preparation_Construction": "",
            "Affected_Operation": "",
            "Affected_Discharge_Management": "",
            "Impact_Health_Safety": "",
            "Impact_Environmental": "",
            "Impact_Regulatory": ""
        }

    def load_knowledge_base(self, file_path: str):
        """
        Load parsed knowledge base from JSON file.
        Returns a Python object (list/dict) suitable for analysis.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.knowledge_base = data
            logger.info("Knowledge base loaded successfully: %s", file_path)
            return data
        except FileNotFoundError:
            logger.error("Knowledge base file not found: %s", file_path)
            raise
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON format in knowledge base: %s", e)
            raise

    def analyze_commitments(self, commitments_data, knowledge_base):
        """
        Main loop to analyze commitments with full schema.
        
        Args:
            commitments_data: List of lists containing initial commitment data (same format as standalone script)
            knowledge_base: Prepared knowledge base from load_and_prepare_knowledge_base()
        """
        
        logger.info(f"Columns defined: {len(self.columns)} → {self.columns}")
        logger.info(f"Number of commitment rows: {len(commitments_data)}")
        if commitments_data:
            logger.info(f"First row of commitments_data: {commitments_data[0]}")
            logger.info(f"Number of fields in first row: {len(commitments_data[0])}")

        # Create DataFrame with the predefined structure and initial data
        df_initial = pd.DataFrame(commitments_data, columns=self.columns)
        df_final = df_initial.copy()

        # Progress bar for commitment analysis
        for index, row in tqdm(df_initial.iterrows(), total=len(df_initial), desc="Analyzing Commitments"):
            commitment_description = row[('Commitment Register Overview', 'Description', '')]
            commitment_id = row[('Commitment Register Overview', 'Commitment Identifier', '')]
            
            relevant_texts = self.find_relevant_documents(commitment_description, knowledge_base)

            if not relevant_texts:
                tqdm.write(f"  -> No relevant documents for '{commitment_id}'. Skipping.")
                continue

            extracted_data = self.call_gemini(row, relevant_texts)
            time.sleep(2)  # API rate limit

            if extracted_data:
                # Fill in the extracted data (matching the standalone script exactly)
                df_final.loc[index, ('Commitment Management', 'Impact or Hazard Addressed', '')] = extracted_data.get("Impact or Hazard Addressed", "")
                df_final.loc[index, ('Commitment Management', 'Approving Agencies', '')] = extracted_data.get("Approving Agencies", "")
                df_final.loc[index, ('Commitment Management', 'Comments', '')] = extracted_data.get("Comments", "")
                df_final.loc[index, ('Commitment Management', 'Affected Areas or Processes', 'Preparation/construction')] = extracted_data.get("Affected_Preparation_Construction", "")
                df_final.loc[index, ('Commitment Management', 'Affected Areas or Processes', 'Operation')] = extracted_data.get("Affected_Operation", "")
                df_final.loc[index, ('Commitment Management', 'Affected Areas or Processes', 'Input Management')] = extracted_data.get("Affected_Input_Management", "")
                df_final.loc[index, ('Commitment Management', 'Affected Areas or Processes', 'Discharge management')] = extracted_data.get("Affected_Discharge_Management", "")
                df_final.loc[index, ('Commitment Management', 'Affected Areas or Processes', 'Off-Sites')] = extracted_data.get("Affected_Off_Sites", "")
                df_final.loc[index, ('Commitment Management', 'Affected Areas or Processes', 'Other')] = extracted_data.get("Affected_Other", "")
                df_final.loc[index, ('Commitment Management', 'Affected Areas or Processes', 'Fungibility')] = extracted_data.get("Affected_Fungibility", "")
                df_final.loc[index, ('Commitment Management', 'Impact', 'CAPEX')] = extracted_data.get("Impact_CAPEX", "")
                df_final.loc[index, ('Commitment Management', 'Impact', 'OPEX')] = extracted_data.get("Impact_OPEX", "")
                df_final.loc[index, ('Commitment Management', 'Impact', 'Health & Safety')] = extracted_data.get("Impact_Health_Safety", "")
                df_final.loc[index, ('Commitment Management', 'Impact', 'Social')] = extracted_data.get("Impact_Social", "")
                df_final.loc[index, ('Commitment Management', 'Impact', 'Economic')] = extracted_data.get("Impact_Economic", "")
                df_final.loc[index, ('Commitment Management', 'Impact', 'Environmental')] = extracted_data.get("Impact_Environmental", "")
                df_final.loc[index, ('Commitment Management', 'Impact', 'Regulatory')] = extracted_data.get("Impact_Regulatory", "")
            else:
                tqdm.write(f"  -> Failed to get a valid response from Gemini for '{commitment_id}'.")

        return df_final
    

