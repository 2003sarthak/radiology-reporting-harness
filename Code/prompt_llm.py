import os
import json
import re
from typing import Dict, Any, Tuple
from Code.template_editor import TemplateEditor

SYSTEM_PROMPT_STAGE1 = """You are an expert radiologist assistant. Your task is to edit a radiology report template based ONLY on transcribed radiologist dictation.

Core Rules:
1. Treat the provided template_content as the baseline report.
2. Route every dictated finding to the matching template field label (e.g., LUNGS, PLEURA, BONES, etc.).
3. Replace or modify normal statements ONLY when the dictation explicitly describes an abnormality in that field.
4. Keep template statements unchanged for routinely visualized regions NOT mentioned in the dictation.
5. Update the IMPRESSION to concisely summarize important abnormal findings.
6. Do NOT add findings unsupported by dictation.
7. Preserve exact wording, laterality (right/left), severity, and measurements (mm/cm).

Return your response strictly as valid JSON matching this schema:
{
  "field_updates": {
    "FIELD_KEY": "Updated sentence for this field"
  },
  "impression": "Concise summary of major abnormalities"
}
"""

def build_prompt_stage1(case: Dict[str, str]) -> str:
    """Build user prompt for Stage 1 case processing."""
    return f"""Case Information:
Modality: {case.get('modality', '')}
Body Part: {case.get('body_part', '')}
Study Description: {case.get('study_description', '')}
Patient Age Band: {case.get('patient_age_band', '')}
Patient Sex: {case.get('patient_sex', '')}

Normal Template:
{case.get('template_content', '')}

Dictation:
{case.get('dictation', '')}

Respond with valid JSON containing field_updates and impression:"""


class Stage1LLMPipeline:
    """
    Stage 1 LLM Pipeline supporting Google GenAI API / custom API / rule-based fallback.
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.client = None
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[Warning] Could not initialize Gemini client: {e}")

    def call_llm(self, prompt: str) -> str:
        """Execute LLM call using google-genai client if available."""
        if not self.client:
            return ""
        try:
            response = self.client.models.generate_content(
                model=config.MODEL_NAME,
                contents=prompt,
                config={
                    'system_instruction': SYSTEM_PROMPT_STAGE1,
                    'temperature': 0.1,
                    'response_mime_type': 'application/json'
                }
            )
            return response.text
        except Exception as e:
            print(f"[LLM Call Error] {e}")
            return ""

    def process_case_heuristic_fallback(self, case: Dict[str, str]) -> Dict[str, Any]:
        """
        Rule-based heuristic extraction fallback when no API key is available.
        Maps dictation terms to matching template fields using keyword matching.
        """
        template_content = case.get('template_content', '')
        dictation = case.get('dictation', '').strip()
        fields, default_impression = TemplateEditor.parse_template(template_content)

        field_updates = {}
        impression_parts = []

        if not dictation or dictation.lower() in ['normal', 'none', 'unremarkable', 'no acute abnormality']:
            return {'field_updates': {}, 'impression': default_impression}

        dictation_sentences = [s.strip() for s in re.split(r'[\.\;]\s*', dictation) if s.strip()]

        for sentence in dictation_sentences:
            s_lower = sentence.lower()
            routed = False
            
            for key, val in fields:
                k_lower = key.lower()
                
                # Check anatomical keywords matching field name
                keywords = k_lower.split()
                if any(kw in s_lower for kw in keywords if len(kw) > 3):
                    field_updates[key] = sentence.capitalize() if not sentence.endswith('.') else sentence
                    impression_parts.append(sentence)
                    routed = True
                    break
            
            if not routed:
                # Route to OTHER FINDINGS if present or fallback
                other_key = next((k for k, _ in fields if 'OTHER' in k.upper()), None)
                if other_key:
                    field_updates[other_key] = sentence
                impression_parts.append(sentence)

        impression = " ".join(impression_parts) if impression_parts else default_impression
        return {'field_updates': field_updates, 'impression': impression}

    def process_case(self, case: Dict[str, str]) -> Tuple[str, Dict[str, Any]]:
        """
        Processes a single case and returns the final rendered report and extracted updates.
        """
        template_content = case.get('template_content', '')
        
        if self.client:
            prompt = build_prompt_stage1(case)
            raw_response = self.call_llm(prompt)
            if raw_response:
                try:
                    data = json.loads(raw_response)
                    field_updates = data.get('field_updates', {})
                    impression = data.get('impression', '')
                    report = TemplateEditor.render_report(template_content, field_updates, impression)
                    return report, data
                except Exception as e:
                    print(f"[JSON Parse Error] {e}")

        # Fallback to rule-based heuristic extraction
        data = self.process_case_heuristic_fallback(case)
        report = TemplateEditor.render_report(template_content, data.get('field_updates', {}), data.get('impression', ''))
        return report, data
