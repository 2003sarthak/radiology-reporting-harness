import json
from typing import Dict, List, Any
from Code.template_editor import TemplateEditor

SYSTEM_PROMPT_RAG = """You are an expert radiologist assistant. Your task is to convert radiologist dictation into a minimally edited, template-complete report.

Core Rules:
1. Treat template_content as the baseline starting report.
2. Route dictated findings to the matching template field label (e.g. LUNGS, PLEURA, BONES, etc.).
3. Replace or modify normal statements ONLY when dictation describes an abnormality in that field.
4. Keep template statements unchanged for routinely visualized regions NOT mentioned in dictation.
5. Update IMPRESSION to concisely summarize important abnormal findings.
6. Preserve exact wording, laterality (right/left), severity, and numerical measurements (mm/cm).

Return your response strictly as valid JSON matching this schema:
{
  "field_updates": {
    "FIELD_KEY": "Updated sentence replacing normal statement"
  },
  "impression": "Concise clinical summary"
}
"""

def extract_reference_updates(ref_case: Dict[str, str]) -> Dict[str, Any]:
    """
    Extracts the changed field updates and impression from a training reference case.
    """
    tmpl_content = ref_case.get('template_content', '')
    ref_report = ref_case.get('report', '')
    
    tmpl_fields, tmpl_imp = TemplateEditor.parse_template(tmpl_content)
    ref_fields, ref_imp = TemplateEditor.parse_template(ref_report)
    
    tmpl_dict = dict(tmpl_fields)
    
    field_updates = {}
    for k, ref_val in ref_fields:
        tmpl_val = tmpl_dict.get(k, '')
        if ref_val.strip() != tmpl_val.strip():
            field_updates[k] = ref_val.strip()
            
    return {
        "field_updates": field_updates,
        "impression": ref_imp.strip()
    }


def build_rag_prompt(query_case: Dict[str, str], retrieved_examples: List[Dict[str, str]]) -> str:
    """
    Constructs a dynamic few-shot prompt using top retrieved reference cases.
    """
    prompt = "Below are examples showing how past dictations were incorporated into templates:\n\n"
    
    for i, ex in enumerate(retrieved_examples, 1):
        updates = extract_reference_updates(ex)
        prompt += f"--- EXAMPLE {i} ---\n"
        prompt += f"Modality: {ex.get('modality', '')} | Body Part: {ex.get('body_part', '')}\n"
        prompt += f"Template:\n{ex.get('template_content', '')}\n\n"
        prompt += f"Dictation:\n{ex.get('dictation', '')}\n\n"
        prompt += f"Expected Report Updates (JSON):\n{json.dumps(updates, indent=2)}\n\n"

    prompt += "=== TARGET CASE TO PROCESS ===\n"
    prompt += f"Modality: {query_case.get('modality', '')} | Body Part: {query_case.get('body_part', '')}\n"
    prompt += f"Study Description: {query_case.get('study_description', '')}\n"
    prompt += f"Template:\n{query_case.get('template_content', '')}\n\n"
    prompt += f"Dictation:\n{query_case.get('dictation', '')}\n\n"
    prompt += "Generate the target case field_updates and impression strictly in valid JSON format:"
    
    return prompt
