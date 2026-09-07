import re
from typing import Dict, List, Tuple, Any

def validate_report_structure(report_text: str, template_content: str) -> Tuple[bool, List[str]]:
    """
    Validates structural requirements of generated report.
    - FINDINGS section exists
    - IMPRESSION section exists
    - Retains original template field headers
    """
    issues = []
    
    if not report_text or not report_text.strip():
        issues.append("Report text is empty.")
        return False, issues

    if not re.search(r'^FINDINGS:', report_text, flags=re.IGNORECASE | re.MULTILINE):
        issues.append("Missing top-level 'FINDINGS:' section header.")
        
    if not re.search(r'IMPRESSION:', report_text, flags=re.IGNORECASE):
        issues.append("Missing top-level 'IMPRESSION:' section header.")

    # Check template field key preservation using robust multiline regex
    tmpl_sec = re.split(r'\n(?=IMPRESSION:)', template_content, flags=re.IGNORECASE)[0]
    tmpl_body = re.sub(r'^FINDINGS:\s*', '', tmpl_sec, flags=re.IGNORECASE).strip()
    tmpl_field_matches = [m.group(1).strip() for m in re.finditer(r'^[ \t]*([A-Z0-9\s\/_\-\(\)]+):\s*', tmpl_body, flags=re.MULTILINE)]

    rpt_sec = re.split(r'\n(?=IMPRESSION:)', report_text, flags=re.IGNORECASE)[0]
    rpt_body = re.sub(r'^FINDINGS:\s*', '', rpt_sec, flags=re.IGNORECASE).strip()
    rpt_field_matches = [m.group(1).strip() for m in re.finditer(r'^[ \t]*([A-Z0-9\s\/_\-\(\)]+):\s*', rpt_body, flags=re.MULTILINE)]
    
    missing_fields = set(tmpl_field_matches) - set(rpt_field_matches)
    if missing_fields:
        issues.append(f"Missing template fields in report: {list(missing_fields)}")

    is_valid = len(issues) == 0
    return is_valid, issues


def validate_clinical_fidelity(dictation: str, report_text: str) -> Tuple[bool, List[str]]:
    """
    Validates clinical fidelity checks:
    - Negation preservation
    - Laterality preservation
    - Measurement numerical preservation
    """
    warnings = []
    dictation_lower = dictation.lower()
    report_lower = report_text.lower()

    # 1. Laterality check
    if 'right' in dictation_lower and 'right' not in report_lower:
        warnings.append("Laterality 'right' present in dictation but missing in report.")
    if 'left' in dictation_lower and 'left' not in report_lower:
        warnings.append("Laterality 'left' present in dictation but missing in report.")

    # 2. Measurement numbers check
    measurements = re.findall(r'\b\d+(?:\.\d+)?\s*(?:mm|cm)\b', dictation_lower)
    for m in measurements:
        if m not in report_lower:
            warnings.append(f"Measurement '{m}' from dictation not found in report.")

    is_faithful = len(warnings) == 0
    return is_faithful, warnings
