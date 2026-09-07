import re
from typing import Dict, List, Tuple
from Code.dataset import normalize_text, tokenize

# Token Weight Classification
CRITICAL_WORDS = {
    # Negation & Acuity / Severity
    'no', 'not', 'without', 'absent', 'denies', 'negative', 'none',
    'acute', 'chronic', 'subacute', 'mild', 'moderate', 'severe', 'marked', 'gross',
    # Laterality & Anatomy Markers
    'right', 'left', 'bilateral', 'anterior', 'posterior', 'superior', 'inferior',
    'medial', 'lateral', 'proximal', 'distal', 'dorsal', 'ventral', 'upper', 'lower',
    # Units
    'mm', 'cm', 'm', 'kg', 'mg', 'percent', 'degree', 'degrees'
}

FUNCTION_WORDS = {
    'the', 'and', 'of', 'with', 'in', 'on', 'at', 'to', 'a', 'an', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'by', 'for', 'from', 'or', 'as', 'into',
    'that', 'this', 'these', 'those', 'there', 'has', 'have', 'had'
}

def get_token_weight(token: str) -> float:
    """Assign token weight: 4.0 for critical/numbers/units, 0.25 for function words, 2.0 for content."""
    if not token:
        return 0.0
    if token in CRITICAL_WORDS or re.match(r'^[+-]?\d+(\.\d+)?$', token):
        return 4.0
    if token in FUNCTION_WORDS:
        return 0.25
    return 2.0


def weighted_token_edit_distance(ref_text: str, pred_text: str) -> float:
    """
    Computes ordered word-level Levenshtein edit distance with token weights.
    Returns edit distance score between 0.0 and 1.0.
    """
    ref_tokens = tokenize(ref_text)
    pred_tokens = tokenize(pred_text)
    
    if not ref_tokens and not pred_tokens:
        return 0.0
    if not ref_tokens or not pred_tokens:
        return 1.0

    n, m = len(ref_tokens), len(pred_tokens)
    ref_weights = [get_token_weight(t) for t in ref_tokens]
    pred_weights = [get_token_weight(t) for t in pred_tokens]

    total_ref_weight = sum(ref_weights)
    total_pred_weight = sum(pred_weights)
    max_total_weight = max(total_ref_weight, total_pred_weight)

    if max_total_weight == 0:
        return 0.0

    # Dynamic Programming Matrix for Weighted Levenshtein
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = dp[i-1][0] + ref_weights[i-1]
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j-1] + pred_weights[j-1]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i-1] == pred_tokens[j-1]:
                cost = 0.0
            else:
                cost = max(ref_weights[i-1], pred_weights[j-1])

            dp[i][j] = min(
                dp[i-1][j] + ref_weights[i-1],      # Deletion
                dp[i][j-1] + pred_weights[j-1],      # Insertion
                dp[i-1][j-1] + cost                  # Substitution
            )

    edit_cost = dp[n][m]
    normalized_score = edit_cost / max_total_weight
    return min(1.0, max(0.0, normalized_score))


def parse_report_sections(report_text: str) -> Tuple[Dict[str, str], str]:
    """
    Parses report into FINDINGS fields dictionary and IMPRESSION text.
    """
    findings_fields = {}
    impression_text = ""
    
    if not report_text:
        return findings_fields, impression_text

    sections = re.split(r'\n(?=IMPRESSION:)', report_text, flags=re.IGNORECASE)
    findings_sec = sections[0]
    impression_sec = sections[1] if len(sections) > 1 else ""

    # Clean header from findings
    findings_body = re.sub(r'^FINDINGS:\s*', '', findings_sec, flags=re.IGNORECASE).strip()
    
    # Parse field keys using multiline regex
    field_matches = list(re.finditer(r'^[ \t]*([A-Z0-9\s\/_\-\(\)]+):\s*', findings_body, flags=re.MULTILINE))
    
    for idx, match in enumerate(field_matches):
        key = match.group(1).strip()
        start_pos = match.end()
        end_pos = field_matches[idx + 1].start() if idx + 1 < len(field_matches) else len(findings_body)
        value = findings_body[start_pos:end_pos].strip()
        findings_fields[key] = value

    # Parse IMPRESSION body
    impression_body = re.sub(r'^IMPRESSION:\s*', '', impression_sec, flags=re.IGNORECASE).strip()
    impression_text = impression_body

    return findings_fields, impression_text


def compute_res_score(reference_report: str, predicted_report: str, template_content: str) -> Dict[str, float]:
    """
    Computes full Radiology Edit Score (RES) according to Kaggle competition metric rules.
    RES_case = 0.65 * FINDINGS_score + 0.35 * IMPRESSION_score
    """
    ref_findings, ref_impression = parse_report_sections(reference_report)
    pred_findings, pred_impression = parse_report_sections(predicted_report)
    tmpl_findings, _ = parse_report_sections(template_content)

    # 1. Compute FINDINGS score F
    all_keys = set(ref_findings.keys()).union(set(pred_findings.keys()))
    if not all_keys:
        findings_score = 0.0
    else:
        weighted_edit_sum = 0.0
        field_weight_sum = 0.0

        for key in ref_findings:
            ref_val = ref_findings[key]
            tmpl_val = tmpl_findings.get(key, "")
            
            # Check if reference field was changed from normal template
            is_changed = normalize_text(ref_val) != normalize_text(tmpl_val)
            field_weight = 3.0 if is_changed else 1.0

            if key in pred_findings:
                pred_val = pred_findings[key]
                edit = weighted_token_edit_distance(ref_val, pred_val)
            else:
                # Missing expected field penalty
                edit = 1.0

            weighted_edit_sum += field_weight * edit
            field_weight_sum += field_weight

        # Extra unexpected fields penalty
        for key in pred_findings:
            if key not in ref_findings:
                field_weight = 2.0
                edit = 1.0
                weighted_edit_sum += field_weight * edit
                field_weight_sum += field_weight

        findings_score = weighted_edit_sum / field_weight_sum if field_weight_sum > 0 else 0.0

    # 2. Compute IMPRESSION score I
    impression_score = weighted_token_edit_distance(ref_impression, pred_impression)

    # 3. Overall case RES score
    res_case = 0.65 * findings_score + 0.35 * impression_score

    return {
        'res_case': res_case,
        'findings_score': findings_score,
        'impression_score': impression_score
    }
