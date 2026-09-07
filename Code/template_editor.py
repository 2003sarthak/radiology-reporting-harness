import re
from typing import Dict, Any, Tuple, List

class TemplateEditor:
    """
    Deterministic template parser and updater.
    Parses template_content into structured sections, updates only modified fields,
    and preserves untouched normal statements byte-for-byte in exact order.
    """
    
    @staticmethod
    def parse_template(template_content: str) -> Tuple[List[Tuple[str, str]], str]:
        """
        Parses template_content into an ordered list of (field_key, default_value) tuples
        and default impression text.
        """
        fields = []
        impression = ""
        
        if not template_content:
            return fields, impression

        sections = re.split(r'\n(?=IMPRESSION:)', template_content, flags=re.IGNORECASE)
        findings_sec = sections[0]
        impression_sec = sections[1] if len(sections) > 1 else ""

        # Remove FINDINGS: header
        findings_body = re.sub(r'^FINDINGS:\s*', '', findings_sec, flags=re.IGNORECASE).strip()
        
        # Regex matching line-starting field labels (e.g. LUNGS:, PLEURA:, BONES:)
        matches = list(re.finditer(r'^[ \t]*([A-Z0-9\s\/_\-\(\)]+):\s*', findings_body, flags=re.MULTILINE))
        
        for idx, m in enumerate(matches):
            key = m.group(1).strip()
            start = m.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(findings_body)
            val = findings_body[start:end].strip()
            fields.append((key, val))

        # Impression
        impression_body = re.sub(r'^IMPRESSION:\s*', '', impression_sec, flags=re.IGNORECASE).strip()
        impression = impression_body

        return fields, impression

    @staticmethod
    def render_report(template_content: str, field_updates: Dict[str, str], impression_update: str = None) -> str:
        """
        Applies field_updates onto template_content.
        Fields in field_updates update the default text.
        All untouched fields remain 100% identical to the template_content default text.
        """
        fields, default_impression = TemplateEditor.parse_template(template_content)
        
        updated_findings = []
        for key, default_val in fields:
            # Match case-insensitively or exact match
            key_lookup = None
            for k in field_updates:
                if k.strip().upper() == key.strip().upper():
                    key_lookup = k
                    break
            
            if key_lookup and field_updates[key_lookup]:
                val = field_updates[key_lookup].strip()
            else:
                val = default_val.strip()
            
            updated_findings.append(f"{key}: {val}")

        # Check if there are extra field updates not in the template (e.g., OTHER FINDINGS)
        template_keys = {k.strip().upper() for k, _ in fields}
        for k, v in field_updates.items():
            if k.strip().upper() not in template_keys and v:
                updated_findings.append(f"{k.strip().upper()}: {v.strip()}")

        findings_block = "FINDINGS:\n" + "\n".join(updated_findings)
        
        final_impression = impression_update.strip() if impression_update and impression_update.strip() else default_impression.strip()
        impression_block = f"IMPRESSION:\n{final_impression}"

        return f"{findings_block}\n\n{impression_block}"
