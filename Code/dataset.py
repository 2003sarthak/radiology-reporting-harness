import csv
import re
import unicodedata
from typing import Dict, List, Any, Tuple

UNIT_MAP = {
    'millimeter': 'mm',
    'millimeters': 'mm',
    'millimetre': 'mm',
    'millimetres': 'mm',
    'centimeter': 'cm',
    'centimeters': 'cm',
    'centimetre': 'cm',
    'centimetres': 'cm',
}

def normalize_text(text: str) -> str:
    """
    Official benchmark text normalization rules:
    - Unicode NFC normalization, lowercase.
    - Punctuation ignored EXCEPT leading + or - attached directly to a number.
    - Leading list markers (1., 2), -, *, •) removed.
    - Hyphens between letters removed (air-space -> airspace).
    - Letter-number boundaries separated into tokens.
    - Standardized unit spellings.
    """
    if not text:
        return ""
    
    # 1. Unicode NFC normalization & lowercase
    text = unicodedata.normalize('NFC', text).lower()
    
    # 2. Split lines and strip leading list markers
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        # Strip leading bullet/list markers
        line = re.sub(r'^(\d+[\.\)]|[\-\*•])\s*', '', line)
        cleaned_lines.append(line)
    text = " ".join(cleaned_lines)
    
    # 3. Remove hyphens between letters (air-space -> airspace)
    text = re.sub(r'(?<=[a-z])\-(?=[a-z])', '', text)
    
    # 4. Separate letter-number boundaries into distinct tokens (e.g., L4-L5 -> l4 l5, 5mm -> 5 mm)
    text = re.sub(r'([a-z]+)(\d+)', r'\1 \2', text)
    text = re.sub(r'(\d+)([a-z]+)', r'\1 \2', text)
    
    # 5. Punctuation handling: preserve signed numbers (+5, -3.2), remove all other punctuation
    # Replace any punctuation that is not part of a signed number
    tokens = text.split()
    processed_tokens = []
    for token in tokens:
        # Check unit standardization first
        token_clean = token.strip(',.;:!?()[]{}"\'')
        if token_clean in UNIT_MAP:
            token_clean = UNIT_MAP[token_clean]
        
        if token_clean:
            # Preserve signed numbers like +5, -2.5
            if re.match(r'^[+-]?\d+(\.\d+)?$', token_clean):
                processed_tokens.append(token_clean)
            else:
                # Strip non-alphanumeric
                sub_tokens = re.findall(r'[a-z0-9]+|[+-]?\d+(?:\.\d+)?', token_clean)
                processed_tokens.extend(sub_tokens)
                
    return " ".join(processed_tokens)


def tokenize(text: str) -> List[str]:
    """Tokenize normalized text into list of words."""
    norm = normalize_text(text)
    return norm.split() if norm else []


def load_dataset(csv_path: str) -> List[Dict[str, str]]:
    """
    Load train.csv or test.csv safely using utf-8-sig.
    """
    records = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(dict(row))
    return records
