import re
import numpy as np
from typing import Dict, List, Any, Tuple
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Code.dataset import tokenize, normalize_text
from Code.template_editor import TemplateEditor


def get_template_signature(template_content: str) -> List[str]:
    """Extract ordered field keys from template_content."""
    fields, _ = TemplateEditor.parse_template(template_content)
    return [k.upper().strip() for k, _ in fields]


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0.0


class RAGRetriever:
    """
    Hybrid CPU RAG Retriever combining Template Field Signature Matching,
    BM25 Lexical Dictation Search, and TF-IDF Cosine Similarity.
    """
    def __init__(self, train_cases: List[Dict[str, str]]):
        self.train_cases = train_cases
        self.train_signatures = [set(get_template_signature(c.get('template_content', ''))) for c in train_cases]
        
        # Tokenize dictations for BM25
        self.corpus_tokens = [tokenize(c.get('dictation', '')) for c in train_cases]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        
        # Build TF-IDF vectorizer over dictation + study_description
        corpus_texts = [f"{c.get('study_description', '')} {c.get('dictation', '')}" for c in train_cases]
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.tfidf_matrix = self.tfidf.fit_transform(corpus_texts)

    def retrieve(self, query_case: Dict[str, str], top_k: int = 3, exclude_case_id: str = None) -> List[Dict[str, str]]:
        """
        Retrieves top-k most relevant reference training cases for query_case.
        Optionally excludes query_case case_id when running cross-validation.
        """
        query_sig = set(get_template_signature(query_case.get('template_content', '')))
        query_modality = query_case.get('modality', '')
        query_body_part = query_case.get('body_part', '')
        query_dictation = query_case.get('dictation', '')
        query_text = f"{query_case.get('study_description', '')} {query_dictation}"

        query_tokens = tokenize(query_dictation)
        
        # 1. BM25 scores
        bm25_raw_scores = self.bm25.get_scores(query_tokens) if query_tokens else np.zeros(len(self.train_cases))
        max_bm25 = np.max(bm25_raw_scores) if np.max(bm25_raw_scores) > 0 else 1.0
        bm25_norm_scores = bm25_raw_scores / max_bm25

        # 2. TF-IDF Cosine Similarity
        query_vec = self.tfidf.transform([query_text])
        tfidf_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # 3. Hybrid scoring
        final_scores = []
        for i, ref_case in enumerate(self.train_cases):
            # Exclude self if evaluating cross-validation
            if exclude_case_id and ref_case.get('case_id') == exclude_case_id:
                final_scores.append(-1.0)
                continue

            # Signature similarity
            sig_sim = jaccard_similarity(query_sig, self.train_signatures[i])
            
            # Metadata boost
            mod_boost = 1.2 if ref_case.get('modality') == query_modality else 0.8
            bp_boost = 1.3 if ref_case.get('body_part') == query_body_part else 0.7

            # Combined Score Formula
            combined = (0.4 * sig_sim + 0.35 * bm25_norm_scores[i] + 0.25 * tfidf_scores[i]) * mod_boost * bp_boost
            final_scores.append(combined)

        top_indices = np.argsort(final_scores)[::-1][:top_k]
        return [self.train_cases[idx] for idx in top_indices if final_scores[idx] > 0]
