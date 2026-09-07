import os
import csv
import json
import numpy as np
from typing import Dict, List, Any
import config
from Code.dataset import load_dataset
from Code.evaluator import compute_res_score
from Code.prompt_llm import Stage1LLMPipeline
from Code.rag_retriever import RAGRetriever
from Code.rag_llm import RAGLLMPipeline
from Code.validator import validate_report_structure, validate_clinical_fidelity

def create_stratified_folds(dataset: List[Dict[str, str]], num_folds: int = 5) -> List[List[int]]:
    """Create 5-fold splits stratified by modality + body_part."""
    strata = {}
    for idx, case in enumerate(dataset):
        key = f"{case.get('modality', '')}_{case.get('body_part', '')}"
        strata.setdefault(key, []).append(idx)

    folds = [[] for _ in range(num_folds)]
    for key, indices in strata.items():
        for i, idx in enumerate(indices):
            folds[i % num_folds].append(idx)
            
    return folds


def run_stage2_cross_validation(train_path: str = "train.csv", top_k: int = 3, max_cases: int = None) -> Dict[str, float]:
    """
    Runs 5-fold cross-validation of Stage 2 RAG on train.csv and computes average RES scores.
    """
    print("=" * 60)
    print("RUNNING STAGE 2 RAG CROSS-VALIDATION EVALUATION")
    print("=" * 60)

    dataset = load_dataset(train_path)
    if max_cases:
        dataset = dataset[:max_cases]

    folds = create_stratified_folds(dataset, num_folds=5)
    
    # Initialize RAG Retriever with full dataset
    retriever = RAGRetriever(dataset)
    rag_pipeline = RAGLLMPipeline()

    all_res_scores = []
    all_findings_scores = []
    all_impression_scores = []
    validation_failures = 0

    total_cases = len(dataset)

    for fold_idx, val_indices in enumerate(folds):
        fold_res = []
        fold_findings = []
        fold_impression = []

        print(f"\nEvaluating Fold {fold_idx + 1}/5 ({len(val_indices)} cases)...")
        
        for case_idx in val_indices:
            query_case = dataset[case_idx]
            ref_report = query_case.get('report', '')
            tmpl_content = query_case.get('template_content', '')

            # Retrieve top-k reference cases excluding self
            retrieved_examples = retriever.retrieve(query_case, top_k=top_k, exclude_case_id=query_case.get('case_id'))

            # Generate Stage 2 RAG report
            pred_report, _ = rag_pipeline.process_case_rag(query_case, retrieved_examples)

            # Structural validation check
            is_valid, struct_issues = validate_report_structure(pred_report, tmpl_content)
            if not is_valid:
                validation_failures += 1

            # RES metric calculation
            scores = compute_res_score(ref_report, pred_report, tmpl_content)

            fold_res.append(scores['res_case'])
            fold_findings.append(scores['findings_score'])
            fold_impression.append(scores['impression_score'])

        mean_fold_res = float(np.mean(fold_res))
        mean_fold_f = float(np.mean(fold_findings))
        mean_fold_i = float(np.mean(fold_impression))

        print(f"Fold {fold_idx + 1} RES: {mean_fold_res:.4f} | FINDINGS: {mean_fold_f:.4f} | IMPRESSION: {mean_fold_i:.4f}")

        all_res_scores.extend(fold_res)
        all_findings_scores.extend(fold_findings)
        all_impression_scores.extend(fold_impression)

    overall_res = float(np.mean(all_res_scores))
    overall_f = float(np.mean(all_findings_scores))
    overall_i = float(np.mean(all_impression_scores))

    print("\n" + "=" * 60)
    print("STAGE 2 RAG CROSS-VALIDATION SUMMARY RESULTS")
    print("=" * 60)
    print(f"Total Cases Evaluated: {total_cases}")
    print(f"Mean Leaderboard RES Score : {overall_res:.4f} (Lower is better)")
    print(f"Mean FINDINGS Edit Score F : {overall_f:.4f}")
    print(f"Mean IMPRESSION Edit Score I: {overall_i:.4f}")
    print(f"Structural Validation Failures: {validation_failures}")
    print("=" * 60)

    summary = {
        'overall_res': overall_res,
        'overall_findings': overall_f,
        'overall_impression': overall_i,
        'validation_failures': validation_failures,
        'total_cases': total_cases
    }

    # Save summary
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/stage2_cv_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def generate_submission_rag(train_path: str = "train.csv", test_path: str = "test.csv", output_csv: str = "submission.csv", top_k: int = 3):
    """
    Generates submission.csv for test.csv cases using Stage 2 RAG pipeline.
    """
    print("\n" + "=" * 60)
    print("GENERATING STAGE 2 RAG SUBMISSION CSV FOR TEST DATASET")
    print("=" * 60)

    train_cases = load_dataset(train_path)
    test_cases = load_dataset(test_path)

    retriever = RAGRetriever(train_cases)
    rag_pipeline = RAGLLMPipeline()

    submission_rows = []
    validation_issues_count = 0

    for idx, query_case in enumerate(test_cases):
        case_id = query_case['case_id']
        
        # Retrieve top-k examples from train_cases
        retrieved_examples = retriever.retrieve(query_case, top_k=top_k)
        
        # Generate report
        pred_report, _ = rag_pipeline.process_case_rag(query_case, retrieved_examples)

        # Validate report
        is_valid, issues = validate_report_structure(pred_report, query_case.get('template_content', ''))
        if not is_valid:
            validation_issues_count += 1

        submission_rows.append({
            'case_id': case_id,
            'report': pred_report
        })

    # Write submission.csv cleanly using csv.DictWriter with utf-8
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['case_id', 'report'])
        writer.writeheader()
        writer.writerows(submission_rows)

    print(f"Successfully generated '{output_csv}' with {len(submission_rows)} rows.")
    print(f"Validation warning count: {validation_issues_count}")
    print("=" * 60)


if __name__ == "__main__":
    run_stage2_cross_validation()
    generate_submission_rag()
