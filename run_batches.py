"""
Batch runner for the radiology reporting harness (free-tier Gemini friendly).

Nothing in Code/ or config.py is modified: this script only *uses* the existing
pipeline pieces (RAGRetriever, RAGLLMPipeline, TemplateEditor, evaluator) and
drives them in small, resumable chunks so a free-tier API quota is never asked
to do 132 test cases (or 636 train cases) in one shot.

Progress is appended to JSONL shard files after every single case, so an
interrupted / quota-exhausted run never loses completed work. Re-running the
same command skips case_ids that are already done.

Typical usage
-------------
  # Generate test predictions, 50 cases at a time
  python run_batches.py test --start 0   --count 50
  python run_batches.py test --start 50  --count 50
  python run_batches.py test --start 100 --count 32

  # Leave-one-out RES scoring on train.csv, 50 cases at a time
  python run_batches.py cv --start 0 --count 50

  # Where am I?
  python run_batches.py status

  # Aggregate the CV scores + write submission.csv
  python run_batches.py score
  python run_batches.py merge
"""

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

import config
from Code.dataset import load_dataset
from Code.evaluator import compute_res_score
from Code.rag_llm import RAGLLMPipeline
from Code.rag_prompt_builder import build_rag_prompt
from Code.rag_retriever import RAGRetriever
from Code.template_editor import TemplateEditor
from Code.validator import validate_report_structure

BATCH_DIR = os.path.join("outputs", "batches")
TEST_SHARD = os.path.join(BATCH_DIR, "test_preds.jsonl")
CV_SHARD = os.path.join(BATCH_DIR, "cv_scores.jsonl")

# How many times to re-issue a call that came back empty (503 "high demand" etc.)
TRANSIENT_RETRIES = 4


# ----------------------------------------------------------------------------
# JSONL shard helpers (resume support)
# ----------------------------------------------------------------------------

def read_shard(path: str) -> Dict[str, Dict[str, Any]]:
    """Load a JSONL shard keyed by case_id. Tolerates a truncated last line."""
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # partially written final line after a hard kill
            done[rec["case_id"]] = rec
    return done


def append_shard(path: str, record: Dict[str, Any]) -> None:
    """Append one record and fsync so an abrupt kill cannot lose completed work."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ----------------------------------------------------------------------------
# Single-case generation (explicit, no silent heuristic fallback)
# ----------------------------------------------------------------------------

class QuotaExhausted(RuntimeError):
    """Raised when the LLM stops returning anything - stop the batch, resume later."""


def generate_report(rag: RAGLLMPipeline,
                    query_case: Dict[str, str],
                    retrieved: List[Dict[str, str]]) -> str:
    """
    Run one case through the RAG prompt -> Gemini -> TemplateEditor path.

    Unlike RAGLLMPipeline.process_case_rag, an API failure is *not* swallowed
    into the Stage-1 heuristic fallback. A silent fallback would quietly poison
    the shard with lower-quality rows that look identical to real ones, so we
    raise instead and let the caller stop the batch cleanly.
    """
    if not rag.client:
        raise QuotaExhausted("Gemini client not initialised - check config.GEMINI_API_KEY")
    if not retrieved:
        raise QuotaExhausted("No RAG examples retrieved for " + str(query_case.get("case_id")))

    prompt = build_rag_prompt(query_case, retrieved)

    # RAGLLMPipeline.call_llm only retries on 429; it gives up immediately on the
    # 503 "high demand" responses that the free tier throws constantly. Retry the
    # whole call here with backoff so one transient blip doesn't end a 50-case batch.
    raw = ""
    for attempt in range(1, TRANSIENT_RETRIES + 1):
        raw = rag.call_llm(prompt)
        if raw:
            break
        if attempt < TRANSIENT_RETRIES:
            wait = 10 * attempt
            print("    (empty response, retry %d/%d in %ds)" % (attempt, TRANSIENT_RETRIES, wait))
            time.sleep(wait)
    if not raw:
        raise QuotaExhausted("no response after %d attempts (quota / auth / model down)"
                             % TRANSIENT_RETRIES)

    data = json.loads(raw)
    return TemplateEditor.render_report(
        query_case.get("template_content", ""),
        data.get("field_updates", {}),
        data.get("impression", ""),
    )


def preflight(model: str) -> None:
    """
    One cheap call before spending a batch, so a bad key / bad model name fails
    in 2 seconds instead of after 50 wasted slots.
    """
    key = config.get_api_key()
    if not key:
        sys.exit("No API key. Set GEMINI_API_KEY in your environment or in config.py.")
    print("Preflight: key=%s...%s (%d chars), model=%s"
          % (key[:6], key[-4:], len(key), model))
    try:
        from google import genai
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(model=model, contents="Reply with: OK")
        print("Preflight OK -> %r\n" % (resp.text or "").strip()[:40])
    except Exception as e:
        msg = str(e)
        sys.exit(
            "Preflight FAILED: %s: %s\n\n"
            "  401/UNAUTHENTICATED -> the key is not a valid AI Studio key. A Gemini\n"
            "     Developer API key starts with 'AIza' (get one at aistudio.google.com/apikey).\n"
            "     Then run:  setx GEMINI_API_KEY \"AIza...\"   (reopen the shell), or use --api-key.\n"
            "  404/NOT_FOUND    -> the model name is wrong for this key; try --model gemini-2.5-flash.\n"
            "  429/RESOURCE_EXHAUSTED -> free-tier quota is spent; resume later, the shard is kept."
            % (type(e).__name__, msg[:300])
        )


def apply_overrides(api_key: str, model: str) -> str:
    """
    Point the existing pipeline at a different key/model *in memory only*.
    config.py is never rewritten - the modules read config.MODEL_NAME and
    config.get_api_key() at call time, so patching the loaded module is enough.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
    if key:
        config.GEMINI_API_KEY = key.strip()
    if model:
        config.MODEL_NAME = model
    return config.MODEL_NAME


def run_slice(mode: str, start: int, count: int, sleep_s: float, top_k: int) -> None:
    """Process dataset[start:start+count] for 'test' (predict) or 'cv' (score)."""
    is_cv = mode == "cv"
    shard_path = CV_SHARD if is_cv else TEST_SHARD

    train_cases = load_dataset("train.csv")
    targets = train_cases if is_cv else load_dataset("test.csv")

    if start >= len(targets):
        print("start=%d is past the end of the %s set (%d cases). Nothing to do."
              % (start, mode, len(targets)))
        return
    end = min(start + count, len(targets))

    done = read_shard(shard_path)
    window = targets[start:end]
    todo = [c for c in window if c["case_id"] not in done]

    print("=" * 68)
    print("MODE=%s  slice=[%d:%d]  window=%d  already_done=%d  to_run=%d"
          % (mode.upper(), start, end, len(window), len(window) - len(todo), len(todo)))
    print("model=%s  top_k=%d  sleep=%ss  shard=%s"
          % (config.MODEL_NAME, top_k, sleep_s, shard_path))
    print("=" * 68)
    if not todo:
        print("Every case in this slice is already in the shard. Nothing to do.")
        return

    preflight(config.MODEL_NAME)

    # The retriever is always built over the full train set; in CV mode the case
    # itself is excluded at query time, exactly as pipeline.py does.
    retriever = RAGRetriever(train_cases)
    rag = RAGLLMPipeline()

    completed = 0
    for i, case in enumerate(todo, 1):
        case_id = case["case_id"]
        try:
            retrieved = retriever.retrieve(
                case, top_k=top_k,
                exclude_case_id=case_id if is_cv else None,
            )
            report = generate_report(rag, case, retrieved)
        except QuotaExhausted as e:
            print("\n[STOPPED] %s: %s" % (case_id, e))
            print("Saved %d new case(s) this run. Re-run the same command later to resume." % completed)
            _print_progress(shard_path, len(targets), mode)
            sys.exit(2)
        except Exception as e:  # bad JSON, template edge case, etc. - skip this one
            print("[SKIP] %s: %s: %s" % (case_id, type(e).__name__, e))
            continue

        is_valid, issues = validate_report_structure(report, case.get("template_content", ""))
        record = {"case_id": case_id, "report": report, "valid": is_valid}

        if is_cv:
            scores = compute_res_score(case.get("report", ""), report, case.get("template_content", ""))
            record.update(
                res=scores["res_case"],
                findings=scores["findings_score"],
                impression=scores["impression_score"],
                modality=case.get("modality", ""),
                body_part=case.get("body_part", ""),
            )
            print("[%d/%d] %s  RES=%.4f F=%.4f I=%.4f%s"
                  % (i, len(todo), case_id, scores["res_case"], scores["findings_score"],
                     scores["impression_score"], "" if is_valid else "  (STRUCT WARN)"))
        else:
            print("[%d/%d] %s  %d chars%s"
                  % (i, len(todo), case_id, len(report),
                     "" if is_valid else "  (STRUCT WARN: " + "; ".join(issues) + ")"))

        append_shard(shard_path, record)
        completed += 1

        if sleep_s and i < len(todo):
            time.sleep(sleep_s)

    print("\nSlice finished. %d new case(s) written." % completed)
    _print_progress(shard_path, len(targets), mode)


def _print_progress(shard_path: str, total: int, mode: str) -> None:
    n = len(read_shard(shard_path))
    print("Overall %s progress: %d/%d cases done (%.1f%%)." % (mode, n, total, 100.0 * n / total))


# ----------------------------------------------------------------------------
# status / score / merge
# ----------------------------------------------------------------------------

def cmd_status() -> None:
    test_cases = load_dataset("test.csv")
    train_cases = load_dataset("train.csv")
    test_done = read_shard(TEST_SHARD)
    cv_done = read_shard(CV_SHARD)

    print("=" * 68)
    print("BATCH PROGRESS")
    print("=" * 68)
    print("TEST predictions : %4d/%d   shard=%s" % (len(test_done), len(test_cases), TEST_SHARD))
    print("TRAIN CV scores  : %4d/%d   shard=%s" % (len(cv_done), len(train_cases), CV_SHARD))

    for label, cases, done in (("test", test_cases, test_done), ("cv", train_cases, cv_done)):
        missing = [i for i, c in enumerate(cases) if c["case_id"] not in done]
        if not missing:
            print("\n%s: COMPLETE." % label)
            continue
        print("\n%s: %d remaining. Next command:" % (label, len(missing)))
        print("  python run_batches.py %s --start %d --count 50" % (label, missing[0]))
    print("=" * 68)


def cmd_score() -> None:
    """Aggregate every CV shard record into the overall leaderboard-style RES."""
    recs = list(read_shard(CV_SHARD).values())
    if not recs:
        print("No CV scores yet in %s. Run: python run_batches.py cv --start 0 --count 50" % CV_SHARD)
        return

    train_total = len(load_dataset("train.csv"))
    res = np.array([r["res"] for r in recs])
    findings = np.array([r["findings"] for r in recs])
    impression = np.array([r["impression"] for r in recs])
    failures = sum(1 for r in recs if not r.get("valid", True))

    print("=" * 68)
    print("AGGREGATED CROSS-VALIDATION SCORE (lower RES is better)")
    print("=" * 68)
    print("Cases scored                 : %d/%d" % (len(recs), train_total))
    print("Mean Leaderboard RES         : %.4f" % res.mean())
    print("Mean FINDINGS edit score  F  : %.4f" % findings.mean())
    print("Mean IMPRESSION edit score I : %.4f" % impression.mean())
    print("Median RES                   : %.4f" % np.median(res))
    print("Std dev RES                  : %.4f" % res.std())
    print("Structural validation fails  : %d" % failures)

    by_mod: Dict[str, List[float]] = {}
    for r in recs:
        by_mod.setdefault(r.get("modality", "?"), []).append(r["res"])
    print("\nRES by modality:")
    for mod, vals in sorted(by_mod.items(), key=lambda kv: -float(np.mean(kv[1]))):
        print("  %-10s n=%4d  RES=%.4f" % (mod, len(vals), np.mean(vals)))

    print("\n10 worst cases (biggest edit distance):")
    for r in sorted(recs, key=lambda r: -r["res"])[:10]:
        print("  %-16s RES=%.4f  %s/%s"
              % (r["case_id"], r["res"], r.get("modality", ""), r.get("body_part", "")))

    summary = {
        "overall_res": float(res.mean()),
        "overall_findings": float(findings.mean()),
        "overall_impression": float(impression.mean()),
        "median_res": float(np.median(res)),
        "std_res": float(res.std()),
        "validation_failures": failures,
        "cases_scored": len(recs),
        "train_total": train_total,
        "model": config.MODEL_NAME,
    }
    os.makedirs("outputs", exist_ok=True)
    out = os.path.join("outputs", "batch_cv_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nWrote %s" % out)
    print("=" * 68)


def cmd_merge(output_csv: str, allow_partial: bool) -> None:
    """Assemble submission.csv from the test shard, in test.csv row order."""
    test_cases = load_dataset("test.csv")
    done = read_shard(TEST_SHARD)
    missing = [c["case_id"] for c in test_cases if c["case_id"] not in done]

    if missing and not allow_partial:
        print("REFUSING TO WRITE: %d of %d test cases are still missing."
              % (len(missing), len(test_cases)))
        print("First missing: %s" % missing[:5])
        print("Run the remaining batches, or pass --allow-partial to fill the gaps with")
        print("the unedited template (valid CSV, but those rows will score poorly).")
        sys.exit(1)

    rows, filled = [], 0
    for case in test_cases:
        rec = done.get(case["case_id"])
        if rec is None:
            report = case.get("template_content", "")  # structurally valid placeholder
            filled += 1
        else:
            report = rec["report"]
        rows.append({"case_id": case["case_id"], "report": report})

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "report"])
        writer.writeheader()
        writer.writerows(rows)

    invalid = sum(1 for r in done.values() if not r.get("valid", True))
    print("=" * 68)
    print("Wrote %s: %d rows (%d generated, %d template placeholders)."
          % (output_csv, len(rows), len(rows) - filled, filled))
    print("Structural validation warnings among generated rows: %d" % invalid)
    print("=" * 68)


# ----------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Chunked runner for the radiology harness.")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, help_text in (("test", "generate test predictions for a slice"),
                            ("cv", "leave-one-out RES scoring for a train slice")):
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("--start", type=int, default=0, help="index of the first case in the slice")
        sp.add_argument("--count", type=int, default=50, help="cases per batch (default 50)")
        sp.add_argument("--sleep", type=float, default=4.0,
                        help="seconds between API calls, to stay under the free-tier RPM limit")
        sp.add_argument("--top-k", type=int, default=config.TOP_K_EXAMPLES,
                        help="RAG examples per prompt")
        sp.add_argument("--model", default="",
                        help="override config.MODEL_NAME for this run only (e.g. gemini-2.5-flash)")
        sp.add_argument("--api-key", default="",
                        help="override the key for this run only; also read from $GEMINI_API_KEY")

    sub.add_parser("status", help="show how many cases are done and the next command to run")
    sub.add_parser("score", help="aggregate all CV shard scores into the overall RES")

    mp = sub.add_parser("merge", help="build submission.csv from the test shard")
    mp.add_argument("--output", default="submission.csv")
    mp.add_argument("--allow-partial", action="store_true",
                    help="write the CSV even if some cases are missing (gaps get the raw template)")

    args = p.parse_args()

    if args.cmd in ("test", "cv"):
        apply_overrides(args.api_key, args.model)
        run_slice(args.cmd, args.start, args.count, args.sleep, args.top_k)
    elif args.cmd == "status":
        cmd_status()
    elif args.cmd == "score":
        cmd_score()
    elif args.cmd == "merge":
        cmd_merge(args.output, args.allow_partial)


if __name__ == "__main__":
    main()
