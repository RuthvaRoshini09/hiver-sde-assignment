"""
src/llm_judge.py
----------------
LLM-as-Judge Quality Evaluation Harness for AppleSupport Agent.

Evaluates generated support replies across four core quality criteria:
  A. Correctness: Does the response appropriately address the customer's issue?
  B. Grounding: Is the response supported by the retrieved historical support evidence?
  C. Relevance: Does the response stay focused on the customer's request without unrelated claims?
  D. Actionability: Does the response provide useful and appropriate next steps?

Scoring Scale:
  1 = Very Poor (Completely fails the criterion)
  2 = Poor (Major deficiencies or inaccuracies)
  3 = Acceptable (Meets baseline support expectations)
  4 = Good (High quality with minor room for improvement)
  5 = Excellent (Exemplary Apple Support response)

Overall Metrics:
  overall_score = mean(Correctness, Grounding, Relevance, Actionability)
  pass          = overall_score >= 3.0
  strong_pass   = overall_score >= 4.0

Input:
  data/processed/evaluation_results.csv (200 rows)

Outputs:
  data/processed/llm_judge_results.csv
  data/processed/llm_judge_summary.json
  docs/evaluation_report.md (updated with LLM-as-Judge section)

Usage:
  python src/llm_judge.py
"""

import os
import sys
import json
import time
import re
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Setup project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe UTF-8 console output for Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd
import requests

# File paths
EVAL_RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "evaluation_results.csv"
OUTPUT_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "llm_judge_results.csv"
OUTPUT_JSON_PATH = PROJECT_ROOT / "data" / "processed" / "llm_judge_summary.json"
REPORT_MD_PATH = PROJECT_ROOT / "docs" / "evaluation_report.md"

CSV_HEADERS = [
    "tweet_id",
    "customer_message",
    "predicted_intent",
    "ground_truth_intent",
    "retrieved_reply",
    "generated_reply",
    "correctness",
    "grounding",
    "relevance",
    "actionability",
    "overall_score",
    "pass",
    "strong_pass",
    "judge_reason"
]

JUDGE_SYSTEM_PROMPT = """You are an expert customer support quality auditor evaluating an AI assistant for Apple Support on Twitter/X.
You will evaluate the AI's drafted response against the customer message, the predicted intent, the ground-truth intent, and the retrieved historical evidence.

Evaluate the generated reply across exactly four criteria on a 1-to-5 integer scale:
1. Correctness (1-5): Does the response appropriately and accurately address the customer's technical or general problem?
   1 = Completely incorrect or contradicts Apple facts
   3 = Plausible guidance addressing the problem adequately
   5 = Completely accurate, perfectly tailored technical troubleshooting
2. Grounding (1-5): Is the response supported and anchored by the retrieved historical support evidence?
   1 = Fabricated or completely contradicts retrieved facts
   3 = Partially aligned or general safe guidance consistent with evidence
   5 = Fully faithful to and supported by the retrieved support context
3. Relevance (1-5): Does the response stay strictly focused on the customer's query without distracting claims?
   1 = Totally off-topic or answering a different problem
   3 = Mostly relevant with minor tangent
   5 = Laser-focused on the exact user issue
4. Actionability (1-5): Does the response provide concrete, helpful, and appropriate next steps (e.g. DM invite, settings path, reset)?
   1 = Vague dismissal with no actionable steps
   3 = Basic next step (e.g., restart or reach out)
   5 = Clear, specific, step-by-step diagnostic or support path

You must return valid JSON only, with the following exact keys:
{
  "correctness": <int 1-5>,
  "grounding": <int 1-5>,
  "relevance": <int 1-5>,
  "actionability": <int 1-5>,
  "judge_reason": "<brief 1-2 sentence justification>"
}"""


def build_judge_prompt(
    customer_message: str,
    predicted_intent: str,
    ground_truth_intent: str,
    retrieved_reply: str,
    generated_reply: str
) -> str:
    """Builds user prompt for the LLM judge."""
    return f"""Please evaluate the following support interaction:

[CUSTOMER MESSAGE]
"{customer_message}"

[PREDICTED INTENT]
{predicted_intent}

[GROUND-TRUTH INTENT]
{ground_truth_intent}

[RETRIEVED HISTORICAL EVIDENCE]
"{retrieved_reply if retrieved_reply.strip() else '[No retrieval evidence available]'}"

[AI GENERATED REPLY]
"{generated_reply if generated_reply.strip() else '[No reply generated]'}"

Evaluate across Correctness (1-5), Grounding (1-5), Relevance (1-5), and Actionability (1-5). Return JSON only."""


def call_openai_judge(
    api_key: str,
    model: str,
    user_prompt: str,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Calls OpenAI Chat Completions API with retries for transient errors only.
    Fails immediately on authentication errors (HTTP 401) and client errors.
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)

            # Success
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(content)

                # Validate required keys and ranges
                scores = {}
                for k in ["correctness", "grounding", "relevance", "actionability"]:
                    val = float(parsed.get(k, 0))
                    val = max(1.0, min(5.0, round(val)))
                    scores[k] = int(val)

                reason = str(parsed.get("judge_reason", "")).strip()
                scores["judge_reason"] = reason if reason else "Evaluated via LLM-as-Judge criteria."
                return scores

            # Authentication error (401) -> DO NOT RETRY, fail immediately!
            elif resp.status_code == 401:
                raise RuntimeError(
                    f"OpenAI Authentication Error (HTTP 401): Invalid API key or unauthorized. Details: {resp.text[:120]}"
                )

            # Transient errors: 429 (rate limit), 500/502/503/504 (server errors) -> retry with backoff
            elif resp.status_code in [429, 500, 502, 503, 504]:
                wait_secs = attempt * 2
                time.sleep(wait_secs)
                last_error = f"Transient HTTP {resp.status_code}: {resp.text[:100]}"

            # Other non-transient client errors (e.g. 400 bad request, 403, 404) -> fail immediately
            else:
                raise RuntimeError(f"OpenAI Client Error (HTTP {resp.status_code}): {resp.text[:120]}")

        except requests.exceptions.Timeout as e:
            # Network timeouts are transient -> retry with backoff
            last_error = f"Request Timeout: {e}"
            if attempt < max_retries:
                time.sleep(attempt * 2)
        except requests.exceptions.ConnectionError as e:
            # Connection errors are transient -> retry with backoff
            last_error = f"Connection Error: {e}"
            if attempt < max_retries:
                time.sleep(attempt * 2)

    raise RuntimeError(f"Failed after {max_retries} attempts: {last_error}")


def load_env_file(env_path: Path):
    """Optionally loads key-value pairs from a local .env file if present."""
    if not env_path.is_file():
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


def is_valid_openai_api_key(key: Optional[str]) -> Tuple[bool, str]:
    """
    Validates if an OpenAI API key is configured and not an obvious placeholder.
    Detects missing values, empty strings, and known placeholder templates such as:
      - 'YOUR_REAL_API_KEY', 'YOUR_API_KEY', 'YOUR_KEY_HERE'
      - 'sk-your-real-key-here', 'sk-your-...'
      - 'placeholder', 'dummy', 'none', 'test', 'fake'
    Returns (is_valid: bool, reason: str).
    """
    if not key or not str(key).strip():
        return False, "API key is empty or not set."

    cleaned_key = str(key).strip()

    # Known placeholder patterns (case-insensitive)
    placeholder_patterns = [
        r"^sk-your.*",
        r".*your[_-]?real[_-]?api[_-]?key.*",
        r".*your[_-]?api[_-]?key.*",
        r".*your[_-]?key.*",
        r"^sk-placeholder.*",
        r"^placeholder.*",
        r"^dummy.*",
        r"^test.*",
        r"^fake.*",
        r"^sk-\.\.\..*",
        r"^\.\.\..*",
        r"^<.*>$",
        r"^sk-xxx.*",
        r"^xxx.*",
        r"^changeme.*",
        r"^none$",
        r"^null$"
    ]

    for pattern in placeholder_patterns:
        if re.search(pattern, cleaned_key, flags=re.IGNORECASE):
            return False, f"Detected placeholder API key ('{cleaned_key[:15]}...')."

    # OpenAI API keys typically start with 'sk-' and are at least 20 characters
    if not cleaned_key.startswith("sk-"):
        return False, "OpenAI API keys typically start with 'sk-'."

    if len(cleaned_key) < 20:
        return False, "API key is too short to be a valid OpenAI key."

    return True, "Valid API key format."


def update_evaluation_report_markdown(summary: Dict[str, Any]):
    """
    Appends or updates the 'LLM-as-Judge Evaluation' section in docs/evaluation_report.md.
    """
    if not REPORT_MD_PATH.exists():
        return

    content = REPORT_MD_PATH.read_text(encoding="utf-8")

    # Build the LLM-as-Judge Markdown section
    status = summary.get("evaluation_status", "NOT RUN — API authentication unavailable")
    section_lines = [
        "## LLM-as-Judge Evaluation",
        "",
        "In accordance with assignment requirements, an automated **LLM-as-Judge** evaluation protocol "
        "was developed to assess the qualitative properties of the agent's drafted replies. "
        "The judge evaluates each generated response against four core criteria scored on a 1-to-5 integer scale:",
        "",
        "1. **Correctness (1–5):** Does the response appropriately and accurately address the customer's problem?",
        "2. **Grounding (1–5):** Is the response supported by the retrieved historical support evidence without hallucinations?",
        "3. **Relevance (1–5):** Does the response stay focused on the user's issue without extraneous or distracting claims?",
        "4. **Actionability (1–5):** Does the response provide clear, constructive next steps (e.g., specific settings menu, DM transition)?",
        "",
        "### Scoring Rubric & Decision Thresholds",
        r"- **Overall Score:** Arithmetic mean of Correctness, Grounding, Relevance, and Actionability: $\frac{C + G + R + A}{4}$",
        r"- **Pass Threshold:** $\ge 3.0$ (response meets standard baseline support quality)",
        r"- **Strong Pass Threshold:** $\ge 4.0$ (response demonstrates high fidelity, groundedness, and helpfulness)",
        ""
    ]

    succ_judged = summary.get("successfully_judged", 0)
    if status == "COMPLETED" and succ_judged > 0:
        pass_rate_val = summary.get('pass_rate', 0.0) or 0.0
        strong_rate_val = summary.get('strong_pass_rate', 0.0) or 0.0
        section_lines.extend([
            f"### Execution Results (Evaluated on {succ_judged} / {summary['dataset_size']} Examples)",
            f"- **Mean Correctness:** {summary['mean_correctness']:.2f} / 5.0",
            f"- **Mean Grounding:** {summary['mean_grounding']:.2f} / 5.0",
            f"- **Mean Relevance:** {summary['mean_relevance']:.2f} / 5.0",
            f"- **Mean Actionability:** {summary['mean_actionability']:.2f} / 5.0",
            f"- **Mean Overall Score:** {summary['mean_overall_score']:.2f} / 5.0",
            f"- **Pass Rate (Score >= 3.0):** {pass_rate_val * 100:.1f}%",
            f"- **Strong Pass Rate (Score >= 4.0):** {strong_rate_val * 100:.1f}%",
            ""
        ])
    else:
        section_lines.extend([
            "### Evaluation Status: NOT RUN — API authentication unavailable",
            "The LLM-as-Judge evaluation harness is fully implemented in `src/llm_judge.py`. "
            "Because a valid `OPENAI_API_KEY` was not configured in the execution environment, automated judging was **cleanly skipped** "
            "rather than generating fabricated or synthetic scores.",
            "",
            "- **Mean Correctness:** N/A",
            "- **Mean Grounding:** N/A",
            "- **Mean Relevance:** N/A",
            "- **Mean Actionability:** N/A",
            "- **Mean Overall Score:** N/A",
            "- **Pass Rate (Score >= 3.0):** N/A",
            "- **Strong Pass Rate (Score >= 4.0):** N/A",
            "",
            "> **How to Enable LLM-as-Judge:**",
            "> 1. Set your OpenAI API key: `export OPENAI_API_KEY='sk-...'` (or in PowerShell: `$env:OPENAI_API_KEY='sk-...'`).",
            "> 2. Run: `python src/llm_judge.py`.",
            "> 3. Results will be incrementally recorded in `data/processed/llm_judge_results.csv` and `data/processed/llm_judge_summary.json`.",
            ""
        ])

    section_lines.extend([
        "### Methodological Limitations & Ground-Truth Boundaries",
        "- **Not Ground Truth:** The LLM judge serves as a secondary heuristic signal. LLM-as-judge scores **must NOT be treated as ground truth** or claimed as human agreement.",
        "- **No Escalation Modification:** The judge evaluates reply phrasing quality only; it does **not** alter, replace, or invent escalation labels for the 56 unlabeled golden set examples.",
        "- **Model Bias:** LLM judges may exhibit verbosity or agreement bias; scores should be interpreted alongside lexical retrieval overlap and classifier accuracy.",
        ""
    ])

    new_section_text = "\n".join(section_lines)

    # Check if section already exists
    if "## LLM-as-Judge Evaluation" in content:
        pattern = r"## LLM-as-Judge Evaluation.*?(?=\n## |\Z)"
        content = re.sub(pattern, lambda _: new_section_text.strip(), content, flags=re.DOTALL)
    else:
        # Insert before ## Limitations if present, else append
        if "## Limitations" in content:
            content = content.replace("## Limitations", new_section_text + "\n## Limitations")
        else:
            content = content.rstrip() + "\n\n" + new_section_text

    REPORT_MD_PATH.write_text(content, encoding="utf-8")
    print(f"Updated docs/evaluation_report.md with LLM-as-Judge section.")


def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge Quality Evaluation Harness")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of new rows to evaluate (e.g. --limit 1 for testing). Defaults to evaluating all rows."
    )
    args = parser.parse_args()

    print("=" * 80)
    print("LLM-AS-JUDGE QUALITY EVALUATION HARNESS")
    print("=" * 80)
    print(f"Source Results: {EVAL_RESULTS_PATH}")
    print(f"Output CSV:     {OUTPUT_CSV_PATH}")
    print(f"Output JSON:    {OUTPUT_JSON_PATH}")
    print(f"Report MD:      {REPORT_MD_PATH}")
    if args.limit is not None and args.limit > 0:
        print(f"Evaluation Mode: TEST MODE (Limit: {args.limit} new example(s))\n")
    else:
        print(f"Evaluation Mode: FULL EVALUATION (All examples)\n")

    if not EVAL_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Evaluation results not found at {EVAL_RESULTS_PATH}. Run src/evaluate_agent.py first.")

    df_eval = pd.read_csv(EVAL_RESULTS_PATH)
    total_examples = len(df_eval)
    print(f"Loaded {total_examples} agent evaluation results from {EVAL_RESULTS_PATH.name}.")

    # Load .env if present in project root
    load_env_file(PROJECT_ROOT / ".env")

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()

    is_valid_key, key_reason = is_valid_openai_api_key(api_key)

    # Handle Missing or Placeholder API Key cleanly without fabrication
    if not is_valid_key:
        print("\n" + "!" * 80)
        print("No valid OpenAI API key configured. LLM-as-Judge skipped.")
        print(f"Detail: {key_reason}")
        print("!" * 80)
        print("\nTo enable LLM-as-Judge evaluation with a real API key:")
        print("  1. In PowerShell:  $env:OPENAI_API_KEY = \"sk-your-actual-key\"")
        print("  2. In Windows CMD: set OPENAI_API_KEY=sk-your-actual-key")
        print("  3. In Linux/macOS: export OPENAI_API_KEY=\"sk-your-actual-key\"")
        print("  4. Or add OPENAI_API_KEY=sk-... to a local .env file (git-ignored)")
        print("  5. Re-run:         python src/llm_judge.py")
        print("     (or test mode): python src/llm_judge.py --limit 1\n")

        # Create empty CSV with required schema if it doesn't exist or is empty
        if not OUTPUT_CSV_PATH.exists() or os.path.getsize(OUTPUT_CSV_PATH) <= 2:
            OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
            df_empty = pd.DataFrame(columns=CSV_HEADERS)
            df_empty.to_csv(OUTPUT_CSV_PATH, index=False)
            print(f"Initialized empty results file with required schema: {OUTPUT_CSV_PATH}")

        # Create SKIPPED summary JSON with explicit status
        summary_payload = {
            "evaluation_status": "NOT RUN — API authentication unavailable",
            "reason": f"No valid OpenAI API key configured ({key_reason}).",
            "how_to_enable": "Set a valid OPENAI_API_KEY environment variable and rerun python src/llm_judge.py",
            "model": None,
            "dataset_size": total_examples,
            "successfully_judged": 0,
            "failed_examples": 0,
            "mean_correctness": None,
            "mean_grounding": None,
            "mean_relevance": None,
            "mean_actionability": None,
            "mean_overall_score": None,
            "pass_rate": None,
            "strong_pass_rate": None,
            "criteria_definitions": {
                "A_correctness": "Does the response appropriately address the customer's issue? (1-5 scale)",
                "B_grounding": "Is the response supported by the retrieved historical support evidence? (1-5 scale)",
                "C_relevance": "Does the response stay focused on the customer's request without unrelated claims? (1-5 scale)",
                "D_actionability": "Does the response provide useful and appropriate next steps? (1-5 scale)"
            },
            "thresholds": {
                "pass": "overall_score >= 3.0",
                "strong_pass": "overall_score >= 4.0"
            }
        }

        OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)
        print(f"Saved evaluation summary to: {OUTPUT_JSON_PATH}")

        # Update evaluation report markdown
        update_evaluation_report_markdown(summary_payload)

        print("\n" + "=" * 80)
        print("LLM-AS-JUDGE EXECUTION COMPLETE (STATUS: NOT RUN — API authentication unavailable)")
        print("=" * 80)
        return

    # If API key IS available and valid format, execute evaluation over target rows
    print(f"Using OpenAI Model: {model_name}")
    print(f"Beginning LLM evaluation over {total_examples} examples...\n")

    # Load previously judged results if resuming incrementally
    judged_records = []
    judged_tweet_ids = set()
    if OUTPUT_CSV_PATH.exists() and os.path.getsize(OUTPUT_CSV_PATH) > 0:
        try:
            df_existing = pd.read_csv(OUTPUT_CSV_PATH)
            if not df_existing.empty and "tweet_id" in df_existing.columns:
                judged_records = df_existing.to_dict(orient="records")
                judged_tweet_ids = set(df_existing["tweet_id"].tolist())
                print(f"Resuming evaluation: found {len(judged_records)} previously judged rows.")
        except Exception:
            judged_records = []
            judged_tweet_ids = set()

    newly_judged_count = 0
    failed_count = 0
    for idx, row in df_eval.iterrows():
        tweet_id = row["tweet_id"]
        if tweet_id in judged_tweet_ids:
            continue

        if args.limit is not None and newly_judged_count >= args.limit:
            print(f"\nLimit reached ({newly_judged_count}/{args.limit} new example(s) evaluated). Stopping execution.")
            break

        row_num = idx + 1
        clean_msg = str(row.get("clean_message", ""))
        pred_intent = str(row.get("predicted_intent", ""))
        gt_intent = str(row.get("ground_truth_intent", ""))
        retrieved_supp = str(row.get("top_retrieved_support_reply", "")) if pd.notna(row.get("top_retrieved_support_reply")) else ""
        generated_reply = str(row.get("drafted_reply", "")) if pd.notna(row.get("drafted_reply")) else ""

        current_progress = len(judged_records) + 1
        print(f"  Judging example {current_progress} (Row {row_num}, Tweet ID: {tweet_id})...")

        prompt = build_judge_prompt(
            customer_message=clean_msg,
            predicted_intent=pred_intent,
            ground_truth_intent=gt_intent,
            retrieved_reply=retrieved_supp,
            generated_reply=generated_reply
        )

        try:
            scores = call_openai_judge(api_key=api_key, model=model_name, user_prompt=prompt)
            c = scores["correctness"]
            g = scores["grounding"]
            r = scores["relevance"]
            a = scores["actionability"]
            overall = round((c + g + r + a) / 4.0, 2)
            passed = bool(overall >= 3.0)
            strong_passed = bool(overall >= 4.0)

            rec = {
                "tweet_id": tweet_id,
                "customer_message": clean_msg,
                "predicted_intent": pred_intent,
                "ground_truth_intent": gt_intent,
                "retrieved_reply": retrieved_supp,
                "generated_reply": generated_reply,
                "correctness": c,
                "grounding": g,
                "relevance": r,
                "actionability": a,
                "overall_score": overall,
                "pass": passed,
                "strong_pass": strong_passed,
                "judge_reason": scores["judge_reason"]
            }
            judged_records.append(rec)
            judged_tweet_ids.add(tweet_id)
            newly_judged_count += 1
        except Exception as e:
            failed_count += 1
            print(f"    [ERROR] Failed to judge Tweet ID {tweet_id}: {e}")
            if "Authentication Error (HTTP 401)" in str(e):
                print(f"    [CRITICAL] Halting evaluation due to authentication failure.")
                break

        # Save incrementally every 10 rows or when target reached or at end
        if len(judged_records) % 10 == 0 or (args.limit is not None and newly_judged_count >= args.limit) or idx == total_examples - 1:
            df_curr = pd.DataFrame(judged_records)
            df_curr.to_csv(OUTPUT_CSV_PATH, index=False)

    df_final = pd.DataFrame(judged_records)
    df_final.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nSaved {len(df_final)} judged records to: {OUTPUT_CSV_PATH}")

    # Calculate summary metrics across judged examples
    succ_count = len(df_final)
    if succ_count > 0:
        status_label = "COMPLETED"
        mean_c = round(float(df_final["correctness"].mean()), 4)
        mean_g = round(float(df_final["grounding"].mean()), 4)
        mean_r = round(float(df_final["relevance"].mean()), 4)
        mean_a = round(float(df_final["actionability"].mean()), 4)
        mean_overall = round(float(df_final["overall_score"].mean()), 4)
        pass_rate = round(float(df_final["pass"].mean()), 4)
        strong_pass_rate = round(float(df_final["strong_pass"].mean()), 4)
    else:
        status_label = "NOT RUN — API authentication unavailable"
        mean_c = mean_g = mean_r = mean_a = mean_overall = pass_rate = strong_pass_rate = None

    summary_payload = {
        "evaluation_status": status_label,
        "model": model_name if succ_count > 0 else None,
        "dataset_size": total_examples,
        "successfully_judged": succ_count,
        "failed_examples": failed_count,
        "mean_correctness": mean_c,
        "mean_grounding": mean_g,
        "mean_relevance": mean_r,
        "mean_actionability": mean_a,
        "mean_overall_score": mean_overall,
        "pass_rate": pass_rate,
        "strong_pass_rate": strong_pass_rate,
        "criteria_definitions": {
            "A_correctness": "Does the response appropriately address the customer's issue? (1-5 scale)",
            "B_grounding": "Is the response supported by the retrieved historical support evidence? (1-5 scale)",
            "C_relevance": "Does the response stay focused on the customer's request without unrelated claims? (1-5 scale)",
            "D_actionability": "Does the response provide useful and appropriate next steps? (1-5 scale)"
        },
        "thresholds": {
            "pass": "overall_score >= 3.0",
            "strong_pass": "overall_score >= 4.0"
        }
    }

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"Saved evaluation summary to: {OUTPUT_JSON_PATH}")

    update_evaluation_report_markdown(summary_payload)

    print("\n" + "=" * 80)
    print("LLM-AS-JUDGE EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Status:                   {status_label}")
    print(f"Model:                    {model_name}")
    print(f"Successfully Judged:      {succ_count} / {total_examples}")
    print(f"Failed Examples:          {failed_count}")
    print("-" * 80)
    if succ_count > 0:
        print(f"Mean Correctness:         {mean_c:.2f} / 5.0")
        print(f"Mean Grounding:           {mean_g:.2f} / 5.0")
        print(f"Mean Relevance:           {mean_r:.2f} / 5.0")
        print(f"Mean Actionability:       {mean_a:.2f} / 5.0")
        print(f"Mean Overall Score:       {mean_overall:.2f} / 5.0")
        print("-" * 80)
        print(f"Pass Rate (>= 3.0):       {pass_rate * 100:.1f}%")
        print(f"Strong Pass Rate (>= 4.0):{strong_pass_rate * 100:.1f}%")
    else:
        print("Mean Correctness:         N/A")
        print("Mean Grounding:           N/A")
        print("Mean Relevance:           N/A")
        print("Mean Actionability:       N/A")
        print("Mean Overall Score:       N/A")
        print("-" * 80)
        print("Pass Rate (>= 3.0):       N/A")
        print("Strong Pass Rate (>= 4.0):N/A")
    print("=" * 80)


if __name__ == "__main__":
    main()
