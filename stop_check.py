#!/usr/bin/env python3
"""
stop_check.py - Post-Phase-5 round verdict helper for the active learning loop.

Evaluates whether another AL round is worth running by reading 3 output
files from the most recent round and comparing 4 signals against
thresholds. Outputs STOP / CONTINUE / INVESTIGATE with reasoning.

Designed to be run AFTER `active_learning_round.py --phase 5 train`
completes for round N. Read-only - does not modify any pipeline state.

USAGE
-----
    # Basic check on round 2 (uses default strict thresholds)
    uv run stop_check.py --round 2

    # Show trend across rounds 1..N
    uv run stop_check.py --round 2 --all-rounds

    # JSON output for scripting / CI gating
    uv run stop_check.py --round 2 --json

    # Quiet: print verdict only (1 line)
    uv run stop_check.py --round 2 -q

    # Looser thresholds (e.g., budget-constrained mode)
    uv run stop_check.py --round 2 --stop-kappa 0.6 --stop-uncertain-pool 500

WHEN TO RUN
-----------
After completing a full AL round for round N:
    1. Phase 1 (inference)         -> predictions/master_predictions_round{N}.csv
    2. Phase 2 generate / verify   -> predictions/per_class_accuracy_round{N}.json
    3. Phase 4 compose / Phase 5   -> models/round{N}/model.json
Then:
    4. uv run stop_check.py --round N

INPUT FILES (read-only, must exist)
------------------------------------
- predictions/per_class_accuracy_round{N}.json   (Phase 2 verify output)
- models/round{N}/model.json                     (Phase 5 train output)
- predictions/master_predictions_round{N}.csv    (Phase 1 output)

If any are missing, verdict is INVESTIGATE and missing_inputs lists
which files to generate.

VERDICT LOGIC
-------------
Verdict is based on 5 signals. ALL 5 must be met for STOP:

  1. Fleiss' Kappa >= --stop-kappa          (default 0.81 = excellent)
  2. Zero class decisions == 'disabled'     (all commit or recheck)
  3. Val accuracy plateau                   (delta < --stop-acc-delta, default 0.005)
  4. Uncertain pool size < --stop-uncertain-pool  (default 50 patches with margin < 0.2)
  5. Needs-review ratio < --stop-needs-review-ratio  (default 0.01 = 1% of applicable,
     recomputed with per-class threshold from per_class_accuracy_roundN.json:
       commit  -> 0.9, recheck -> 0.95, disabled -> excluded from denominator)

Verdict mapping:
  - 5 of 5 met       -> STOP         (exit 0)
  - 2-4 of 5 met     -> INVESTIGATE  (exit 2; mixed signals, manual review)
  - 0-1 of 5 met     -> CONTINUE     (exit 1; another round likely productive)
  - missing inputs   -> INVESTIGATE  (exit 2; round not complete)

FLAGS
-----
  --round N                  Round to evaluate (required)
  --all-rounds               Show trend table for rounds 1..N
  --stop-kappa F             Kappa threshold for STOP (default: 0.81)
  --stop-acc-delta F         Val acc delta threshold for plateau (default: 0.005)
  --stop-uncertain-pool N    Uncertain pool size threshold (default: 50)
  --stop-needs-review-ratio F  Needs-review ratio threshold (default: 0.01 = 1%)
  --margin-threshold F       Margin cutoff for "uncertain" (default: 0.2)
  --json                     Output JSON to stdout (programmatic use)
  -q, --quiet                Print verdict line only, no tables

EXIT CODES
----------
  0   STOP recommended (all 5 signals met)
  1   CONTINUE recommended (< 2 signals met)
  2   INVESTIGATE (mixed signals OR missing inputs)

Use in shell scripts:
    if uv run stop_check.py --round 2 -q; then
        echo "STOP: AL converged"
    else
        echo "Round 2 not done; check output"
    fi

SEE ALSO
--------
- active_learning_round.py    - the main AL pipeline (phases 1-5)
- merge_annotation_consensus.py - computes Fleiss' Kappa for full reviews
- train_consensus_model.py    - generates models/round{N}/model.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
PREDICTIONS_DIR = BASE_DIR / "predictions"
MODELS_DIR = BASE_DIR / "models"

# Default thresholds (strict, production-grade)
DEFAULT_STOP_KAPPA = 0.81
DEFAULT_STOP_ACC_DELTA = 0.005
DEFAULT_STOP_UNCERTAIN_POOL = 50
DEFAULT_STOP_NEEDS_REVIEW_RATIO = 0.01
DEFAULT_MARGIN_THRESHOLD = 0.2
# Per-class threshold constants (must match active_learning_round.py)
AL_RECHECK_THRESHOLD = 0.95


def load_per_class_accuracy(round_n: int) -> dict:
    """Read predictions/per_class_accuracy_round{N}.json. Returns {} if missing."""
    path = PREDICTIONS_DIR / f"per_class_accuracy_round{round_n}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_model_metrics(round_n: int) -> dict:
    """Read models/round{N}/model.json. Returns {} if missing."""
    path = MODELS_DIR / f"round{round_n}" / "model.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def count_uncertain_pools(round_n: int, margin_threshold: float) -> int:
    """Count unlabeled rows with margin < threshold. Returns -1 if file missing."""
    path = PREDICTIONS_DIR / f"master_predictions_round{round_n}.csv"
    if not path.exists():
        return -1
    count = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                if (row.get("agreement_level") == "unlabeled"
                        and float(row.get("margin", 1.0)) < margin_threshold):
                    count += 1
            except (ValueError, TypeError):
                continue
    return count


def compute_needs_review_ratio(round_n: int,
                                per_class: dict,
                                confidence_threshold: float = 0.9,
                                recheck_threshold: float = AL_RECHECK_THRESHOLD
                                ) -> tuple[float | None, list[str]]:
    """Recompute needs_review with per-class thresholds from Phase 2 verify.

    Per-class effective threshold:
      commit   -> confidence_threshold (0.9)
      recheck  -> recheck_threshold    (0.95)
      disabled -> excluded from ratio (treated as not-applicable)

    Returns (ratio, missing_inputs). ratio = needs_review / applicable_total.
    Returns (None, [...]) if master_predictions CSV is missing.
    """
    path = PREDICTIONS_DIR / f"master_predictions_round{round_n}.csv"
    if not path.exists():
        return None, [f"predictions/master_predictions_round{round_n}.csv"]

    class_decisions = per_class.get("class_decisions", {})
    class_thr: dict[str, float | None] = {
        cls: (recheck_threshold if d.get("decision") == "recheck"
              else confidence_threshold if d.get("decision") == "commit"
              else None)
        for cls, d in class_decisions.items()
    }

    needs_review = 0
    applicable = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            cls = row.get("class_name")
            thr = class_thr.get(cls, confidence_threshold)
            if thr is None:
                continue
            try:
                if float(row.get("confidence", 1.0)) < thr:
                    needs_review += 1
                applicable += 1
            except (ValueError, TypeError):
                continue
    return (needs_review / applicable if applicable > 0 else 0.0), []


def compute_signals(round_n: int, margin_threshold: float) -> dict:
    """Read all 3 input files and extract the 5 signals + missing-inputs list."""
    per_class = load_per_class_accuracy(round_n)
    model = load_model_metrics(round_n)
    uncertain = count_uncertain_pools(round_n, margin_threshold)
    needs_review_ratio, nr_missing = compute_needs_review_ratio(round_n, per_class)

    missing = []
    if not per_class:
        missing.append(f"predictions/per_class_accuracy_round{round_n}.json")
    if not model:
        missing.append(f"models/round{round_n}/model.json")
    master_path = f"predictions/master_predictions_round{round_n}.csv"
    if uncertain < 0 and master_path not in missing:
        missing.append(master_path)
    for m in nr_missing:
        if m not in missing:
            missing.append(m)

    class_decisions = per_class.get("class_decisions", {})
    n_disabled = sum(1 for d in class_decisions.values() if d == "disabled")

    return {
        "fleiss_kappa": per_class.get("fleiss_kappa"),
        "n_disabled": n_disabled,
        "val_accuracy": model.get("best_val_accuracy"),
        "uncertain_pool": uncertain if uncertain >= 0 else None,
        "needs_review_ratio": needs_review_ratio,
        "missing_inputs": missing,
    }


def compute_verdict(signals: dict, thresholds: dict) -> str:
    """Map signal flags to STOP / CONTINUE / INVESTIGATE string."""
    if signals.get("missing_inputs"):
        return "investigate"

    kappa = signals.get("fleiss_kappa")
    n_disabled = signals.get("n_disabled", -1)
    val_acc = signals.get("val_accuracy")
    uncertain = signals.get("uncertain_pool")
    needs_review = signals.get("needs_review_ratio")
    acc_delta = signals.get("acc_delta")

    flags = [
        kappa is not None and kappa >= thresholds["kappa"],
        n_disabled == 0,
        acc_delta is not None and acc_delta < thresholds["acc_delta"],
        uncertain is not None and uncertain < thresholds["uncertain_pool"],
        needs_review is not None and needs_review < thresholds["needs_review_ratio"],
    ]
    n_met = sum(1 for f in flags if f)

    if n_met == 5:
        return "stop"
    if n_met >= 2:
        return "investigate"
    return "continue"


def collect_trend(round_n: int, margin_threshold: float) -> list:
    """Scan all rounds 1..N and return trend data list."""
    trend = []
    for r in range(1, round_n + 1):
        r_per_class = load_per_class_accuracy(r)
        r_model = load_model_metrics(r)
        r_uncertain = count_uncertain_pools(r, margin_threshold)
        class_decisions = r_per_class.get("class_decisions", {})
        n_disabled = sum(1 for d in class_decisions.values() if d == "disabled")
        trend.append({
            "round": r,
            "kappa": r_per_class.get("fleiss_kappa"),
            "val_acc": r_model.get("best_val_accuracy"),
            "f1": r_model.get("avg_per_class_f1"),
            "n_disabled": n_disabled,
            "uncertain_pool": r_uncertain if r_uncertain >= 0 else None,
        })
    return trend


def format_text_output(round_n, signals, trend, verdict, thresholds):
    """Format human-readable multi-line output with PASS/FAIL per signal."""
    lines = [f"=== Active Learning Stop Check - Round {round_n} ===", ""]

    if signals["missing_inputs"]:
        lines.append("MISSING INPUT FILES:")
        for m in signals["missing_inputs"]:
            lines.append(f"  - {m}")
        lines.append("")
        lines.append(">>> VERDICT: INVESTIGATE")
        lines.append("    Round not complete. Run missing phases first.")
        return "\n".join(lines)

    kappa = signals["fleiss_kappa"]
    if kappa is not None:
        interp = (
            "excellent" if kappa >= 0.81
            else "substantial" if kappa >= 0.61
            else "moderate" if kappa >= 0.41
            else "fair" if kappa >= 0.21
            else "slight" if kappa >= 0.0
            else "poor"
        )
        kappa_str = f"{kappa:.4f}  ({interp})"
    else:
        kappa_str = "N/A"
    kappa_pass = kappa is not None and kappa >= thresholds["kappa"]
    disabled_pass = signals["n_disabled"] == 0

    lines.append("Phase 2 verify:")
    lines.append(f"  Fleiss' Kappa:        {kappa_str:<32} "
                 f"[threshold: {thresholds['kappa']}]  {'PASS' if kappa_pass else 'FAIL'}")
    lines.append(f"  Class decisions:      {signals['n_disabled']} disabled              "
                 f"[target: 0]            {'PASS' if disabled_pass else 'FAIL'}")
    lines.append("")

    val_acc = signals["val_accuracy"]
    acc_delta = signals.get("acc_delta")
    plateau_pass = acc_delta is not None and acc_delta < thresholds["acc_delta"]
    lines.append("Phase 5 train:")
    if val_acc is not None:
        lines.append(f"  Val accuracy:         {val_acc:.4f}")
    else:
        lines.append("  Val accuracy:         N/A")
    if acc_delta is not None:
        lines.append(f"  Delta from prev:      {acc_delta:+.4f}                    "
                     f"[plateau: < {thresholds['acc_delta']}]  {'PASS' if plateau_pass else 'FAIL'}")
    else:
        lines.append("  Delta from prev:      N/A (no previous round for comparison)")
    lines.append("")

    uncertain = signals["uncertain_pool"]
    unc_pass = uncertain is not None and uncertain < thresholds["uncertain_pool"]
    unc_str = f"{uncertain:,} patches" if uncertain is not None else "N/A"
    lines.append("Phase 1 inference:")
    lines.append(f"  Uncertain pool:       {unc_str:<32} "
                 f"[threshold: < {thresholds['uncertain_pool']}, margin < {thresholds['margin']}]  "
                 f"{'PASS' if unc_pass else 'FAIL'}")

    needs_review = signals.get("needs_review_ratio")
    nr_pass = needs_review is not None and needs_review < thresholds["needs_review_ratio"]
    if needs_review is not None:
        nr_str = f"{needs_review * 100:.2f}% of applicable"
    else:
        nr_str = "N/A"
    lines.append(f"  Needs-review ratio:   {nr_str:<32} "
                 f"[threshold: < {thresholds['needs_review_ratio'] * 100:.0f}%, per-class threshold]  "
                 f"{'PASS' if nr_pass else 'FAIL'}")
    lines.append("")

    if trend:
        lines.append("Trend (--all-rounds):")
        lines.append(f"  {'Round':<6} {'Kappa':<10} {'Val Acc':<10} {'F1':<10} "
                     f"{'Disabled':<10} {'Uncertain':<12}")
        for r in trend:
            kappa_r = f"{r['kappa']:.4f}" if r.get("kappa") is not None else "N/A"
            val_r = f"{r['val_acc']:.4f}" if r.get("val_acc") is not None else "N/A"
            f1_r = f"{r['f1']:.4f}" if r.get("f1") is not None else "N/A"
            unc_r = f"{r['uncertain_pool']:,}" if r.get("uncertain_pool") is not None else "N/A"
            disabled_r = str(r.get("n_disabled", "N/A"))
            lines.append(f"  {r['round']:<6} {kappa_r:<10} {val_r:<10} {f1_r:<10} "
                         f"{disabled_r:<10} {unc_r:<12}")
        lines.append("")

    n_met = sum([kappa_pass, disabled_pass, plateau_pass, unc_pass, nr_pass])
    lines.append(f"Stop signals met: {n_met} of 5")
    lines.append("")

    verdict_messages = {
        "stop": "All 5 signals met. AL has converged.",
        "continue": "Less than 2 signals met. Another round likely productive.",
        "investigate": "Mixed signals. Manual review recommended.",
    }
    lines.append(f">>> VERDICT: {verdict.upper()}")
    lines.append(f"    {verdict_messages[verdict]}")

    return "\n".join(lines)


def format_json_output(round_n, signals, verdict, thresholds, trend):
    """Format machine-readable JSON output for programmatic use."""
    kappa = signals.get("fleiss_kappa")
    n_disabled = signals.get("n_disabled", -1)
    val_acc = signals.get("val_accuracy")
    uncertain = signals.get("uncertain_pool")
    acc_delta = signals.get("acc_delta")
    needs_review = signals.get("needs_review_ratio")

    flags = [
        kappa is not None and kappa >= thresholds["kappa"],
        n_disabled == 0,
        acc_delta is not None and acc_delta < thresholds["acc_delta"],
        uncertain is not None and uncertain < thresholds["uncertain_pool"],
        needs_review is not None and needs_review < thresholds["needs_review_ratio"],
    ]
    n_met = sum(1 for f in flags if f)

    out = {
        "round": round_n,
        "verdict": verdict,
        "signals": {
            "fleiss_kappa": kappa,
            "kappa_met": flags[0],
            "n_disabled": n_disabled,
            "disabled_met": flags[1],
            "val_accuracy": val_acc,
            "prev_val_accuracy": signals.get("prev_val_accuracy"),
            "acc_delta": acc_delta,
            "plateau_met": flags[2],
            "uncertain_pool": uncertain,
            "uncertain_pool_met": flags[3],
            "needs_review_ratio": needs_review,
            "needs_review_met": flags[4],
        },
        "n_stop_signals": n_met,
        "thresholds": thresholds,
        "missing_inputs": signals.get("missing_inputs", []),
        "trend": trend,
    }
    return json.dumps(out, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Post-Phase-5 verdict helper for the active learning loop.",
    )
    parser.add_argument("--round", type=int, required=True, help="Round to evaluate (required)")
    parser.add_argument("--all-rounds", action="store_true",
                        help="Show trend table for rounds 1..N")
    parser.add_argument("--stop-kappa", type=float, default=DEFAULT_STOP_KAPPA,
                        help="Kappa threshold for STOP (default: 0.81)")
    parser.add_argument("--stop-acc-delta", type=float, default=DEFAULT_STOP_ACC_DELTA,
                        help="Val acc delta plateau (default: 0.005)")
    parser.add_argument("--stop-uncertain-pool", type=int, default=DEFAULT_STOP_UNCERTAIN_POOL,
                        help="Uncertain pool threshold for STOP (default: 50)")
    parser.add_argument("--stop-needs-review-ratio", type=float,
                        default=DEFAULT_STOP_NEEDS_REVIEW_RATIO,
                        help="Needs-review ratio threshold for STOP (default: 0.01 = 1%%)")
    parser.add_argument("--margin-threshold", type=float, default=DEFAULT_MARGIN_THRESHOLD,
                        help="Margin cutoff for uncertain (default: 0.2)")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="JSON output to stdout")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Print verdict line only, no tables")
    args = parser.parse_args()

    thresholds = {
        "kappa": args.stop_kappa,
        "acc_delta": args.stop_acc_delta,
        "uncertain_pool": args.stop_uncertain_pool,
        "needs_review_ratio": args.stop_needs_review_ratio,
        "margin": args.margin_threshold,
    }

    signals = compute_signals(args.round, args.margin_threshold)
    trend = collect_trend(args.round, args.margin_threshold) if args.all_rounds else []

    # ponytail: val_acc delta must work without --all-rounds, since it is one of
    # the 4 verdict signals. Load round N-1 model directly to compute delta.
    if args.round > 1 and signals.get("val_accuracy") is not None:
        prev_model = load_model_metrics(args.round - 1)
        prev_val = prev_model.get("best_val_accuracy")
        if prev_val is not None:
            signals["prev_val_accuracy"] = prev_val
            signals["acc_delta"] = signals["val_accuracy"] - prev_val

    verdict = compute_verdict(signals, thresholds)

    if args.as_json:
        print(format_json_output(args.round, signals, verdict, thresholds, trend))
    elif args.quiet:
        print(f"round {args.round}: {verdict.upper()}")
    else:
        print(format_text_output(args.round, signals, trend, verdict, thresholds))

    sys.exit({"stop": 0, "continue": 1, "investigate": 2}[verdict])


if __name__ == "__main__":
    main()
