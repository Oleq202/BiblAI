import json
import sys
import time
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src" / "agent"))

from graph import verify_statement

EVAL_SET_PATH = BASE_DIR / "eval" / "eval_set.jsonl"
RESULTS_PATH = BASE_DIR / "eval" / "results" / "verdict_accuracy.jsonl"


def load_eval_set(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    eval_items = load_eval_set(EVAL_SET_PATH)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = []
    correct = 0
    by_category = defaultdict(lambda: {"correct": 0, "total": 0})

    for item in eval_items:
        print(f"[{item['id']}] {item['statement'][:60]}...")
        verdict = verify_statement(item["statement"])

        is_correct = verdict.verdict.value == item["expected_verdict"]
        correct += is_correct
        by_category[item["category"]]["total"] += 1
        by_category[item["category"]]["correct"] += is_correct

        results.append({
            "id": item["id"],
            "statement": item["statement"],
            "category": item["category"],
            "expected_verdict": item["expected_verdict"],
            "actual_verdict": verdict.verdict.value,
            "correct": is_correct,
            "confidence": verdict.confidence,
            "citations": [{"ref": c.ref, "quote": c.quote, "relation": c.relation} for c in verdict.citations],
            "reasoning": verdict.reasoning,
        })

        status = "Correct" if is_correct else "Wrong"
        print(f"  {status} expected={item['expected_verdict']} actual={verdict.verdict.value}")
        time.sleep(1)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n=== Overall: {correct}/{len(eval_items)} ({100*correct/len(eval_items):.1f}%) ===")
    print("\n=== By category ===")
    for cat, stats in sorted(by_category.items()):
        pct = 100 * stats["correct"] / stats["total"]
        print(f"  {cat}: {stats['correct']}/{stats['total']} ({pct:.1f}%)")

    print(f"\nFull results -> {RESULTS_PATH}")