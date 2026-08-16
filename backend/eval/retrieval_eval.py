import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src" / "agent"))

from tools import retrieve, rerank

EVAL_SET_PATH = BASE_DIR / "eval" / "eval_set.jsonl"
RESULTS_PATH = BASE_DIR / "eval" / "results" / "retrieval_eval.jsonl"

RETRIEVE_TOP_K = 20
RERANK_TOP_K = 5


def load_eval_set(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def chunk_contains_ref(chunk, expected_ref_prefix):
    for ref in chunk["verse_refs"]:
        if ref.startswith(expected_ref_prefix):
            return True
    return False


if __name__ == "__main__":
    eval_items = load_eval_set(EVAL_SET_PATH)
    testable = [item for item in eval_items if "expected_ref_contains" in item]

    print(f"{len(testable)} of {len(eval_items)} eval items have a known expected reference\n")

    results = []
    hits_at_bi_encoder = 0
    hits_at_rerank_top5 = 0

    for item in testable:
        expected = item["expected_ref_contains"]

        bi_encoder_candidates = retrieve(item["statement"], top_k=RETRIEVE_TOP_K)
        bi_encoder_hit = any(chunk_contains_ref(c, expected) for c in bi_encoder_candidates)

        reranked = rerank(item["statement"], bi_encoder_candidates)
        top5_chunks = [chunk for chunk, score in reranked[:RERANK_TOP_K]]
        rerank_hit = any(chunk_contains_ref(c, expected) for c in top5_chunks)

        hits_at_bi_encoder += bi_encoder_hit
        hits_at_rerank_top5 += rerank_hit

        status = "[HIT]" if rerank_hit else ("[BI_ONLY]" if bi_encoder_hit else "[MISS]")
        print(f"  {status} [{item['id']}] expected={expected} "
              f"bi_encoder_hit={bi_encoder_hit} rerank_top5_hit={rerank_hit}")

        results.append({
            "id": item["id"],
            "statement": item["statement"],
            "expected_ref_contains": expected,
            "bi_encoder_top10_hit": bi_encoder_hit,
            "rerank_top5_hit": rerank_hit,
        })

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(testable)
    print(f"\n=== Retrieval recall @ bi-encoder top-{RETRIEVE_TOP_K}: "
          f"{hits_at_bi_encoder}/{n} ({100*hits_at_bi_encoder/n:.1f}%) ===")
    print(f"=== Retrieval recall @ rerank top-{RERANK_TOP_K}: "
          f"{hits_at_rerank_top5}/{n} ({100*hits_at_rerank_top5/n:.1f}%) ===")
    print(f"\nResults -> {RESULTS_PATH}")