import json
import re
from pathlib import Path
from difflib import SequenceMatcher

BASE_DIR = Path(__file__).resolve().parents[1]
VERSES_TXT = BASE_DIR / "data" / "raw" / "biblia_tysiaclecia.txt"

VERSE_LINE = re.compile(r"^(.+?)\s(\d+),(\d+)\s-\s(.*)$")

QUOTE_MATCH_THRESHOLD = 0.7

def normalize_ref(ref_str):
    m = re.match(r"^(.+?)\s+(\d+)[:,\.](\d+)$", ref_str.strip())
    if m:
        abbr, ch, v = m.groups()
        return f"{abbr} {int(ch)}:{int(v)}"
    return ref_str.strip()


def load_verse_lookup():
    lookup = {}
    with open(VERSES_TXT, encoding="utf-8") as f:
        for line in f:
            m = VERSE_LINE.match(line.strip())
            if not m:
                continue
            abbr, chapter, verse_num, text = m.groups()
            ref = f"{abbr} {int(chapter)}:{int(verse_num)}"
            lookup[ref] = text.strip()
    return lookup


def check_citation(ref, quote, lookup):
    norm_ref = normalize_ref(ref)
    if norm_ref not in lookup:
        return {"ref": ref, "ref_exists": False, "quote_match": 0.0, "valid": False}

    actual_text = lookup[norm_ref]
    ratio = SequenceMatcher(None, quote, actual_text).ratio()
    contains = quote.strip(' ."\'') in actual_text or actual_text in quote

    valid = contains or ratio >= QUOTE_MATCH_THRESHOLD
    return {"ref": ref, "ref_exists": True, "quote_match": round(ratio, 3), "valid": valid}


if __name__ == "__main__":
    lookup = load_verse_lookup()
    print(f"Loaded {len(lookup)} verses for lookup")

    results_path = BASE_DIR / "eval" / "results" / "verdict_accuracy.jsonl"
    with open(results_path, encoding="utf-8") as f:
        eval_results = [json.loads(line) for line in f]

    all_checks = []
    for item in eval_results:
        for cit in item.get("citations", []):
            check = check_citation(cit["ref"], cit["quote"], lookup)
            check["item_id"] = item["id"]
            all_checks.append(check)

    invalid_refs = [c for c in all_checks if not c["ref_exists"]]
    invalid_quotes = [c for c in all_checks if not c["valid"]]
    print(f"\nTotal citations checked: {len(all_checks)}")
    print(f"Citations with non-existent refs (hallucinated verse numbers): {len(invalid_refs)}")
    for c in invalid_refs:
        print(f"  [{c['item_id']}] {c['ref']} -- NOT FOUND IN CORPUS")

    print(f"Citations with valid quote matching: {len(all_checks) - len(invalid_quotes)}/{len(all_checks)}")
    for c in invalid_quotes:
        print(f"  [{c['item_id']}] {c['ref']} -- quote match ratio: {c['quote_match']}")