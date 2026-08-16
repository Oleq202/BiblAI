import os
import json
import time
import hashlib
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src" / "agent"))

from google import genai
from google.genai import types

JUDGE_MODEL = "models/gemini-3.1-flash-lite"
CACHE_DIR = BASE_DIR / "data" / "llm_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

if "GEMINI_API_KEY" not in os.environ:
    env_file = BASE_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip("'\"")

_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict_is_correct": {"type": "boolean"},
        "reasoning_follows_from_citations": {"type": "boolean"},
        "issues": {"type": "string"},
    },
    "required": ["verdict_is_correct", "reasoning_follows_from_citations", "issues"],
}

JUDGE_SYSTEM_PROMPT = """Jesteś niezależnym recenzentem oceniającym, czy \
werdykt i uzasadnienie systemu AI faktycznie wynikają z podanych cytatów \
biblijnych. Nie ufaj systemowi na słowo - sprawdź samodzielnie, czy cytaty \
rzeczywiście potwierdzają werdykt."""


def _judge_cache_key(statement, verdict, citations, reasoning):
    raw = f"judge||{statement}||{verdict}||{json.dumps(citations, sort_keys=True)}||{reasoning}||{JUDGE_MODEL}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def judge(statement, verdict, citations, reasoning, max_retries=5, use_cache=True):
    cache_path = CACHE_DIR / f"{_judge_cache_key(statement, verdict, citations, reasoning)}.json"
    if use_cache and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    citations_text = "\n".join(f"[{c['ref']}] {c['quote']}" for c in citations) or "(brak cytatów)"

    prompt = f"""Stwierdzenie: "{statement}"

Werdykt systemu: {verdict}

Cytaty podane przez system:
{citations_text}

Uzasadnienie systemu: {reasoning}

Oceń: czy werdykt jest poprawny na podstawie podanych cytatów, i czy \
uzasadnienie faktycznie wynika z tych cytatów (a nie z wiedzy spoza \
podanego kontekstu)?"""

    last_error = None
    for attempt in range(max_retries):
        try:
            response = _client.models.generate_content(
                model=JUDGE_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=JUDGE_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=JUDGE_SCHEMA,
                    max_output_tokens=1024,
                ),
            )
            result = json.loads(response.text)
            if use_cache:
                cache_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            return result
        except Exception as e:
            last_error = e
            err_msg = str(e)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                wait = max(20, 15 * (attempt + 1))
            else:
                wait = 2 ** attempt
            print(f"  attempt {attempt + 1} failed ({e.__class__.__name__}: {e}), retrying in {wait}s...")
            time.sleep(wait)

    raise last_error


if __name__ == "__main__":
    results_path = BASE_DIR / "eval" / "results" / "verdict_accuracy.jsonl"
    with open(results_path, encoding="utf-8") as f:
        eval_results = [json.loads(line) for line in f]

    judged = []
    for item in eval_results:
        print(f"[{item['id']}] judging...")
        verdict_judgment = judge(
            item["statement"], item["actual_verdict"], item["citations"], item["reasoning"]
        )
        judged.append({**item, "judge": verdict_judgment})
        time.sleep(1)

    out_path = BASE_DIR / "eval" / "results" / "llm_judge.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for j in judged:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")

    agree_rate = sum(j["judge"]["verdict_is_correct"] for j in judged) / len(judged)
    print(f"\nJudge agrees with verdict: {agree_rate*100:.1f}%")
    print(f"Results -> {out_path}")
    