#!/usr/bin/env python3
"""BiblAI Evaluation Benchmark Dashboard Generator.

Reads JSONL files from backend/eval/results/ and generates a self-contained,
interactive HTML dashboard with KPIs, category breakdown, confusion matrix,
retrieval recall comparisons, LLM-as-a-judge metrics, and a searchable claims explorer.
"""

import json
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "eval" / "results"
ACCURACY_PATH = RESULTS_DIR / "verdict_accuracy.jsonl"
RETRIEVAL_PATH = RESULTS_DIR / "retrieval_eval.jsonl"
JUDGE_PATH = RESULTS_DIR / "llm_judge.jsonl"
OUTPUT_HTML_PATH = BASE_DIR / "eval" / "dashboard.html"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_metrics(accuracy_data: list[dict], retrieval_data: list[dict], judge_data: list[dict]) -> dict:
    total_claims = len(accuracy_data)
    correct_count = sum(1 for item in accuracy_data if item.get("correct", False))
    accuracy_pct = (correct_count / total_claims * 100) if total_claims > 0 else 0.0

    by_category = defaultdict(lambda: {"correct": 0, "total": 0, "items": []})
    verdicts = ["directly_supported", "directly_contradicted", "not_directly_stated"]
    confusion_matrix = {exp: {act: 0 for act in verdicts} for exp in verdicts}

    for item in accuracy_data:
        cat = item.get("category", "unknown")
        is_corr = item.get("correct", False)
        by_category[cat]["total"] += 1
        if is_corr:
            by_category[cat]["correct"] += 1
        by_category[cat]["items"].append(item)

        exp = item.get("expected_verdict")
        act = item.get("actual_verdict")
        if exp in confusion_matrix and act in confusion_matrix[exp]:
            confusion_matrix[exp][act] += 1

    category_stats = []
    for cat, stats in sorted(by_category.items()):
        total = stats["total"]
        corr = stats["correct"]
        pct = (corr / total * 100) if total > 0 else 0.0
        category_stats.append(
            {
                "category": cat,
                "correct": corr,
                "total": total,
                "accuracy_pct": round(pct, 1),
            }
        )

    retrieval_total = len(retrieval_data)
    bi_encoder_hits = sum(
        1 for r in retrieval_data if r.get("bi_encoder_top10_hit", False) or r.get("bi_encoder_hit", False)
    )
    rerank_hits = sum(1 for r in retrieval_data if r.get("rerank_top5_hit", False) or r.get("rerank_hit", False))

    bi_encoder_recall = (bi_encoder_hits / retrieval_total * 100) if retrieval_total > 0 else 0.0
    rerank_recall = (rerank_hits / retrieval_total * 100) if retrieval_total > 0 else 0.0

    judge_total = len(judge_data)
    judge_verdict_correct = 0
    judge_reasoning_faithful = 0
    judge_issues_count = 0

    for item in judge_data:
        j = item.get("judge", {})
        if j.get("verdict_is_correct", False):
            judge_verdict_correct += 1
        if j.get("reasoning_follows_from_citations", False):
            judge_reasoning_faithful += 1
        issues = j.get("issues", "").strip().lower()
        if issues and issues not in ["none", "none.", "brak", "brak uwag"]:
            judge_issues_count += 1

    judge_verdict_pct = (judge_verdict_correct / judge_total * 100) if judge_total > 0 else 0.0
    judge_reasoning_pct = (judge_reasoning_faithful / judge_total * 100) if judge_total > 0 else 0.0

    judge_map = {item.get("id"): item.get("judge") for item in judge_data if "id" in item}
    retrieval_map = {item.get("id"): item for item in retrieval_data if "id" in item}

    merged_claims = []
    for item in accuracy_data:
        cid = item.get("id")
        merged = dict(item)
        merged["judge"] = judge_map.get(cid)
        merged["retrieval"] = retrieval_map.get(cid)
        merged_claims.append(merged)

    return {
        "total_claims": total_claims,
        "correct_count": correct_count,
        "accuracy_pct": round(accuracy_pct, 1),
        "categories_count": len(category_stats),
        "category_stats": category_stats,
        "confusion_matrix": confusion_matrix,
        "retrieval": {
            "total": retrieval_total,
            "bi_encoder_hits": bi_encoder_hits,
            "bi_encoder_recall": round(bi_encoder_recall, 1),
            "rerank_hits": rerank_hits,
            "rerank_recall": round(rerank_recall, 1),
        },
        "judge": {
            "total": judge_total,
            "verdict_correct_pct": round(judge_verdict_pct, 1),
            "reasoning_faithful_pct": round(judge_reasoning_pct, 1),
            "issues_count": judge_issues_count,
        },
        "claims": merged_claims,
    }


def generate_dashboard_html(metrics: dict) -> str:
    metrics_json = json.dumps(metrics, ensure_ascii=False)

    template = f"""<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BiblAI - Raport Ewaluacji i Benchmarków</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Source+Serif+4:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500;600&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #0F1318;
            --bg-card: #181E26;
            --bg-card-hover: #1E2530;
            --bg-inset: #12161C;
            --border: #2B3542;
            --border-light: #3A4758;
            --text-primary: #F0F4F8;
            --text-secondary: #94A3B8;
            --text-muted: #64748B;
            --accent-gold: #EAB308;
            --accent-green: #10B981;
            --accent-green-bg: rgba(16, 185, 129, 0.12);
            --accent-red: #EF4444;
            --accent-red-bg: rgba(239, 68, 68, 0.12);
            --accent-blue: #3B82F6;
            --accent-blue-bg: rgba(59, 130, 246, 0.12);
            --font-display: 'Fraunces', Georgia, serif;
            --font-ui: 'Plus Jakarta Sans', -apple-system, sans-serif;
            --font-serif: 'Source Serif 4', Georgia, serif;
            --font-mono: 'JetBrains Mono', monospace;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            background: var(--bg);
            color: var(--text-primary);
            font-family: var(--font-ui);
            line-height: 1.5;
            padding: 36px 20px 80px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        /* Header */
        header {{
            margin-bottom: 32px;
            padding-bottom: 24px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            flex-wrap: wrap;
            gap: 16px;
        }}

        .header-title h1 {{
            font-family: var(--font-display);
            font-size: 32px;
            font-weight: 600;
            color: var(--text-primary);
            letter-spacing: -0.02em;
            margin-bottom: 6px;
        }}

        .header-title p {{
            color: var(--text-secondary);
            font-size: 15px;
        }}

        .badge-live {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: #34D399;
            padding: 6px 12px;
            border-radius: 9999px;
            font-family: var(--font-mono);
            font-size: 12px;
            font-weight: 500;
        }}

        .badge-dot {{
            width: 7px;
            height: 7px;
            background: #10B981;
            border-radius: 50%;
            box-shadow: 0 0 8px #10B981;
        }}

        /* KPI Grid */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }}

        .kpi-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            transition: transform 0.15s ease, border-color 0.15s ease;
        }}

        .kpi-card:hover {{
            transform: translateY(-2px);
            border-color: var(--border-light);
        }}

        .kpi-label {{
            font-size: 13px;
            font-weight: 500;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
        }}

        .kpi-value {{
            font-family: var(--font-display);
            font-size: 36px;
            font-weight: 600;
            color: var(--text-primary);
            line-height: 1.1;
        }}

        .kpi-subtext {{
            font-size: 13px;
            color: var(--text-muted);
            margin-top: 6px;
        }}

        .color-green {{ color: var(--accent-green) !important; }}
        .color-gold {{ color: var(--accent-gold) !important; }}
        .color-blue {{ color: var(--accent-blue) !important; }}

        /* Main Section Layout */
        .dashboard-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-bottom: 32px;
        }}

        @media (max-width: 900px) {{
            .dashboard-grid {{ grid-template-columns: 1fr; }}
        }}

        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
        }}

        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }}

        .card-title {{
            font-size: 17px;
            font-weight: 600;
            color: var(--text-primary);
        }}

        /* Category Bar List */
        .category-row {{
            margin-bottom: 16px;
        }}

        .category-meta {{
            display: flex;
            justify-content: space-between;
            font-size: 13px;
            margin-bottom: 6px;
        }}

        .category-name {{
            font-family: var(--font-mono);
            color: var(--text-primary);
            font-weight: 500;
        }}

        .category-score {{
            color: var(--text-secondary);
        }}

        .progress-bar-bg {{
            background: var(--bg-inset);
            height: 10px;
            border-radius: 9999px;
            overflow: hidden;
            border: 1px solid var(--border);
        }}

        .progress-bar-fill {{
            height: 100%;
            border-radius: 9999px;
            background: linear-gradient(90deg, #10B981, #34D399);
            transition: width 0.6s ease-in-out;
        }}

        .progress-bar-fill.warn {{
            background: linear-gradient(90deg, #F59E0B, #FBBF24);
        }}

        .progress-bar-fill.alert {{
            background: linear-gradient(90deg, #EF4444, #F87171);
        }}

        /* Confusion Matrix */
        .matrix-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            text-align: center;
            margin-top: 10px;
        }}

        .matrix-table th, .matrix-table td {{
            padding: 12px 8px;
            border: 1px solid var(--border);
        }}

        .matrix-table th {{
            background: var(--bg-inset);
            color: var(--text-secondary);
            font-weight: 600;
            font-size: 12px;
        }}

        .matrix-cell-diagonal {{
            background: rgba(16, 185, 129, 0.15);
            color: #34D399;
            font-weight: 700;
            font-size: 16px;
            font-family: var(--font-mono);
        }}

        .matrix-cell-off {{
            background: rgba(239, 68, 68, 0.08);
            color: #F87171;
            font-family: var(--font-mono);
        }}

        .matrix-cell-zero {{
            color: var(--text-muted);
            font-family: var(--font-mono);
        }}

        /* Retrieval & Judge Cards */
        .metrics-pair {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-top: 12px;
        }}

        .metric-box {{
            background: var(--bg-inset);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }}

        .metric-box-val {{
            font-family: var(--font-display);
            font-size: 28px;
            font-weight: 600;
            margin-top: 4px;
        }}

        .metric-box-sub {{
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 4px;
        }}

        /* Searchable Explorer Table */
        .explorer-section {{
            margin-top: 36px;
        }}

        .explorer-controls {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-bottom: 16px;
            align-items: center;
            justify-content: space-between;
        }}

        .search-input {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 10px 14px;
            border-radius: 8px;
            font-size: 14px;
            min-width: 280px;
            font-family: var(--font-ui);
        }}

        .search-input:focus {{
            outline: 2px solid var(--accent-blue);
            border-color: transparent;
        }}

        .filter-buttons {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }}

        .filter-btn {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-secondary);
            padding: 8px 14px;
            border-radius: 6px;
            font-size: 13px;
            cursor: pointer;
            transition: all 0.15s ease;
        }}

        .filter-btn:hover, .filter-btn.active {{
            background: var(--bg-inset);
            color: var(--text-primary);
            border-color: var(--text-primary);
        }}

        .table-wrap {{
            overflow-x: auto;
            border: 1px solid var(--border);
            border-radius: 12px;
            background: var(--bg-card);
        }}

        .claims-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
            text-align: left;
        }}

        .claims-table th {{
            background: var(--bg-inset);
            padding: 14px 16px;
            color: var(--text-secondary);
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border-bottom: 1px solid var(--border);
        }}

        .claims-table td {{
            padding: 14px 16px;
            border-bottom: 1px solid var(--border);
            vertical-align: top;
        }}

        .claims-table tr:hover {{
            background: var(--bg-card-hover);
        }}

        .verdict-tag {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-family: var(--font-mono);
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
        }}

        .verdict-tag.directly_supported {{
            background: var(--accent-green-bg);
            color: #34D399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }}

        .verdict-tag.directly_contradicted {{
            background: var(--accent-red-bg);
            color: #F87171;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }}

        .verdict-tag.not_directly_stated {{
            background: var(--accent-blue-bg);
            color: #60A5FA;
            border: 1px solid rgba(59, 130, 246, 0.3);
        }}

        .status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 4px;
            font-weight: 600;
            font-size: 13px;
        }}

        .status-pill.pass {{ color: var(--accent-green); }}
        .status-pill.fail {{ color: var(--accent-red); }}

        .details-box {{
            font-family: var(--font-serif);
            font-size: 13.5px;
            color: var(--text-secondary);
            margin-top: 8px;
            line-height: 1.6;
            background: var(--bg-inset);
            padding: 10px 14px;
            border-radius: 6px;
            border-left: 3px solid var(--border-light);
        }}

        .citation-chip {{
            display: inline-block;
            background: rgba(234, 179, 8, 0.12);
            color: #FDE047;
            border: 1px solid rgba(234, 179, 8, 0.3);
            border-radius: 4px;
            padding: 2px 6px;
            font-family: var(--font-mono);
            font-size: 11px;
            margin-right: 4px;
            margin-bottom: 4px;
        }}

        footer {{
            margin-top: 48px;
            text-align: center;
            color: var(--text-muted);
            font-size: 13px;
            border-top: 1px solid var(--border);
            padding-top: 24px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="header-title">
                <h1>BiblAI - Raport Ewaluacji i Benchmarków</h1>
                <p>Analiza jakości agenta weryfikacji faktów biblijnych (LangGraph + HyDE + Hybrid Search + LLM Judge)</p>
            </div>
            <div class="badge-live">
                <span class="badge-dot"></span>
                <span>Wygenerowano automatycznie</span>
            </div>
        </header>

        <!-- KPI Cards -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Dokładność Werdyktów</div>
                <div class="kpi-value color-green">{metrics["accuracy_pct"]}%</div>
                <div class="kpi-subtext">{metrics["correct_count"]} z {
        metrics["total_claims"]
    } poprawnych klasyfikacji</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Recall @ Rerank Top-5</div>
                <div class="kpi-value color-gold">{metrics["retrieval"]["rerank_recall"]}%</div>
                <div class="kpi-subtext">Bi-encoder Recall@20: {metrics["retrieval"]["bi_encoder_recall"]}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Zgodność Sędziego (Judge)</div>
                <div class="kpi-value color-blue">{metrics["judge"]["verdict_correct_pct"]}%</div>
                <div class="kpi-subtext">Wierność uzasadnienia: {metrics["judge"]["reasoning_faithful_pct"]}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Zbadane Kategorie</div>
                <div class="kpi-value">{metrics["categories_count"]}</div>
                <div class="kpi-subtext">{metrics["total_claims"]} sprawdzonych stwierdzeń</div>
            </div>
        </div>

        <!-- 2 Column Analytics -->
        <div class="dashboard-grid">
            <!-- Category Breakdown -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">Skuteczność wg Kategorii Stwierdzeń</div>
                </div>
                <div id="category-bars">
                    {
        "".join(
            f'''
                    <div class="category-row">
                        <div class="category-meta">
                            <span class="category-name">{cat['category']}</span>
                            <span class="category-score">{cat['correct']}/{cat['total']} ({cat['accuracy_pct']}%)</span>
                        </div>
                        <div class="progress-bar-bg">
                            <div class="progress-bar-fill {'alert' if cat['accuracy_pct'] < 60 else ('warn' if cat['accuracy_pct'] < 85 else '')}" style="width: {cat['accuracy_pct']}%;"></div>
                        </div>
                    </div>
                    '''
            for cat in metrics["category_stats"]
        )
    }
                </div>
            </div>

            <!-- Confusion Matrix -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">Macierz Pomyłek (Oczekiwany vs Faktyczny)</div>
                </div>
                <table class="matrix-table">
                    <thead>
                        <tr>
                            <th>Oczekiwany / Faktyczny</th>
                            <th>Supported</th>
                            <th>Contradicted</th>
                            <th>Not Stated</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <th>Directly Supported</th>
                            <td class="matrix-cell-diagonal">{
        metrics["confusion_matrix"]["directly_supported"]["directly_supported"]
    }</td>
                            <td class="{
        "matrix-cell-off"
        if metrics["confusion_matrix"]["directly_supported"]["directly_contradicted"] > 0
        else "matrix-cell-zero"
    }">{metrics["confusion_matrix"]["directly_supported"]["directly_contradicted"]}</td>
                            <td class="{
        "matrix-cell-off"
        if metrics["confusion_matrix"]["directly_supported"]["not_directly_stated"] > 0
        else "matrix-cell-zero"
    }">{metrics["confusion_matrix"]["directly_supported"]["not_directly_stated"]}</td>
                        </tr>
                        <tr>
                            <th>Directly Contradicted</th>
                            <td class="{
        "matrix-cell-off"
        if metrics["confusion_matrix"]["directly_contradicted"]["directly_supported"] > 0
        else "matrix-cell-zero"
    }">{metrics["confusion_matrix"]["directly_contradicted"]["directly_supported"]}</td>
                            <td class="matrix-cell-diagonal">{
        metrics["confusion_matrix"]["directly_contradicted"]["directly_contradicted"]
    }</td>
                            <td class="{
        "matrix-cell-off"
        if metrics["confusion_matrix"]["directly_contradicted"]["not_directly_stated"] > 0
        else "matrix-cell-zero"
    }">{metrics["confusion_matrix"]["directly_contradicted"]["not_directly_stated"]}</td>
                        </tr>
                        <tr>
                            <th>Not Directly Stated</th>
                            <td class="{
        "matrix-cell-off"
        if metrics["confusion_matrix"]["not_directly_stated"]["directly_supported"] > 0
        else "matrix-cell-zero"
    }">{metrics["confusion_matrix"]["not_directly_stated"]["directly_supported"]}</td>
                            <td class="{
        "matrix-cell-off"
        if metrics["confusion_matrix"]["not_directly_stated"]["directly_contradicted"] > 0
        else "matrix-cell-zero"
    }">{metrics["confusion_matrix"]["not_directly_stated"]["directly_contradicted"]}</td>
                            <td class="matrix-cell-diagonal">{
        metrics["confusion_matrix"]["not_directly_stated"]["not_directly_stated"]
    }</td>
                        </tr>
                    </tbody>
                </table>

                <div class="metrics-pair">
                    <div class="metric-box">
                        <div class="kpi-label">Groundedness Sędziego</div>
                        <div class="metric-box-val color-blue">{metrics["judge"]["reasoning_faithful_pct"]}%</div>
                        <div class="metric-box-sub">Brak halucynacji poza kontekst</div>
                    </div>
                    <div class="metric-box">
                        <div class="kpi-label">Retrieval Hits</div>
                        <div class="metric-box-val color-gold">{metrics["retrieval"]["rerank_hits"]}/{
        metrics["retrieval"]["total"]
    }</div>
                        <div class="metric-box-sub">Trafienia w top-5 wersetów</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Explorer Section -->
        <div class="explorer-section card">
            <div class="card-header">
                <div class="card-title">Eksplorator Zbadanych Przypadków Testowych</div>
                <div class="explorer-controls">
                    <input type="text" id="search-box" class="search-input" placeholder="Szukaj po treści, ID lub wersecie...">
                    <div class="filter-buttons">
                        <button class="filter-btn active" onclick="filterTable('all')">Wszystkie ({
        metrics["total_claims"]
    })</button>
                        <button class="filter-btn" onclick="filterTable('correct')">Poprawne ({
        metrics["correct_count"]
    })</button>
                        <button class="filter-btn" onclick="filterTable('wrong')">Błędne ({
        metrics["total_claims"] - metrics["correct_count"]
    })</button>
                    </div>
                </div>
            </div>

            <div class="table-wrap">
                <table class="claims-table" id="claims-table">
                    <thead>
                        <tr>
                            <th style="width: 80px;">ID</th>
                            <th>Stwierdzenie & Uzasadnienie</th>
                            <th style="width: 140px;">Kategoria</th>
                            <th style="width: 160px;">Werdykt (Oczekiwany / Otrzymany)</th>
                            <th style="width: 90px;">Status</th>
                        </tr>
                    </thead>
                    <tbody id="claims-tbody">
                    </tbody>
                </table>
            </div>
        </div>

        <footer>
            BiblAI Benchmark Dashboard &bull; Biblia Tysiąclecia Claim Verification Engine
        </footer>
    </div>

    <script>
        const data = {metrics_json};
        let currentFilter = 'all';

        function renderTable() {{
            const tbody = document.getElementById('claims-tbody');
            const searchVal = document.getElementById('search-box').value.toLowerCase();
            tbody.innerHTML = '';

            const filtered = data.claims.filter(item => {{
                if (currentFilter === 'correct' && !item.correct) return false;
                if (currentFilter === 'wrong' && item.correct) return false;

                if (searchVal) {{
                    const matchText = (item.statement || '').toLowerCase();
                    const matchId = (item.id || '').toLowerCase();
                    const matchCat = (item.category || '').toLowerCase();
                    const matchReasoning = (item.reasoning || '').toLowerCase();
                    const matchCitations = (item.citations || []).map(c => c.ref).join(' ').toLowerCase();
                    return matchText.includes(searchVal) || matchId.includes(searchVal) || matchCat.includes(searchVal) || matchReasoning.includes(searchVal) || matchCitations.includes(searchVal);
                }}
                return true;
            }});

            if (filtered.length === 0) {{
                tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; padding: 24px; color: var(--text-muted);">Brak wyników spełniających kryteria.</td></tr>';
                return;
            }}

            filtered.forEach(item => {{
                const tr = document.createElement('tr');
                const isPass = item.correct;

                const citationsHtml = (item.citations || []).map(c =>
                    `<span class="citation-chip">${{c.ref}}</span>`
                ).join('');

                let judgeHtml = '';
                if (item.judge) {{
                    const jCorrect = item.judge.verdict_is_correct ? '✅ Werdykt OK' : '❌ Werdykt zakwestionowany';
                    const jFaithful = item.judge.reasoning_follows_from_citations ? '✅ Uzasadnienie ugruntowane' : '⚠️ Uzasadnienie nie wynika z cytatów';
                    judgeHtml = `<div style="margin-top: 6px; font-size: 12px; color: var(--text-muted);">${{jCorrect}} &bull; ${{jFaithful}}</div>`;
                }}

                tr.innerHTML = `
                    <td style="font-family: var(--font-mono); font-size: 12px; color: var(--text-muted);">${{item.id}}</td>
                    <td>
                        <div style="font-weight: 600; font-size: 15px; margin-bottom: 4px;">${{item.statement}}</div>
                        <div style="margin-bottom: 6px;">${{citationsHtml}}</div>
                        <div class="details-box">${{item.reasoning || '(brak uzasadnienia)'}}</div>
                        ${{judgeHtml}}
                    </td>
                    <td style="font-family: var(--font-mono); font-size: 12px;">${{item.category}}</td>
                    <td>
                        <div style="margin-bottom: 4px;"><span class="verdict-tag ${{item.actual_verdict}}">${{item.actual_verdict.replace('_', ' ')}}</span></div>
                        <div style="font-size: 11px; color: var(--text-muted);">Oczekiwany: ${{item.expected_verdict.replace('_', ' ')}}</div>
                    </td>
                    <td>
                        <span class="status-pill ${{isPass ? 'pass' : 'fail'}}">
                            ${{isPass ? '✔ Zgoda' : '✘ Błąd'}}
                        </span>
                    </td>
                `;
                tbody.appendChild(tr);
            }});
        }}

        function filterTable(type) {{
            currentFilter = type;
            document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            renderTable();
        }}

        document.getElementById('search-box').addEventListener('input', renderTable);
        renderTable();
    </script>
</body>
</html>
"""
    return template


def generate_dashboard():
    accuracy_data = load_jsonl(ACCURACY_PATH)
    retrieval_data = load_jsonl(RETRIEVAL_PATH)
    judge_data = load_jsonl(JUDGE_PATH)

    if not accuracy_data and not retrieval_data and not judge_data:
        print(f"[WARN] No evaluation result files found in {RESULTS_DIR}")

    metrics = compute_metrics(accuracy_data, retrieval_data, judge_data)
    html_content = generate_dashboard_html(metrics)

    OUTPUT_HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML_PATH.write_text(html_content, encoding="utf-8")
    print(f"[Dashboard] Successfully generated dashboard at: {OUTPUT_HTML_PATH}")
    print(
        f"[Dashboard] Overall Accuracy: {metrics['accuracy_pct']}% ({metrics['correct_count']}/{metrics['total_claims']})"
    )
    print(f"[Dashboard] Categories Evaluated: {metrics['categories_count']}")


if __name__ == "__main__":
    generate_dashboard()
