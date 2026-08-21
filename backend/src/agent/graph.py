from typing import TypedDict

from langgraph.graph import StateGraph, END

try:
    from .tools import (
        retrieve,
        multi_retrieve,
        hybrid_retrieve,
        expand_query,
        rerank,
        classify_support,
        USE_CROSS_ENCODER,
    )
    from .schemas import StatementVerdict
except (ImportError, ValueError):
    from tools import (
        retrieve,
        multi_retrieve,
        hybrid_retrieve,
        expand_query,
        rerank,
        classify_support,
        USE_CROSS_ENCODER,
    )
    from schemas import StatementVerdict

DENSE_TOP_K = 15
SPARSE_TOP_K = 15
FUSED_TOP_N = 25
RERANK_TOP_K = 5

BROADER_DENSE_TOP_K = 25
BROADER_SPARSE_TOP_K = 25
BROADER_FUSED_TOP_N = 35
RELEVANCE_SCORE_THRESHOLD = 0.35


class AgentState(TypedDict):
    statement: str
    expanded_queries: list[str]
    chunks: list[dict]
    rerank_scores: list[float]
    retrieval_attempts: int
    verdict: StatementVerdict | None


def expand_query_node(state):
    expanded = expand_query(state["statement"])
    return {
        **state,
        "expanded_queries": expanded,
    }


def retrieve_and_rerank_node(state):
    queries = state.get("expanded_queries") or [state["statement"]]
    attempt = state.get("retrieval_attempts", 0)
    top_k_dense = DENSE_TOP_K if attempt == 0 else BROADER_DENSE_TOP_K
    top_k_sparse = SPARSE_TOP_K if attempt == 0 else BROADER_SPARSE_TOP_K
    fused_n = FUSED_TOP_N if attempt == 0 else BROADER_FUSED_TOP_N

    candidates, rrf_scores = hybrid_retrieve(
        queries=queries,
        top_k_dense=top_k_dense,
        top_k_sparse=top_k_sparse,
        fused_top_n=fused_n,
        return_scores=True,
    )
    reranked = rerank(state["statement"], candidates, scores=rrf_scores)

    top_chunks = [chunk for chunk, score in reranked[:RERANK_TOP_K]]
    top_scores = [float(score) for chunk, score in reranked[:RERANK_TOP_K]]

    return {
        **state,
        "chunks": top_chunks,
        "rerank_scores": top_scores,
        "retrieval_attempts": attempt + 1,
    }



def check_relevance(state):
    threshold = 0.0 if USE_CROSS_ENCODER else 0.35
    best_score = max(state["rerank_scores"]) if state["rerank_scores"] else -999
    if best_score < threshold and state.get("retrieval_attempts", 0) < 2:
        return "broaden"
    return "classify"


def classify_node(state):
    verdict = classify_support(state["statement"], state["chunks"])
    return {**state, "verdict": verdict}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("expand_query", expand_query_node)
    graph.add_node("retrieve_and_rerank", retrieve_and_rerank_node)
    graph.add_node("classify", classify_node)

    graph.set_entry_point("expand_query")
    graph.add_edge("expand_query", "retrieve_and_rerank")
    graph.add_conditional_edges(
        "retrieve_and_rerank",
        check_relevance,
        {"broaden": "retrieve_and_rerank", "classify": "classify"},
    )
    graph.add_edge("classify", END)
    return graph.compile()


def verify_statement(statement):
    app = build_graph()
    result = app.invoke({
        "statement": statement,
        "expanded_queries": [],
        "chunks": [],
        "rerank_scores": [],
        "retrieval_attempts": 0,
        "verdict": None,
    })
    return result["verdict"]
