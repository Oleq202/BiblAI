from typing import TypedDict

from langgraph.graph import StateGraph, END

try:
    from .tools import retrieve, rerank, classify_support, USE_CROSS_ENCODER
    from .schemas import StatementVerdict
except (ImportError, ValueError):
    from tools import retrieve, rerank, classify_support, USE_CROSS_ENCODER
    from schemas import StatementVerdict

RETRIEVE_TOP_K = 20
RERANK_TOP_K = 5
BROADER_RETRIEVE_TOP_K = 30
RELEVANCE_SCORE_THRESHOLD = 0.35


class AgentState(TypedDict):
    statement: str
    chunks: list[dict]
    rerank_scores: list[float]
    retrieval_attempts: int
    verdict: StatementVerdict | None


def retrieve_and_rerank_node(state):
    top_k = RETRIEVE_TOP_K if state.get("retrieval_attempts", 0) == 0 else BROADER_RETRIEVE_TOP_K
    candidates, scores = retrieve(state["statement"], top_k=top_k, return_scores=True)
    reranked = rerank(state["statement"], candidates, scores=scores)

    top_chunks = [chunk for chunk, score in reranked[:RERANK_TOP_K]]
    top_scores = [float(score) for chunk, score in reranked[:RERANK_TOP_K]]

    return {
        **state,
        "chunks": top_chunks,
        "rerank_scores": top_scores,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
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
    graph.add_node("retrieve_and_rerank", retrieve_and_rerank_node)
    graph.add_node("classify", classify_node)

    graph.set_entry_point("retrieve_and_rerank")
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
        "chunks": [],
        "rerank_scores": [],
        "retrieval_attempts": 0,
        "verdict": None,
    })
    return result["verdict"]
