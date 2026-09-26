"""Evaluation for the RAG pipeline.

Two layers, matching how RAG systems are actually judged:

1. Retrieval quality - does the right document come back?
   Metrics: recall@k and MRR (mean reciprocal rank), against a small
   hand-built set of (query, expected_source) pairs.

2. Answer quality - is the generated answer faithful to the retrieved context?
   A lightweight lexical grounding check (fraction of answer tokens that appear
   in the context). For production, swap in RAGAS (faithfulness, answer
   relevancy) - see README.
"""

from __future__ import annotations

from typing import List, Tuple

from src.rag import RAGPipeline, RetrievedChunk


# A small, hand-written eval set: (query, expected source substring).
# Each query is answerable from a specific doc in m-ric/huggingface_doc.
EVAL_QUERIES: List[Tuple[str, str]] = [
    ("How do I create an Inference Endpoint?", "create_endpoint"),
    ("How do I choose an evaluation metric for my task?", "choosing_a_metric"),
    ("What are the main features of Gradio?", "gradio"),
    ("How do I upload a model to the Hub?", "upload"),
    ("What is a dataset card?", "dataset_card"),
]


def evaluate_retrieval(
    pipeline: RAGPipeline, queries: List[Tuple[str, str]] = EVAL_QUERIES, k: int = 5
) -> dict:
    """Compute recall@k and MRR over the eval set."""
    recall_hits = 0
    reciprocal_ranks = []

    for query, expected in queries:
        results = pipeline.retrieve(query, top_k=k)
        sources = [r.source for r in results]

        # A hit: the expected source appears in the top-k results.
        hit = any(expected in s for s in sources)
        recall_hits += int(hit)

        # Reciprocal rank: 1 / position of the first hit (0 if none).
        rank = next((i + 1 for i, s in enumerate(sources) if expected in s), None)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

    n = len(queries)
    return {
        f"recall@{k}": recall_hits / n,
        "mrr": sum(reciprocal_ranks) / n,
        "num_queries": n,
    }


def lexical_grounding(answer: str, retrieved: List[RetrievedChunk]) -> float:
    """Fraction of answer tokens that appear in the retrieved context.

    A cheap faithfulness proxy: a hallucinated answer shares few tokens with
    its context. Not a substitute for an LLM judge, but a useful smoke signal.
    """
    context = " ".join(c.text for c in retrieved).lower().split()
    context_set = set(context)
    answer_tokens = answer.lower().split()
    if not answer_tokens:
        return 0.0
    grounded = sum(1 for t in answer_tokens if t in context_set)
    return grounded / len(answer_tokens)


def evaluate_answers(
    pipeline: RAGPipeline, queries: List[Tuple[str, str]] = EVAL_QUERIES
) -> dict:
    """Generate answers for each query and score lexical grounding."""
    scores = []
    for query, _ in queries:
        answer, retrieved = pipeline.answer(query)
        scores.append(lexical_grounding(answer, retrieved))
    return {
        "mean_grounding": sum(scores) / len(scores) if scores else 0.0,
        "num_queries": len(scores),
    }


if __name__ == "__main__":
    pipeline = RAGPipeline()
    pipeline.load_and_chunk(max_docs=200)  # subset for a fast eval
    pipeline.build_index()

    print("Retrieval:", evaluate_retrieval(pipeline))
    print("Answers:", evaluate_answers(pipeline))
