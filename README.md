# HF Docs RAG Assistant

An end-to-end **Retrieval-Augmented Generation (RAG)** system that answers questions about the Hugging Face documentation, grounded in retrieved source docs with citations. Built as a portfolio project covering the full Gen AI stack: ingestion, chunking, embeddings, a vector database, retrieval, LLM generation, evaluation, and LoRA fine-tuning.

## What it does

Ask a question → the system retrieves the most relevant documentation chunks from a FAISS vector index → a small LLM generates a grounded answer with numbered citations.

```
                 ┌──────────────┐
   question ───▶ │  embed query  │
                 └──────┬───────┘
                        ▼
                 ┌──────────────┐     top-k chunks
                 │ FAISS search │ ─────────────────┐
                 └──────────────┘                  ▼
                                          ┌──────────────────┐
                                          │  build prompt    │
                                          │  (context + q)   │
                                          └────────┬─────────┘
                                                   ▼
                                          ┌──────────────────┐
                                          │  LLM generation  │ ──▶ answer + citations
                                          └──────────────────┘
```

## Tech stack

| Stage | Tool | Why |
|---|---|---|
| Corpus | [`m-ric/huggingface_doc`](https://huggingface.co/datasets/m-ric/huggingface_doc) | 2.6K real docs, MIT-licensed, `text` + `source` columns |
| Chunking | Recursive character split, 512 chars / 64 overlap | Simple, deterministic, keeps context across boundaries |
| Embeddings | [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 22M params, runs on CPU, strong quality/speed trade-off |
| Vector DB | FAISS (`IndexFlatIP`) | In-memory, no server; inner product == cosine on L2-normalized vectors |
| Generation | [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | Small enough to run locally, real instruction-tuned model |
| Evaluation | recall@k, MRR, lexical grounding | Retrieval quality + a faithfulness smoke signal |
| Fine-tuning | LoRA (PEFT) + TRL `SFTTrainer` | Shows training, not just API calls |

## Design decisions (the "why")

- **Recursive chunking over fixed-size** — fixed-size splits mid-sentence and loses context; overlap mitigates this while staying deterministic and cheap (vs. semantic chunking, which needs an extra model).
- **FAISS over a hosted vector DB** — for a demo, an in-memory index is zero-ops and fast; the retrieval *interface* is identical to Chroma/Qdrant, so swapping is a one-line change.
- **Inner product on normalized embeddings** — normalizing to unit length makes inner product equal cosine similarity, so `IndexFlatIP` gives exact cosine search with no extra code.
- **Small model, run locally** — 0.5B params means the whole pipeline runs on a laptop CPU, no API key or credits required. The same code path works with any larger model.

## Project layout

```
├── README.md
├── requirements.txt
├── app.py              # Gradio demo (run locally)
├── finetune.py         # LoRA fine-tuning script (needs a GPU)
└── src/
    ├── __init__.py
    ├── rag.py          # RAGPipeline: ingest, chunk, embed, index, retrieve, generate
    └── evaluate.py     # recall@k, MRR, lexical grounding
```

## Run it locally

```bash
pip install -r requirements.txt
python app.py            # launches the Gradio demo at http://localhost:7860
```

Or use the pipeline directly:

```python
from src.rag import RAGPipeline

p = RAGPipeline()
p.load_and_chunk(max_docs=500)   # subset for speed
p.build_index()
answer, sources = p.answer("How do I create an Inference Endpoint?")
print(answer)
```

## Evaluation

`src/evaluate.py` measures two layers:

1. **Retrieval** — `recall@k` and **MRR** against a small hand-built set of `(query, expected_source)` pairs.
2. **Answer faithfulness** — a lexical grounding score (fraction of answer tokens present in the retrieved context), a cheap proxy for hallucination.

> Production systems use **RAGAS** (faithfulness, answer relevancy, context precision/recall) with an LLM judge. The lexical check here is a dependency-free smoke signal; swap in RAGAS when you have an API key.

## Fine-tuning

`finetune.py` LoRA-fine-tunes `Qwen2.5-0.5B-Instruct` with TRL's `SFTTrainer`. It needs a GPU (a T4 or A10G is plenty) and a domain Q&A dataset. See the file for the full config and the dataset-swap note.

## Future work

- Hybrid retrieval (BM25 + dense) with a cross-encoder reranker
- RAGAS evaluation with an LLM judge
- Streaming + chat history in the demo
- Deploy as a hosted Space (requires a PRO subscription for Gradio on CPU)

## License

MIT (corpus is MIT-licensed; code is yours to use).
