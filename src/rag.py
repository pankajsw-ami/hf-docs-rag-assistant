"""End-to-end RAG pipeline over the Hugging Face documentation corpus.

Pipeline:
    load corpus -> chunk -> embed -> FAISS index -> retrieve -> generate

Design notes (see README for the full rationale):
- Recursive character chunking with overlap keeps context across boundaries.
- Embeddings are L2-normalized so FAISS inner-product == cosine similarity.
- Retrieval returns the top-k chunks with their source doc for citation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import faiss
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


@dataclass
class Chunk:
    """A single chunk of a source document."""

    text: str
    source: str
    doc_id: int


@dataclass
class RetrievedChunk:
    """A chunk returned by retrieval, with its similarity score."""

    text: str
    source: str
    score: float


class RAGPipeline:
    """A minimal, dependency-light RAG pipeline.

    Each stage is a separate method, so any component (embedder, index,
    generator) can be swapped without touching the rest.
    """

    def __init__(
        self,
        corpus_name: str = "m-ric/huggingface_doc",
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k: int = 4,
    ) -> None:
        self.corpus_name = corpus_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.embedder = SentenceTransformer(embed_model)
        self.chunks: List[Chunk] = []
        self.index: Optional[faiss.Index] = None

    # ---- Ingestion & chunking ------------------------------------------------

    def load_and_chunk(self, max_docs: Optional[int] = None) -> List[Chunk]:
        """Load the corpus and split every document into overlapping chunks."""
        ds = load_dataset(self.corpus_name, split="train")
        if max_docs is not None:
            ds = ds.select(range(min(max_docs, len(ds))))

        chunks: List[Chunk] = []
        for doc_id, row in enumerate(ds):
            for text in self._chunk(row["text"]):
                chunks.append(Chunk(text=text, source=row["source"], doc_id=doc_id))
        self.chunks = chunks
        return chunks

    def _chunk(self, text: str) -> List[str]:
        """Recursive character chunking with overlap."""
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks

    # ---- Embedding & indexing ------------------------------------------------

    def build_index(self) -> None:
        """Embed all chunks and build a FAISS index (cosine via inner product)."""
        if not self.chunks:
            raise ValueError("No chunks yet - call load_and_chunk() first.")

        texts = [c.text for c in self.chunks]
        embeddings = self.embedder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product == cosine on unit vectors
        self.index.add(embeddings.astype("float32"))

    # ---- Retrieval -----------------------------------------------------------

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        """Return the top-k most relevant chunks for a query."""
        if self.index is None:
            raise ValueError("No index yet - call build_index() first.")

        k = top_k or self.top_k
        q_emb = self.embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, indices = self.index.search(q_emb, k)

        results: List[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS pads with -1 when fewer than k results
                continue
            chunk = self.chunks[int(idx)]
            results.append(
                RetrievedChunk(text=chunk.text, source=chunk.source, score=float(score))
            )
        return results

    # ---- Generation ----------------------------------------------------------

    def build_prompt(self, query: str, retrieved: List[RetrievedChunk]) -> str:
        """Assemble a grounded prompt with numbered context for citation."""
        context = "\n\n".join(f"[{i + 1}] {c.text}" for i, c in enumerate(retrieved))
        return (
            "Answer the question using ONLY the context below. "
            "Cite the source numbers you use, e.g. [1]. "
            "If the context does not contain the answer, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )

    def generate(
        self,
        query: str,
        retrieved: List[RetrievedChunk],
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        max_new_tokens: int = 256,
    ) -> str:
        """Generate an answer with a local model (runs on CPU, no API needed)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        prompt = self.build_prompt(query, retrieved)
        messages = [
            {"role": "system", "content": "You are a helpful, grounded assistant."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        answer = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return answer.strip()

    # ---- End-to-end ----------------------------------------------------------

    def answer(self, query: str, **gen_kwargs) -> tuple:
        """Full RAG: retrieve then generate. Returns (answer, retrieved_chunks)."""
        retrieved = self.retrieve(query)
        answer = self.generate(query, retrieved, **gen_kwargs)
        return answer, retrieved
