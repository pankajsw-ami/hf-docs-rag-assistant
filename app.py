"""Gradio demo for the RAG assistant. Run locally with `python app.py`."""

import gradio as gr

from src.rag import RAGPipeline

# Build the pipeline once at startup (embeddings + FAISS index are cached).
PIPELINE = RAGPipeline()
PIPELINE.load_and_chunk(max_docs=500)
PIPELINE.build_index()


def ask(question: str) -> str:
    if not question.strip():
        return "Please enter a question."
    answer, retrieved = PIPELINE.answer(question)
    sources = "\n".join(f"- {r.source}" for r in retrieved)
    return f"{answer}\n\n**Sources:**\n{sources}"


demo = gr.Interface(
    fn=ask,
    inputs=gr.Textbox(
        label="Question", placeholder="e.g. How do I create an Inference Endpoint?"
    ),
    outputs=gr.Markdown(label="Answer"),
    title="HF Docs RAG Assistant",
    description=(
        "Ask questions about the Hugging Face documentation. "
        "Answers are grounded in retrieved docs with citations."
    ),
    examples=[
        "How do I create an Inference Endpoint?",
        "How do I choose an evaluation metric?",
        "What are the main features of Gradio?",
    ],
)

if __name__ == "__main__":
    demo.launch()
