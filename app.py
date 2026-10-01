"""Gradio demo for the RAG assistant. Run locally with `python app.py`."""

import gradio as gr
import faiss

from src.rag import RAGPipeline


# Build the pipeline once at startup.
# This keeps the original Hugging Face documentation mode unchanged.
PIPELINE = RAGPipeline()
PIPELINE.load_and_chunk(max_docs=500)
PIPELINE.build_index()


def build_uploaded_index(uploaded_chunks):
    """Build a temporary FAISS index for the uploaded document.

    This intentionally does not use the normal embeddings cache because
    uploaded documents have different content from the Hugging Face corpus.
    """

    if not uploaded_chunks:
        raise ValueError("The uploaded file does not contain readable text.")

    PIPELINE.chunks = uploaded_chunks

    texts = [chunk.text for chunk in uploaded_chunks]

    embeddings = PIPELINE.embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    dim = embeddings.shape[1]

    PIPELINE.index = faiss.IndexFlatIP(dim)
    PIPELINE.index.add(embeddings.astype("float32"))


def ask(file_path, question: str) -> str:
    """Answer using the uploaded document if provided, otherwise HF docs."""

    if not question or not question.strip():
        return "Please enter a question."

    # If a file is uploaded, switch retrieval to that document.
    if file_path:
        try:
            uploaded_chunks = PIPELINE.load_uploaded_file(file_path)
            build_uploaded_index(uploaded_chunks)
        except Exception as e:
            return f"Error processing uploaded file: {e}"

    try:
        answer, retrieved = PIPELINE.answer(question)

        sources = "\n".join(
            f"- {r.source}"
            for r in retrieved
        )

        return f"{answer}\n\n**Sources:**\n{sources}"

    except Exception as e:
        return f"Error generating answer: {e}"


demo = gr.Interface(
    fn=ask,
    inputs=[
        gr.File(
            label="Upload PDF, DOCX or TXT",
            file_types=[".pdf", ".docx", ".txt"],
            type="filepath",
        ),
        gr.Textbox(
            label="Question",
            placeholder="e.g. How do I create an Inference Endpoint?",
        ),
    ],
    outputs=gr.Markdown(label="Answer"),
    title="HF Docs RAG Assistant",
    description=(
        "Ask questions about the Hugging Face documentation, "
        "or upload a PDF, DOCX, or TXT file and ask questions about it."
    ),
    examples=[
        [None, "How do I create an Inference Endpoint?"],
        [None, "How do I choose an evaluation metric?"],
        [None, "What are the main features of Gradio?"],
    ],
)


if __name__ == "__main__":
    demo.launch()