# Basic RAG Learning

A small terminal-based project built to understand the core concepts behind Retrieval-Augmented Generation (RAG) step by step.

This project focuses on learning the fundamentals directly in Python without using LangChain or a frontend.

## What I Practiced

- Text embeddings with Sentence Transformers
- Cosine similarity
- FAISS vector search
- PDF text extraction with PyMuPDF
- Character-based chunking with overlap
- Query embedding and top-k retrieval
- Building context from retrieved chunks
- Generating grounded answers with Groq

## Learning Flow

```text
Text
↓
Embeddings
↓
Cosine Similarity
↓
FAISS
↓
PDF Extraction
↓
Chunking
↓
Chunk Embeddings
↓
Top-k Retrieval
↓
Context
↓
Groq LLM
↓
Final Answer
```

## Project Structure

```text
basic-rag-learning/
├── data/
│   └── notes.pdf
├── phase01_embeddings.py
├── phase02_faiss.py
├── phase03_pdf_chunking.py
├── phase04_pdf_retrieval.py
├── phase05_basic_rag.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Phase Overview

### Phase 1 — Embeddings

Converted sample sentences into 384-dimensional vectors using `all-MiniLM-L6-v2` and compared their semantic similarity using cosine similarity.

### Phase 2 — FAISS

Stored sentence embeddings in FAISS and retrieved the most semantically similar sentences for a query.

### Phase 3 — PDF Extraction and Chunking

Extracted text from a PDF using PyMuPDF and split the content into overlapping chunks.

### Phase 4 — PDF Retrieval

Generated embeddings for PDF chunks, stored them in FAISS, and retrieved the top matching chunks for user questions.

### Phase 5 — Basic RAG

Combined FAISS retrieval with Groq. The retrieved chunks are used as context so the LLM can answer questions based on the PDF.

## Setup

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Basic_RAG_Mini_Project.git
cd Basic_RAG_Mini_Project
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Git Bash:

```bash
source .venv/Scripts/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Run the final RAG script:

```bash
python phase05_basic_rag.py
```

## Example

```text
Ask a question about the PDF: What is bias in machine learning?

Answer:
Bias is the error between the average model prediction and the ground truth, reflecting the capacity of the underlying model to predict the values.
```

For a question not covered by the document:

```text
Ask a question about the PDF: What is RAG?

Answer:
I do not know based on the document.
```

## Why I Built This

I had already built a full-stack RAG project called DOCUAI. I created this smaller terminal-based version to understand the core RAG pipeline more deeply and practice embeddings, FAISS, retrieval, chunking, context construction, and LLM generation without relying on higher-level frameworks.

The goal of this project is learning and hands-on practice rather than building another large application.

## Tech Stack

- Python
- Sentence Transformers
- FAISS
- PyMuPDF
- NumPy
- Groq API
- python-dotenv

## Key Takeaway

FAISS retrieves the most relevant chunks, while the LLM uses those retrieved chunks as context to generate the final answer.

That separation between retrieval and generation is the core idea behind RAG.
