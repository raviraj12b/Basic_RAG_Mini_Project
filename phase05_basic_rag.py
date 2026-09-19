import os

import faiss
import numpy as np
import pymupdf

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

client = Groq(
    api_key=api_key
)


# --------------------------------------------------
# 2. Read PDF
# --------------------------------------------------

pdf_path = "data/ML Cheatsheet .pdf"

document = pymupdf.open(pdf_path)

all_text = ""

for page in document:
    all_text += page.get_text() + "\n" # type: ignore


print("Total extracted characters:", len(all_text))


# --------------------------------------------------
# 3. Split text into chunks
# --------------------------------------------------

chunk_size = 500
overlap = 100

chunks = []

start = 0

while start < len(all_text):

    end = start + chunk_size

    chunk = all_text[start:end]

    chunks.append(chunk)

    start = end - overlap


print("Total chunks:", len(chunks))


# --------------------------------------------------
# 4. Create chunk embeddings
# --------------------------------------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embeddings = model.encode(chunks)

embeddings = np.array(
    embeddings,
    dtype="float32"
)

faiss.normalize_L2(embeddings)


# --------------------------------------------------
# 5. Store vectors in FAISS
# --------------------------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(
    dimension
)

index.add(embeddings)

print(
    "Vectors stored in FAISS:",
    index.ntotal
)


# --------------------------------------------------
# 6. Ask user question
# --------------------------------------------------

question = input(
    "\nAsk a question about the PDF: "
)


# --------------------------------------------------
# 7. Convert question to embedding
# --------------------------------------------------

query_embedding = model.encode(
    [question]
)

query_embedding = np.array(
    query_embedding,
    dtype="float32"
)

faiss.normalize_L2(
    query_embedding
)


# --------------------------------------------------
# 8. Retrieve relevant chunks
# --------------------------------------------------

k = 3

scores, indices = index.search(
    query_embedding,
    k
)


retrieved_chunks = []

for idx in indices[0]:
    retrieved_chunks.append(
        chunks[idx]
    )


# --------------------------------------------------
# 9. Build context
# --------------------------------------------------

context = "\n\n".join(
    retrieved_chunks
)


# --------------------------------------------------
# 10. Build prompt
# --------------------------------------------------

prompt = f"""
Answer the question using only the context below.

If the answer is not present in the context,
say that you do not know based on the document.

Context:
{context}

Question:
{question}

Answer:
"""


# --------------------------------------------------
# 11. Send prompt to Groq
# --------------------------------------------------

response = client.chat.completions.create(

    model="openai/gpt-oss-120b",

    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


# --------------------------------------------------
# 12. Print final answer
# --------------------------------------------------

answer = response.choices[0].message.content

print("\nAnswer:")
print(answer)