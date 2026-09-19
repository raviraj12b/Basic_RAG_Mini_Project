import pymupdf
import numpy as np 
import faiss

from sentence_transformers import SentenceTransformer

# Extract PDF text

pdf_path = "data/ML Cheatsheet .pdf"

document =pymupdf.open(pdf_path)

all_text =""
for page in document:

    all_text += page.get_text("text") + "\n" # type: ignore

print("Total extracted characters:", len(all_text))

# Chunk the text into smaller pieces

chunk_size = 500

overlap = 100

chunks =[]

start = 0

while start < len(all_text):

    end =start + chunk_size

    chunk = all_text[start:end]

    chunks.append(chunk)

    start = end - overlap

print("Total chunks:", len(chunks))

# Convert all chunks into embeddings

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

print("Embedding shape:", embeddings.shape)


# Prepare vectors for FAISS

embeddings = np.array(embeddings, dtype="float32")

faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]

print("Embeddings dimension:", dimension)

# Create FAISS index

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print("Vectors stored in Faiss:", index.ntotal)

# Ask a real question

query = "What is bias in machine learning?"

query_embedding = model.encode([query])

query_embedding = np.array(query_embedding, dtype ="float32")

faiss.normalize_L2(query_embedding)

# Search FAISS

k = 3

scores, indices = index.search(
    query_embedding, k
)

print("\nQuestion:")
print(query)

print("\nTop retrieved chunks:")

for rank, (score, idx) in enumerate(
    zip(scores[0], indices[0]),
    start=1
):

    print(
        f"\nResult {rank}"
    )

    print(
        f"Score: {score:.4f}"
    )

    print(
        f"Chunk index: {idx}"
    )

    print(
        chunks[idx]
    )

    print("-" * 60)
    
