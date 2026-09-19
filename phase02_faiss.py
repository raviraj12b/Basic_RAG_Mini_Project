from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

#  load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = ["A dog is running in the park.",
             "A puppy is playing outside.",
             "Python is a programming language.",
             "I ordered a cheese pizza."]

# Converts sentences to embeddings 
embeddings = model.encode(sentences)


# convert embeddings to folat32

embeddings = np.array(embeddings , dtype="float32")

# normalize embeddings 
faiss.normalize_L2(embeddings)

# Get embeddings dimension
dimension = embeddings.shape[1]

print("Embeddings dimensions: ", dimension)

# Create a FAISS index
index = faiss.IndexFlatIP(dimension)

# Add embeddings into FAISS 
index.add(embeddings)

print("vectors stored in Faiss:", index.ntotal)

query = "I am hungry."

query_embedding = model.encode([query])

query_embedding =np.array(query_embedding, dtype= "float32")

faiss.normalize_L2(query_embedding)

scores, indice = index.search(query_embedding, k=2)

print("\nQuery:")
print(query)

print("\nTop matches:")

for score,idx in zip(scores[0], indice[0]):
    print(f"{score:.4f} -> {sentences[idx]}")
