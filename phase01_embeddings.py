from sentence_transformers import SentenceTransformer

# load the embedding model

model = SentenceTransformer("all-MiniLM-L6-v2")
sentences = [
    "A dog is running in the park.",
    "A puppy is playing outside.",
    "Python is a programming language.",
    "I ordered a cheese pizza."
]

# Converts sentences to embeddings

embeddings = model.encode(sentences)

print("Number of sentences: ", len(sentences))
print("Embdedding shape: ", embeddings.shape)

print("\nFirst snetnence:")
print(sentences[0])

print("\nFirst 10 embedding values:")
print(embeddings[0][:10])

from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity(embeddings)

print("\nSimilarity matrix:")
print(similarities)

query = "I want something to eat."

query_embedding = model.encode([query])

query_scores = cosine_similarity(query_embedding, embeddings)[0]

print("\nQuery:")
print(query)

print("\nSimilarity scores:")

for sententce, score in zip(sentences, query_scores):
    print(f"{score:.4f} -> {sententce}")
