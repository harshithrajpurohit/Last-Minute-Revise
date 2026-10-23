import json

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# 1. Settings
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION = "sql_lectures"
DB_FOLDER = "qdrant_data"

# 2. Load the chunks we made in Step 4
with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
print(f"Loaded {len(chunks)} chunks")

# 3. Load the embedding model (first time it downloads, so wait a bit)
print("Loading embedding model ...")
model = SentenceTransformer(MODEL_NAME)
vector_size = model.get_sentence_embedding_dimension()   # 384 for this model

# 4. Turn every chunk's text into numbers
texts = [c["text"] for c in chunks]
vectors = model.encode(texts, show_progress_bar=True)

# 5. Open Qdrant (local folder mode)
client = QdrantClient(path=DB_FOLDER)

# start fresh every time we run this file
if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)

client.create_collection(
    collection_name=COLLECTION,
    vectors_config=models.VectorParams(
        size=vector_size,
        distance=models.Distance.COSINE,   # how we measure "closeness"
    ),
)

# 6. Upload: number list + label (video id, title, time, text)
points = []
for chunk, vector in zip(chunks, vectors):
    points.append(
        models.PointStruct(
            id=chunk["chunk_id"],
            vector=vector.tolist(),
            payload=chunk,          # the whole chunk is stored as the label
        )
    )

client.upsert(collection_name=COLLECTION, points=points)
print(f"Uploaded {len(points)} chunks to Qdrant!\n")

# 7. Quick test search
query = "SQL kya hota hai"
query_vector = model.encode(query).tolist()

results = client.query_points(
    collection_name=COLLECTION,
    query=query_vector,
    limit=3,
).points

print(f"Test question: {query}\n")
for r in results:
    p = r.payload
    print(f"Score: {r.score:.2f}")
    print(f"Video: {p['title']}")
    print(f"Time : {p['start']}s")
    print(f"Link : https://www.youtube.com/watch?v={p['video_id']}&t={p['start']}s")
    print(f"Text : {p['text'][:150]}")
    print("-" * 50)