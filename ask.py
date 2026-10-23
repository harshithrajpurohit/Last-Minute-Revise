import os

from groq import Groq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# 1. Settings
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION = "sql_lectures"
DB_FOLDER = "qdrant_data"
TOP_K = 5                            # how many chunks to fetch
LLM_MODEL = "openai/gpt-oss-120b"

# 2. Check that the key is set
if not os.environ.get("GROQ_API_KEY"):
    raise SystemExit("GROQ_API_KEY is not set. Run the $env:GROQ_API_KEY=... command first.")

# 3. Load everything once
print("Loading models ...")
embedder = SentenceTransformer(MODEL_NAME)
qdrant = QdrantClient(path=DB_FOLDER)
llm = Groq()                         # reads GROQ_API_KEY from the terminal


def search(question):
    """Find the TOP_K chunks closest in meaning to the question."""
    query_vector = embedder.encode(question).tolist()
    results = qdrant.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=TOP_K,
    ).points
    return [r.payload for r in results]


def make_link(chunk):
    return f"https://www.youtube.com/watch?v={chunk['video_id']}&t={chunk['start']}s"


def answer(question):
    chunks = search(question)

    # Put the chunks into numbered "notes" for the LLM
    notes = ""
    for i, c in enumerate(chunks, start=1):
        notes += f"[{i}] Video: {c['title']} (starts at {c['start']}s)\n{c['text']}\n\n"

    system_prompt = (
        "You are a helpful SQL teaching assistant. "
        "Answer the student's question using ONLY the lecture notes given below. "
        "The notes are auto-generated transcripts, so spellings may be wrong "
        "(for example 'gait's measure' means 'Gate Smashers'). "
        "Answer in simple Hinglish (Hindi written in English letters), like explaining to a beginner. "
        "After each point, mention the note number like [1] or [2]. "
        "If the notes do not contain the answer, say clearly that the lectures do not cover it."
    )

    user_message = f"Lecture notes:\n\n{notes}\nQuestion: {question}"

    response = llm.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=800,
        temperature=0.3,             # low = stays close to the notes, less creative
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content, chunks


# 4. Ask questions in a loop
print("\nReady! Type your question (or 'exit' to quit).\n")
while True:
    question = input("You: ").strip()
    if question.lower() in ("exit", "quit", ""):
        break

    text, chunks = answer(question)

    print("\nAnswer:\n")
    print(text)
    print("\nWatch here:")
    for i, c in enumerate(chunks, start=1):
        print(f"  [{i}] {c['title'][:50]} @ {c['start']}s -> {make_link(c)}")
    print()