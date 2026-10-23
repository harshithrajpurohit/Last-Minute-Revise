import os
import threading

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from groq import Groq
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# 1. Settings (same as ask.py)
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION = "sql_lectures"
DB_FOLDER = "qdrant_data"
TOP_K = 5
LLM_MODEL = "openai/gpt-oss-120b"

if not os.environ.get("GROQ_API_KEY"):
    raise SystemExit("GROQ_API_KEY is not set. Run the $env:GROQ_API_KEY=... command first.")

# 2. Load everything once when the server starts
print("Loading models ...")
embedder = SentenceTransformer(MODEL_NAME)
qdrant = QdrantClient(path=DB_FOLDER)
llm = Groq()
db_lock = threading.Lock()      # one search at a time (local Qdrant is not multi-user)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


class Question(BaseModel):
    question: str


def search(question):
    query_vector = embedder.encode(question).tolist()
    with db_lock:
        results = qdrant.query_points(
            collection_name=COLLECTION,
            query=query_vector,
            limit=TOP_K,
        ).points
    return [r.payload for r in results]


@app.get("/")
def home():
    """Show the website."""
    return FileResponse("static/index.html")


@app.post("/ask")
def ask(body: Question):
    """Take a question, return an answer plus the video sources."""
    chunks = search(body.question)

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

    response = llm.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=800,
        temperature=0.3,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Lecture notes:\n\n{notes}\nQuestion: {body.question}"},
        ],
    )

    sources = [
        {
            "number": i,
            "title": c["title"],
            "video_id": c["video_id"],
            "start": c["start"],
            "text": c["text"][:200],
        }
        for i, c in enumerate(chunks, start=1)
    ]

    return {"answer": response.choices[0].message.content, "sources": sources}