# YouTube Playlist RAG: Ask Questions, Watch the Exact Moment

Ask a question about a YouTube lecture playlist and get an answer in simple Hinglish,
along with a video player that jumps to the exact timestamp where the topic is explained.

Built and tested on the "SQL Full Course" playlist by Gate Smashers.

## How it works

1. **yt-dlp** downloads the audio of every video in a playlist.
2. **faster-whisper** converts audio to text with timestamps.
3. Text is cut into ~75 second **chunks**; each chunk keeps its video ID and start time.
4. A multilingual **sentence-transformers** model turns each chunk into a vector.
5. Vectors are stored in **Qdrant** (local mode).
6. For a question: embed it, fetch the 5 closest chunks, and ask an LLM
   (**Groq, Llama 3.3 70B**) to answer using only those chunks.
7. A **FastAPI** web app shows the answer and plays the video at the right second.

## Setup

```bash
git clone https://github.com/YOUR-USERNAME/youtube-playlist-rag.git
cd youtube-playlist-rag
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

You also need **ffmpeg** installed (`winget install ffmpeg` on Windows).

## Run

Run these in order:

```bash
python download_audio.py       # set PLAYLIST_URL and MAX_VIDEOS inside the file
python transcribe.py           # slow on CPU, run it once and wait
python chunk_transcripts.py
python build_index.py
```

Then start the app (PowerShell):

```powershell
$env:GROQ_API_KEY="your-key-here"
uvicorn app:app
```

Open http://127.0.0.1:8000 and ask a question.

Get a free API key at https://console.groq.com.

## Project structure

| File | Job |
|---|---|
| `download_audio.py` | playlist to mp3 files |
| `transcribe.py` | mp3 to timestamped text |
| `chunk_transcripts.py` | text to ~75s chunks |
| `build_index.py` | chunks to vectors in Qdrant |
| `ask.py` | ask questions in the terminal |
| `app.py` | FastAPI backend |
| `static/index.html` | web page with video player |

## Notes

- Lecture content belongs to its original creator. This repo contains only code,
  not the audio or transcripts.
- Transcripts are auto-generated, so some spellings are imperfect.
