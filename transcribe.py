import json
import os
import subprocess

import numpy as np
from faster_whisper import WhisperModel

# 1. Settings
MODEL_SIZE = "small"        # options: tiny, base, small, medium
TRANSCRIPT_DIR = "transcripts"

os.makedirs(TRANSCRIPT_DIR, exist_ok=True)


def load_audio(path):
    """Open the mp3 using ffmpeg and return it as a list of numbers (16 kHz, mono)."""
    command = [
        "ffmpeg",
        "-nostdin",
        "-i", path,
        "-f", "f32le",       # raw 32-bit float numbers
        "-ac", "1",          # mono (1 channel)
        "-ar", "16000",      # 16000 samples per second (what Whisper expects)
        "-loglevel", "error",
        "-",                 # send output to Python instead of a file
    ]
    result = subprocess.run(command, capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype=np.float32)


# 2. Load the model (first time it downloads, so wait a bit)
print(f"Loading Whisper model: {MODEL_SIZE} ...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

# 3. Read the register we made in Step 2
with open("videos.json", "r", encoding="utf-8") as f:
    videos = json.load(f)

videos = videos[:3]   # test on first 3 only; remove this line later

# 4. Transcribe each video
for video in videos:
    out_path = f"{TRANSCRIPT_DIR}/{video['video_id']}.json"

    # skip if already done (so we do not repeat work)
    if os.path.exists(out_path):
        print(f"Skipping (already done): {video['title']}")
        continue

    print(f"\nTranscribing: {video['title']}")

    audio = load_audio(video["audio_path"])

    segments, info = model.transcribe(
        audio,
        language="hi",                      # lecture is Hindi + English mix
        initial_prompt="Aaj hum SQL, DBMS, table, query, database ke baare mein padhenge.",
        vad_filter=True,                    # skip silent parts
    )

    segment_list = []
    for seg in segments:
        segment_list.append(
            {
                "start": round(seg.start, 2),   # seconds
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
            }
        )
        print(f"  [{seg.start:7.1f}s] {seg.text.strip()}")

    # 5. Save the result
    result = {
        "video_id": video["video_id"],
        "title": video["title"],
        "segments": segment_list,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

print("\nDone! Transcripts saved in the 'transcripts' folder.")