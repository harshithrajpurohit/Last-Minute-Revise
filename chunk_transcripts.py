import json
import os

# 1. Settings
TRANSCRIPT_DIR = "transcripts"
CHUNK_SECONDS = 75            # target length of one chunk

all_chunks = []

# 2. Go through every transcript file
for file_name in sorted(os.listdir(TRANSCRIPT_DIR)):
    if not file_name.endswith(".json"):
        continue

    with open(f"{TRANSCRIPT_DIR}/{file_name}", "r", encoding="utf-8") as f:
        transcript = json.load(f)

    video_id = transcript["video_id"]
    title = transcript["title"]
    segments = transcript["segments"]

    if not segments:
        continue

    # 3. Group small segments into ~75 second chunks
    current_texts = []
    chunk_start = segments[0]["start"]
    chunk_end = segments[0]["end"]

    for seg in segments:
        # if this chunk is already long enough, close it and start a new one
        if current_texts and (seg["start"] - chunk_start) >= CHUNK_SECONDS:
            all_chunks.append(
                {
                    "video_id": video_id,
                    "title": title,
                    "start": int(chunk_start),
                    "end": int(chunk_end),
                    "text": " ".join(current_texts),
                }
            )
            current_texts = []
            chunk_start = seg["start"]

        current_texts.append(seg["text"])
        chunk_end = seg["end"]

    # 4. Do not forget the last leftover chunk
    if current_texts:
        all_chunks.append(
            {
                "video_id": video_id,
                "title": title,
                "start": int(chunk_start),
                "end": int(chunk_end),
                "text": " ".join(current_texts),
            }
        )

# 5. Give each chunk a number and save everything
for i, chunk in enumerate(all_chunks):
    chunk["chunk_id"] = i

with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)

print(f"Done! Created {len(all_chunks)} chunks.\n")

# 6. Show 2 samples so we can inspect them
for chunk in all_chunks[:2]:
    print(f"Video : {chunk['title']}")
    print(f"Time  : {chunk['start']}s to {chunk['end']}s")
    print(f"Text  : {chunk['text'][:300]}")
    print("-" * 50)