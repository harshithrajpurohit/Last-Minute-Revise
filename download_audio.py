import json
import os

import yt_dlp

# 1. Settings
PLAYLIST_URL = "https://youtube.com/playlist?list=PLxCzCOWd7aiFbom4rYyl5qROgqYuCuAPD"
AUDIO_DIR = "audio"      # folder where audio files will be saved
MAX_VIDEOS = 3           # start small, increase later

# 2. Make the audio folder if it does not exist
os.makedirs(AUDIO_DIR, exist_ok=True)

# 3. Tell yt-dlp how to download
ydl_options = {
    "format": "bestaudio/best",                  # pick the best audio quality
    "outtmpl": f"{AUDIO_DIR}/%(id)s.%(ext)s",    # file name = video id
    "playlist_items": f"1-{MAX_VIDEOS}",         # only first N videos
    "ignoreerrors": True,                        # skip broken videos, do not crash
    "postprocessors": [
        {
            "key": "FFmpegExtractAudio",         # use ffmpeg to extract audio
            "preferredcodec": "mp3",
            "preferredquality": "64",            # 64 kbps is enough for speech
        }
    ],
}

# 4. Download and collect info about each video
with yt_dlp.YoutubeDL(ydl_options) as ydl:
    playlist_info = ydl.extract_info(PLAYLIST_URL, download=True)

videos = []
for entry in playlist_info["entries"]:
    if entry is None:   # this video failed, skip it
        continue
    videos.append(
        {
            "video_id": entry["id"],
            "title": entry["title"],
            "audio_path": f"{AUDIO_DIR}/{entry['id']}.mp3",
        }
    )

# 5. Save the list in a JSON file
with open("videos.json", "w", encoding="utf-8") as f:
    json.dump(videos, f, indent=2, ensure_ascii=False)

print(f"Done! Downloaded {len(videos)} videos.")
for v in videos:
    print("-", v["title"])