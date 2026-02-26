import whisper
import os

AUDIO_PATH = r"E:\BaconBuddy\campfire\audio"
TRANSCRIPT_PATH = r"E:\BaconBuddy\campfire\transcripts"

os.makedirs(TRANSCRIPT_PATH, exist_ok=True)

print("Loading Whisper model...")
model = whisper.load_model("base", device="cuda")

for filename in os.listdir(AUDIO_PATH):
    if filename.endswith(".mp3"):
        transcript_file = os.path.join(TRANSCRIPT_PATH, filename.replace(".mp3", ".txt"))
        
        # Skip if already transcribed
        if os.path.exists(transcript_file):
            print(f"Skipping (already done): {filename}")
            continue
        
        audio_file = os.path.join(AUDIO_PATH, filename)
        print(f"Transcribing: {filename}")
        result = model.transcribe(audio_file, verbose=True)
        
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write(result["text"])
        
        print(f"Saved: {transcript_file}\n")

print("All done!")