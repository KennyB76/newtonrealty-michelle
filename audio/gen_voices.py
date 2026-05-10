"""
Generate Michelle landing voice intros via ElevenLabs.

Setup:
  1. Copy .env.example -> .env
  2. Set ELEVENLABS_API_KEY in .env (get from https://elevenlabs.io/app/settings/api-keys)
  3. pip install elevenlabs python-dotenv
  4. python gen_voices.py
"""
import os
from elevenlabs import ElevenLabs
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("ELEVENLABS_API_KEY")
if not api_key:
    raise RuntimeError(
        "ELEVENLABS_API_KEY not set. Copy .env.example to .env and add your key."
    )

client = ElevenLabs(api_key=api_key)

# Sarah — Mature/Reassuring/Confident premade, multilingual_v2 (KO+EN)
VOICE_ID = "EXAVITQu4vr4xnSDxMaL"
MODEL = "eleven_multilingual_v2"

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

scripts = [
    ("intro-ko.mp3", "당신의 결을 압니다. 미셸의 손에서 자란, 24년의 결."),
    ("intro-en.mp3", "Your hair deserves someone who truly knows it. She knows your hair."),
    ("cta.mp3", "예약은 카카보카에서. Book your seat at Kakaboka."),
]

for filename, text in scripts:
    path = os.path.join(OUTPUT_DIR, filename)
    print(f"Generating {filename} ...", flush=True)
    audio = client.text_to_speech.convert(
        text=text,
        voice_id=VOICE_ID,
        model_id=MODEL,
        voice_settings={
            "stability": 0.65,
            "similarity_boost": 0.80,
            "style": 0.20,
            "use_speaker_boost": True,
        },
    )
    with open(path, "wb") as f:
        for chunk in audio:
            f.write(chunk)
    size = os.path.getsize(path)
    print(f"  OK  {path}  ({size:,} bytes)", flush=True)

print("ALL DONE")
