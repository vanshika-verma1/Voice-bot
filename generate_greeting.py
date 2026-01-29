"""
Generate greeting audio using Deepgram TTS
Run this once to create the static audio file
"""
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GREETING_TEXT = "Hello, I am Aura, how can I help you today?"
OUTPUT_FILE = "static/greeting.wav"


def generate_greeting():
    # Create static folder if not exists
    os.makedirs("static", exist_ok=True)
    
    url = "https://api.deepgram.com/v1/speak"
    params = {
        "model": "aura-2-andromeda-en",
        "encoding": "mulaw",
        "sample_rate": "8000"
    }
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {"text": GREETING_TEXT}
    
    print(f"Generating greeting audio with Deepgram...")
    print(f"Text: {GREETING_TEXT}")
    
    with httpx.Client() as client:
        resp = client.post(url, params=params, headers=headers, json=body)
        
        if resp.status_code == 200:
            with open(OUTPUT_FILE, "wb") as f:
                f.write(resp.content)
            print(f"✅ Saved to {OUTPUT_FILE}")
            print(f"   Size: {len(resp.content)} bytes")
        else:
            print(f"❌ Error: {resp.status_code}")
            print(resp.text)


if __name__ == "__main__":
    generate_greeting()
