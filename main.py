import os
import uuid
import sys
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from livekit import api
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.db import connect_db, close_db

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

@asynccontextmanager
async def lifespan(app: FastAPI):
    connect_db()
    yield
    close_db()

app = FastAPI(lifespan=lifespan)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/getToken")
async def get_token():
    room_name = f"chat-{uuid.uuid4()}"
    participant_identity = f"user-{uuid.uuid4().hex[:8]}"

    token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    token.with_identity(participant_identity)
    token.with_name("User")
    token.with_grants(api.VideoGrants(
        room_join=True,
        room=room_name,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True
    ))

    return {
        "token": token.to_jwt(),
        "url": LIVEKIT_URL,
        "room_name": room_name
    }

STATIC_DIR = os.path.join(BASE_DIR, "static")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    print(f"🚀 Starting Server at http://localhost:8000")
    print(f"📁 Serving static files from: {STATIC_DIR}")
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
    except Exception as e:
        print(f"Uvicorn failed to start: {e}")