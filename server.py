import os
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from livekit import api
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from db import connect_db, close_db

load_dotenv()

templates = Jinja2Templates(directory="templates")

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "secret")


@asynccontextmanager
async def lifespan(app: FastAPI):
    connect_db()
    yield
    close_db()


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def get_webpage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


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


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Server at http://localhost:8000")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
