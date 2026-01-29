import os
import json
import base64
import asyncio
import audioop
import time
import re
from datetime import datetime
from typing import Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from twilio.rest import Client as TwilioClient
from twilio.twiml.voice_response import VoiceResponse, Connect
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions, DeepgramClientOptions
from pysilero_vad import SileroVoiceActivityDetector

from dotenv import load_dotenv
load_dotenv()

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
SERVER_URL = os.getenv("SERVER_URL", "")

app = FastAPI(title="Voice Agent Demo")
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET", "demo-secret-key"))

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
dg_client = DeepgramClient(DEEPGRAM_API_KEY, config=DeepgramClientOptions(options={"keepalive": "true"}))

from db import connect_db, get_db
connect_db()

active_calls: Dict[str, dict] = {}
transcript_connections: Dict[str, WebSocket] = {}


def save_message(call_sid: str, role: str, content: str, phone_number: str = None):
    """Save a chat message to MongoDB"""
    db = get_db()
    if db is None:
        logger.warning("Database not connected, skipping message save")
        return
    try:
        update_doc = {
            "$push": {
                "messages": {
                    "role": role,
                    "content": content,
                    "timestamp": datetime.utcnow()
                }
            },
            "$setOnInsert": {
                "call_sid": call_sid,
                "created_at": datetime.utcnow()
            },
            "$set": {
                "updated_at": datetime.utcnow()
            }
        }
        if phone_number:
            update_doc["$setOnInsert"]["phone_number"] = phone_number
        
        db.phone_call_transcript.update_one(
            {"call_sid": call_sid},
            update_doc,
            upsert=True
        )
    except Exception as e:
        logger.error(f"Failed to save message: {e}")


class TTSManager:
    def __init__(self, callsid: str, voice_id: str = "aura-2-andromeda-en"):
        self.callsid = callsid
        self.voice_id = voice_id
        self.api_key = os.getenv("DEEPGRAM_API_KEY")
        self._client = None
        self._connection = None
        self._main_loop = None
        self._current_ws = None
        self._current_stream_sid = None
        self._stop_event = None
        self._utterance_done = None
        self._last_audio_ts = None
        self._frame_bytes = 160
        self._bytes_per_second = 8000
        self._silence = b'\xFF' * self._frame_bytes
        self._send_queue = None
        self._out_buffer = b""
        self._sender_task = None

    async def setup(self):
        from deepgram import SpeakWebSocketEvents, SpeakWSOptions
        config = DeepgramClientOptions(options={"keepalive": "true"})
        self._client = DeepgramClient(self.api_key, config=config)
        self._connection = self._client.speak.websocket.v("1")
        self._main_loop = asyncio.get_running_loop()
        self._send_queue = asyncio.Queue(maxsize=400)

        def on_binary_data(dg_self, data: bytes, *args, **kwargs):
            if self._stop_event and self._stop_event.is_set():
                if self._utterance_done and not self._utterance_done.is_set():
                    if self._main_loop and not self._main_loop.is_closed():
                        self._main_loop.call_soon_threadsafe(self._utterance_done.set)
                return
            if not self._main_loop or self._main_loop.is_closed() or not self._send_queue:
                return
            self._last_audio_ts = time.monotonic()
            asyncio.run_coroutine_threadsafe(self._send_queue.put(data), self._main_loop)

        def on_close(dg_self, *args, **kwargs):
            if self._utterance_done and not self._utterance_done.is_set():
                if self._main_loop and not self._main_loop.is_closed():
                    self._main_loop.call_soon_threadsafe(self._utterance_done.set)

        def on_error(dg_self, error, *args, **kwargs):
            logger.error(f"TTS Error: {error}")
            if self._utterance_done and not self._utterance_done.is_set():
                if self._main_loop and not self._main_loop.is_closed():
                    self._main_loop.call_soon_threadsafe(self._utterance_done.set)

        self._connection.on(SpeakWebSocketEvents.AudioData, on_binary_data)
        self._connection.on(SpeakWebSocketEvents.Close, on_close)
        self._connection.on(SpeakWebSocketEvents.Error, on_error)

        options = SpeakWSOptions(model=self.voice_id, encoding="mulaw", sample_rate=8000)
        started = self._connection.start(options)
        if not started:
            raise RuntimeError("Failed to start TTS connection")
        self._sender_task = asyncio.create_task(self._paced_sender())

    async def _paced_sender(self):
        frame_interval = self._frame_bytes / self._bytes_per_second
        next_due = time.monotonic()
        try:
            while True:
                if self._send_queue:
                    try:
                        while True:
                            piece = self._send_queue.get_nowait()
                            self._out_buffer += piece
                    except asyncio.QueueEmpty:
                        pass
                now = time.monotonic()
                if next_due > now:
                    await asyncio.sleep(next_due - now)
                if self._stop_event and self._stop_event.is_set():
                    self._out_buffer = b""
                    next_due = time.monotonic() + frame_interval
                    continue
                if len(self._out_buffer) >= self._frame_bytes:
                    chunk = self._out_buffer[:self._frame_bytes]
                    self._out_buffer = self._out_buffer[self._frame_bytes:]
                else:
                    chunk = self._silence
                try:
                    if self._current_ws and self._current_stream_sid:
                        payload = base64.b64encode(chunk).decode("utf-8")
                        await self._current_ws.send_json({
                            "event": "media",
                            "streamSid": self._current_stream_sid,
                            "media": {"payload": payload}
                        })
                except Exception as e:
                    logger.error(f"TTS send error: {e}")
                next_due += frame_interval
        except asyncio.CancelledError:
            return

    async def stream_tts_audio(self, ws, stream_sid: str, text: str, stop_event: asyncio.Event) -> bool:
        if not text or stop_event.is_set():
            return True
        self._current_ws = ws
        self._current_stream_sid = stream_sid
        self._stop_event = stop_event
        self._utterance_done = asyncio.Event()
        self._last_audio_ts = None

        try:
            await self._send_queue.put(self._silence)
            self._connection.send_text(text)
            self._connection.flush()
            idle_seconds = 0.1
            while True:
                if self._stop_event and self._stop_event.is_set():
                    break
                if self._utterance_done and self._utterance_done.is_set():
                    break
                now = time.monotonic()
                if self._last_audio_ts is not None and (now - self._last_audio_ts) >= idle_seconds:
                    await self._wait_drain(timeout=5.0)
                    break
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"TTS stream error: {e}")
        finally:
            if self._utterance_done and not self._utterance_done.is_set():
                self._utterance_done.set()
        return True

    async def _wait_drain(self, timeout: float = 2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            queue_empty = self._send_queue.empty() if self._send_queue else True
            if queue_empty and len(self._out_buffer) < self._frame_bytes:
                return
            await asyncio.sleep(0.01)

    async def clear_buffer(self):
        self._out_buffer = b""
        if self._send_queue:
            try:
                while not self._send_queue.empty():
                    self._send_queue.get_nowait()
            except:
                pass
        if self._connection:
            try:
                self._connection.clear()
            except:
                pass

    async def cleanup(self):
        if self._sender_task:
            self._sender_task.cancel()
            try:
                await self._sender_task
            except asyncio.CancelledError:
                pass
        if self._connection:
            try:
                self._connection.finish()
            except:
                pass
        self._connection = None
        self._client = None


class SimpleAgent:
    def __init__(self):
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.history = []
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self):
        try:
            with open("system_prompt.txt", "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return "You are a helpful voice assistant. Keep responses short and conversational."

    async def get_response(self, user_text: str):
        self.history.append({"role": "user", "content": user_text})
        messages = [{"role": "system", "content": self.system_prompt}] + self.history[-20:]
        
        try:
            response = await self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=150,
                stream=True
            )
            full_response = ""
            async for chunk in response:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    full_response += delta
                    yield delta
            self.history.append({"role": "assistant", "content": full_response})
        except Exception as e:
            logger.error(f"Agent error: {e}")
            yield "Sorry, I had trouble processing that."


@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    if "user" in request.session:
        return templates.TemplateResponse("demo_call.html", {"request": request, "view": "call"})
    return templates.TemplateResponse("demo_call.html", {"request": request, "view": "login"})


DEMO_EMAIL = "demo@test.com"
DEMO_PASSWORD = "demo123"

@app.post("/api/login")
async def login(request: Request, email: str = Form(...), password: str = Form(...)):
    if email != DEMO_EMAIL or password != DEMO_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid credentials. Use demo@test.com / demo123")
    request.session["user"] = email
    return JSONResponse({"status": "success", "email": email})


@app.post("/api/logout")
async def logout(request: Request):
    request.session.clear()
    return JSONResponse({"status": "success"})


@app.get("/call", response_class=HTMLResponse)
async def call_page(request: Request):
    if "user" not in request.session:
        return templates.TemplateResponse("demo_call.html", {"request": request, "view": "login"})
    return templates.TemplateResponse("demo_call.html", {"request": request, "view": "call"})


@app.post("/api/make-call")
async def make_call(request: Request):
    data = await request.json()
    phone_number = data.get("phone_number", "")
    
    if not phone_number:
        raise HTTPException(status_code=400, detail="Phone number required")
    
    if not phone_number.startswith("+"):
        phone_number = "+" + phone_number
    
    base_url = SERVER_URL.replace("https://", "").replace("http://", "")
    
    try:
        call = twilio_client.calls.create(
            to=phone_number,
            from_=TWILIO_PHONE_NUMBER,
            url=f"{SERVER_URL}/outbound-twiml",
            status_callback=f"{SERVER_URL}/call-status",
            status_callback_event=["initiated", "ringing", "answered", "completed"]
        )
        
        active_calls[call.sid] = {
            "phone_number": phone_number,
            "status": "initiated",
            "started_at": datetime.now().isoformat()
        }
        
        return JSONResponse({
            "status": "success",
            "call_sid": call.sid,
            "phone_number": phone_number
        })
    except Exception as e:
        logger.error(f"Call failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/hangup")
async def hangup(request: Request):
    data = await request.json()
    call_sid = data.get("call_sid", "")
    
    if call_sid and call_sid in active_calls:
        try:
            twilio_client.calls(call_sid).update(status="completed")
            return JSONResponse({"status": "success"})
        except Exception as e:
            logger.error(f"Hangup failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))
    
    return JSONResponse({"status": "not_found"})


@app.api_route("/outbound-twiml", methods=["GET", "POST"], response_class=Response)
async def outbound_twiml(request: Request):
    base_url = SERVER_URL.replace("https://", "").replace("http://", "")
    stream_url = f"wss://{base_url}/media-stream"
    # greeting_url = f"https://{base_url}/static/greeting.wav"
    
    response = VoiceResponse()
    # response.play(greeting_url)  # TwiML greeting - commented out, using Deepgram TTS instead
    response.pause(length=0.2)
    connect = Connect()
    connect.stream(url=stream_url, name="voice_stream")
    response.append(connect)
    
    return Response(content=str(response), media_type="application/xml")


@app.post("/call-status")
async def call_status(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid", "")
    status = form.get("CallStatus", "")
    
    logger.info(f"Call status update: {call_sid} -> {status}")
    
    if call_sid in active_calls:
        active_calls[call_sid]["status"] = status
    
    if call_sid in transcript_connections:
        try:
            await transcript_connections[call_sid].send_json({
                "type": "status",
                "status": status
            })
        except:
            pass
    
    if status in ["completed", "failed", "busy", "no-answer", "canceled"]:
        active_calls.pop(call_sid, None)
        db = get_db()
        if db is not None:
            try:
                db.phone_call_transcript.update_one(
                    {"call_sid": call_sid},
                    {"$set": {"status": status, "ended_at": datetime.utcnow()}}
                )
            except Exception as e:
                logger.error(f"Failed to update call status: {e}")
    
    return Response(status_code=200)


@app.websocket("/ws/transcript/{call_sid}")
async def transcript_ws(ws: WebSocket, call_sid: str):
    await ws.accept()
    transcript_connections[call_sid] = ws
    
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        transcript_connections.pop(call_sid, None)


async def broadcast_transcript(call_sid: str, msg_type: str, text: str):
    if call_sid in transcript_connections:
        try:
            await transcript_connections[call_sid].send_json({
                "type": msg_type,
                "text": text,
                "timestamp": datetime.now().isoformat()
            })
        except:
            pass


@app.websocket("/media-stream")
async def media_stream(ws: WebSocket):
    await ws.accept()
    
    stream_sid = None
    call_sid = None
    phone_number = None
    tts_manager = None
    agent = SimpleAgent()
    stop_tts_event = asyncio.Event()
    silero_vad = SileroVoiceActivityDetector()
    
    transcript_queue = asyncio.Queue()
    buffer_sentence = []
    last_spoken_time = None
    should_terminate = False
    
    vad_last_voice_ts = 0.0
    silero_buffer = bytearray()
    ratecv_state = None
    tts_already_cleared = {"value": False}
    
    STRONG_PAUSE = 0.5
    WEAK_PAUSE = 0.7
    NO_PUNCT_PAUSE = 1.0
    
    STRONG_PUNCT_RE = re.compile(r'[.!?]+$')
    WEAK_PUNCT_RE = re.compile(r'[,;:]$')
    
    dg_connection = dg_client.listen.asynclive.v("1")
    
    async def on_transcript(_, result, **kwargs):
        nonlocal last_spoken_time, buffer_sentence
        snippet = getattr(result.channel.alternatives[0], "transcript", "") or ""
        if not snippet:
            return
        
        if buffer_sentence and last_spoken_time:
            elapsed = time.time() - last_spoken_time
            text = " ".join(buffer_sentence).strip()
            
            if STRONG_PUNCT_RE.search(text):
                required_pause = STRONG_PAUSE
            elif WEAK_PUNCT_RE.search(text):
                required_pause = WEAK_PAUSE
            else:
                required_pause = NO_PUNCT_PAUSE
            
            if elapsed < required_pause:
                prev = " ".join(buffer_sentence).strip()
                merged = (prev + " " + snippet).strip()
                buffer_sentence = [merged]
            else:
                buffer_sentence = [snippet]
        else:
            buffer_sentence = [snippet]
        
        last_spoken_time = time.time()
    
    async def on_error(_, error, **kwargs):
        logger.error(f"Deepgram error: {error}")
    
    dg_connection.on(LiveTranscriptionEvents.Transcript, on_transcript)
    dg_connection.on(LiveTranscriptionEvents.Error, on_error)
    
    options = LiveOptions(
        model="nova-3",
        punctuate=True,
        language="en",
        encoding="mulaw",
        channels=1,
        smart_format=True,
        sample_rate=8000,
        vad_events=True
    )
    
    async def monitor_silence():
        nonlocal last_spoken_time, buffer_sentence, should_terminate, vad_last_voice_ts
        
        while not should_terminate:
            if last_spoken_time is not None and buffer_sentence:
                now = time.time()
                silence_duration = now - last_spoken_time
                
                text = " ".join(buffer_sentence).strip()
                if STRONG_PUNCT_RE.search(text):
                    required_pause = STRONG_PAUSE
                elif WEAK_PUNCT_RE.search(text):
                    required_pause = WEAK_PAUSE
                else:
                    required_pause = NO_PUNCT_PAUSE
                
                vad_quiet = (time.time() - vad_last_voice_ts) > 0.25
                
                if silence_duration >= required_pause and vad_quiet:
                    full_sentence = " ".join(buffer_sentence).strip()
                    full_sentence = re.sub(r"\s+", " ", full_sentence)
                    buffer_sentence.clear()
                    last_spoken_time = None
                    
                    if full_sentence:
                        await transcript_queue.put(full_sentence)
                        logger.info(f"FINALIZED: '{full_sentence}'")
            
            await asyncio.sleep(0.05)
    
    async def process_transcripts():
        nonlocal should_terminate
        
        while not should_terminate:
            try:
                user_text = await asyncio.wait_for(transcript_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            
            if call_sid:
                await broadcast_transcript(call_sid, "user", user_text)
                save_message(call_sid, "user", user_text, phone_number)
            
            logger.info(f"USER: {user_text}")
            
            stop_tts_event.clear()
            tts_already_cleared["value"] = False
            
            buffer = ""
            full_response = ""
            sentence_re = re.compile(r'([^.!?]*[.!?])')
            
            async for chunk in agent.get_response(user_text):
                if stop_tts_event.is_set():
                    break
                    
                buffer += chunk
                full_response += chunk
                
                while True:
                    match = sentence_re.search(buffer)
                    if not match:
                        break
                    
                    sentence = match.group(1).strip()
                    buffer = buffer[match.end():].lstrip()
                    
                    if sentence and tts_manager and stream_sid:
                        await tts_manager.stream_tts_audio(ws, stream_sid, sentence, stop_tts_event)
            
            if buffer.strip() and tts_manager and stream_sid and not stop_tts_event.is_set():
                await tts_manager.stream_tts_audio(ws, stream_sid, buffer.strip(), stop_tts_event)
            
            if call_sid and full_response:
                await broadcast_transcript(call_sid, "agent", full_response)
                save_message(call_sid, "agent", full_response, phone_number)
            
            logger.info(f"AGENT: {full_response}")
            
            goodbye_phrases = ["goodbye", "bye", "take care", "have a great day", "have a nice day"]
            if any(phrase in full_response.lower() for phrase in goodbye_phrases):
                should_terminate = True
                if call_sid:
                    await broadcast_transcript(call_sid, "status", "Call ended")
    
    try:
        await dg_connection.start(options)
        
        silence_task = asyncio.create_task(monitor_silence())
        process_task = asyncio.create_task(process_transcripts())
        
        while not should_terminate:
            try:
                message = await asyncio.wait_for(ws.receive_text(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            except WebSocketDisconnect:
                break
            
            try:
                data = json.loads(message)
            except:
                continue
            
            event = data.get("event", "")
            
            if event == "start":
                stream_sid = data.get("start", {}).get("streamSid")
                call_sid = data.get("start", {}).get("callSid")
                
                if call_sid:
                    if call_sid in active_calls:
                        phone_number = active_calls[call_sid].get("phone_number")
                    
                    tts_manager = TTSManager(call_sid)
                    await tts_manager.setup()
                    
                    await broadcast_transcript(call_sid, "status", "connected")
                    
                    greeting = "Hello, how can I help you today?"
                    await broadcast_transcript(call_sid, "agent", greeting)
                    save_message(call_sid, "agent", greeting, phone_number)
                    await tts_manager.stream_tts_audio(ws, stream_sid, greeting, stop_tts_event)
                
                logger.info(f"Stream started: {call_sid}")
            
            elif event == "media":
                payload = data.get("media", {}).get("payload", "")
                if payload:
                    mulaw_data = base64.b64decode(payload)
                    
                    await dg_connection.send(mulaw_data)
                    
                    pcm_8k = audioop.ulaw2lin(mulaw_data, 2)
                    pcm_16k, ratecv_state = audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, ratecv_state)
                    silero_buffer.extend(pcm_16k)
                    
                    if len(silero_buffer) >= silero_vad.chunk_bytes():
                        chunk = bytes(silero_buffer[:silero_vad.chunk_bytes()])
                        del silero_buffer[:silero_vad.chunk_bytes()]
                        
                        try:
                            speech_prob = silero_vad(chunk)
                            if speech_prob >= 0.8:
                                vad_last_voice_ts = time.time()
                                last_spoken_time = time.time()
                                
                                if not stop_tts_event.is_set():
                                    stop_tts_event.set()
                                    try:
                                        await ws.send_json({"event": "clear", "streamSid": stream_sid})
                                    except:
                                        pass
                                    if not tts_already_cleared["value"]:
                                        if tts_manager:
                                            await tts_manager.clear_buffer()
                                        tts_already_cleared["value"] = True
                        except:
                            pass
            
            elif event == "stop":
                should_terminate = True
                break
        
        silence_task.cancel()
        process_task.cancel()
        
    except Exception as e:
        logger.error(f"Media stream error: {e}")
    finally:
        try:
            await dg_connection.finish()
        except:
            pass
        if tts_manager:
            await tts_manager.cleanup()
        if call_sid:
            await broadcast_transcript(call_sid, "status", "disconnected")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8125)
