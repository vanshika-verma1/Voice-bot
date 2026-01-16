import io
from dotenv import load_dotenv
load_dotenv()

import wave
import asyncio
import threading
import numpy as np
from typing import Generator, Tuple, List, Optional
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from deepgram import DeepgramClient
from loguru import logger
import uvicorn
import uuid
from pymongo import MongoClient
from datetime import datetime
import os

from pysilero_vad import SileroVoiceActivityDetector
from agent import SimpleAgent

# Logging configuration
logger.remove()
logger.add(
    lambda msg: print(msg),
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
)


# DeepgramClient will be instantiated per session inside ws_chat

# MongoDB Setup
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://vanshikaverma:Test_123@bharatlogic.ngkrzdr.mongodb.net/")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["voice_bot_db"]
conversations_collection = db["conversations"]

def save_conversation_summary(session_id: str, summary: str, user_data: dict):
    """Saves the conversation summary to MongoDB."""
    try:
        document = {
            "session_id": session_id,
            "timestamp": datetime.utcnow(),
            "summary": summary,
            "user_data": user_data
        }
        conversations_collection.insert_one(document)
        logger.info(f"Summary saved for session {session_id}")
    except Exception as e:
        logger.error(f"Failed to save summary: {e}")


def audio_to_bytes(audio: Tuple[int, np.ndarray]) -> bytes:
    """Convert audio tuple to WAV bytes with 44-byte header."""
    sample_rate, audio_data = audio
    if audio_data.dtype != np.int16:
        if audio_data.dtype in [np.float32, np.float64]:
            audio_data = (audio_data * 32767).astype(np.int16)
        else:
            audio_data = audio_data.astype(np.int16)
    
    # Flatten if stereo/2D
    if len(audio_data.shape) > 1:
        audio_data = audio_data.flatten()
    
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    return buffer.getvalue()

class SileroVADWrapper:
    """Wrapper for Silero VAD to handle chunk buffering and state."""
    def __init__(self, silence_duration=0.5, min_speech_duration=0.4):
        self.vad = SileroVoiceActivityDetector()
        self.chunk_samples = self.vad.chunk_samples() 
        self.audio_buffer = np.array([], dtype=np.int16)
        self.pre_roll = deque(maxlen=10) # 10 chunks ≈ 0.3s of pre-roll
        self.speech_buffer = []
        self.is_speaking = False
        self.silence_counter = 0
        self.min_speech_samples = int(min_speech_duration * 16000)
        self.max_silence_chunks = int(silence_duration * 16000 / self.chunk_samples)
        logger.info(f"VAD Initialized: Threshold 0.8, Min speech={min_speech_duration}s")

    def process(self, audio_chunk: np.ndarray):
        """Processes incoming audio and returns state events."""
        self.audio_buffer = np.append(self.audio_buffer, audio_chunk)
        events = []
        
        while len(self.audio_buffer) >= self.chunk_samples:
            chunk = self.audio_buffer[:self.chunk_samples]
            self.audio_buffer = self.audio_buffer[self.chunk_samples:]
            
            # Silero VAD check
            confidence = self.vad(chunk.tobytes())
            is_speech = confidence >= 0.8
            
            if is_speech:
                if not self.is_speaking:
                    # Start speaking: include pre-roll for better STT context
                    self.speech_buffer = list(self.pre_roll)
                    events.append(("start_speech", None))
                self.is_speaking = True
                self.speech_buffer.append(chunk)
                self.silence_counter = 0
            else:
                if self.is_speaking:
                    self.speech_buffer.append(chunk)
                    self.silence_counter += 1
                    if self.silence_counter >= self.max_silence_chunks:
                        # Speech finished - check if it's long enough to be real
                        complete_audio = np.concatenate(self.speech_buffer)
                        self.speech_buffer = []
                        self.is_speaking = False
                        self.silence_counter = 0
                        
                        if len(complete_audio) >= self.min_speech_samples:
                            events.append(("end_speech", complete_audio))
                        else:
                            logger.debug("VAD: Discarded segment (too short)")
                else:
                    self.pre_roll.append(chunk)
        return events

# ==================== AUDIO PROCESSING ====================

def clean_text_for_tts(text: str) -> str:
    """Removes Markdown characters and extra punctuation for cleaner speech."""
    import re
    # Remove bold/italic (**, *, __, _)
    text = re.sub(r'(\*\*|\*|__|_)', '', text)
    # Remove headers (#)
    text = re.sub(r'#+\s*', '', text)
    # Remove list markers at start of lines (-, +, *, 1., 2.)
    text = re.sub(r'^\s*([-+*]|\d+\.)\s+', '', text, flags=re.MULTILINE)
    # Remove links [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Remove code blocks and backticks
    text = re.sub(r'(`|```[a-z]*\n|```)', '', text)
    # Replace newlines with spaces
    text = text.replace('\n', ' ')
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def transcribe_audio(audio: Tuple[int, np.ndarray], dg_client: DeepgramClient) -> str:
    """Transcribe audio using Deepgram Nova-3 REST API."""
    audio_bytes = audio_to_bytes(audio)
    
    # Validation: Skip if audio is empty (44 bytes is just the WAV header)
    if len(audio_bytes) <= 25:
        return ""
        
    logger.debug(f"Sending STT request: {len(audio_bytes)} bytes")
    try:
        response = dg_client.listen.v1.media.transcribe_file(
            request=audio_bytes,
            model="nova-3",
            smart_format=True,
            language="multi"
        )
        return response.results.channels[0].alternatives[0].transcript
    except Exception as e:
        logger.error(f"STT Error: {e}")
        return ""

def text_to_speech(text: str, dg_client: DeepgramClient) -> Generator[Tuple[int, np.ndarray], None, None]:
    """Convert text to speech using Deepgram Aura TTS."""
    if not text.strip():
        return
    try:
        response = dg_client.speak.v1.audio.generate(
            text=text,
            model="aura-2-arcas-en",
            encoding="linear16",
            sample_rate=24000
        )
        audio_bytes = b''.join(response)
        if not audio_bytes:
            return
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        
        # Fade to prevent clicks (50ms fade)
        fade_samples = min(1200, len(audio_array) // 6)
        if fade_samples > 0:
            audio_array[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples)
            audio_array[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples)
            
        audio_array = audio_array.astype(np.int16).reshape(1, -1)
        yield (24000, audio_array)
    except Exception as e:
        logger.error(f"TTS Error: {e}")

def process_chat_common(text: str, session_agent: SimpleAgent, session_history: List[dict], dg_client: DeepgramClient, suppress_tts: bool = False) -> Generator:
    """Shared pipeline: Agent -> per-sentence TTS (if not suppressed)."""
    

    # Check for explicit 'clear' command (no AI response needed)
    clean_text = text.lower().strip().strip('.').strip('!').strip('?')
    
    if clean_text in ["clear history", "clear"]:
        logger.info("Manual clear triggered")
        
        # correct: generate and save summary before clearing
        try:
             # Generate summary
            summary = session_agent.generate_summary()
            
            # Save to DB
            session_id = session_history[0].get("session_id") if session_history and "session_id" in session_history[0] else str(uuid.uuid4())
            save_conversation_summary(session_id, summary, session_agent.user_data)
        except Exception as e:
            logger.error(f"Error saving summary on clear: {e}")

        session_history.clear()
        session_agent.clear_history()
        yield ("clear", None)
        return

    session_history.append({"role": "user", "content": text})

    # LLM & TTS
    full_response = ""
    current_chunk = ""
    first_chunk_sent = False
    
    # Punctuations that trigger TTS. 
    # For the first chunk, we include mid-sentence punctuation for speed.
    first_chunk_endings = {'.', '!', '?', ',', ';', ':', '।', '。', '\n'}
    sentence_endings = {'.', '!', '?', '।', '。', '\n'}
    
    try:
        for event in session_agent.stream(
            {"messages": [{"role": "user", "content": text}]}, 
            stream_mode="messages"
        ):
            message, _ = event
            if hasattr(message, 'content') and message.content:
                current_chunk += message.content
                full_response += message.content
                yield ("typing", full_response)
                
                # Determine if we should trigger TTS
                if not suppress_tts:
                    trigger = False
                    if not first_chunk_sent:
                        word_count = len(current_chunk.strip().split())
                        # Increased threshold to 8 words for smoother first chunk
                        if any(current_chunk.rstrip().endswith(e) for e in first_chunk_endings) or word_count >= 8:
                            trigger = True
                    else:
                        if any(current_chunk.rstrip().endswith(e) for e in sentence_endings):
                            trigger = True

                    if trigger and len(current_chunk.strip()) > 2:
                        clean_text = clean_text_for_tts(current_chunk.strip())
                        if clean_text:
                            logger.info(f'TTS Trigger ({"Fast" if not first_chunk_sent else "Sentence"}): "{clean_text}"')
                            for audio_chunk in text_to_speech(clean_text, dg_client):
                                yield ("audio", audio_chunk)
                        current_chunk = ""
                        first_chunk_sent = True
        
        if not suppress_tts and current_chunk.strip():
            clean_text = clean_text_for_tts(current_chunk.strip())
            if clean_text:
                logger.info(f'TTS Trigger (final): "{clean_text}"')
                for audio_chunk in text_to_speech(clean_text, dg_client):
                    yield ("audio", audio_chunk)
        
        session_history.append({"role": "assistant", "content": full_response})
        yield ("chat_assistant", full_response)
        
        # Check if session should end (User or AI said goodbye)
        user_clean = text.lower()
        ai_clean = full_response.lower()
        end_phrases = ["goodbye", "bye", "end call", "thank you goodbye", "thanks bye", "have a good day", "have a great day"]
        

        if any(p in user_clean for p in end_phrases) or any(p in ai_clean for p in end_phrases):
            logger.info("End phrase detected. Shutting down after playback.")
            
            # Generate and save summary
            try:
                summary = session_agent.generate_summary()
                # We need to access session_id, which we'll stash in history or pass in. 
                # For now, let's grab it if we put it in history, otherwise generate new (not ideal but safe)
                session_id = session_history[0].get("session_id") if session_history and "session_id" in session_history[0] else str(uuid.uuid4())
                save_conversation_summary(session_id, summary, session_agent.user_data)
            except Exception as e:
                logger.error(f"Error saving summary on shutdown: {e}")

            session_history.clear()
            session_agent.clear_history()
            yield ("shutdown", None)
            return 
            
        logger.debug("Task finished successfully")
        
    except Exception as e:
        logger.error(f"Inference Error: {e}")
        yield ("error", str(e))

def process_audio_sync(audio: Tuple[int, np.ndarray], session_agent: SimpleAgent, session_history: List[dict], dg_client: DeepgramClient) -> Generator:
    """Voice pipeline: STT -> Shared Logic with TTS."""
    transcript = transcribe_audio(audio, dg_client)
    if not transcript.strip():
        return
    logger.info(f'Transcribed: "{transcript}"')
    
    yield ("chat_user", transcript)
    yield from process_chat_common(transcript, session_agent, session_history, dg_client, suppress_tts=False)

def process_text_chat(text: str, session_agent: SimpleAgent, session_history: List[dict], dg_client: DeepgramClient) -> Generator:
    """Text pipeline: Shared Logic without TTS."""
    yield from process_chat_common(text, session_agent, session_history, dg_client, suppress_tts=True)

async def run_sync_gen_in_thread(gen_func, *args, stop_flag: threading.Event = None):
    """Bridge sync generator to async iterator with cancellation."""
    q = asyncio.Queue()
    loop = asyncio.get_event_loop()
    def producer():
        try:
            for item in gen_func(*args):
                if stop_flag and stop_flag.is_set():
                    logger.info("Background thread cancellation detected")
                    break
                loop.call_soon_threadsafe(q.put_nowait, {"type": "data", "item": item})
            loop.call_soon_threadsafe(q.put_nowait, {"type": "done"})
        except Exception as e:
            loop.call_soon_threadsafe(q.put_nowait, {"type": "error", "error": e})
    
    thread = threading.Thread(target=producer, daemon=True)
    thread.start()
    
    while True:
        try:
            msg = await q.get()
            if msg["type"] == "data": yield msg["item"]
            elif msg["type"] == "done": break
            elif msg["type"] == "error": break
        except asyncio.CancelledError:
            if stop_flag: stop_flag.set()
            raise

# ==================== FastAPI & UI ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Aura Voice Agent Active - Port 8000")
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voice Agent</title>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Inter', -apple-system, sans-serif; 
            background: #0a1628; 
            color: #ffffff; 
            display: flex; 
            justify-content: center; 
            height: 100vh;
            overflow: hidden;
        }
        .app-container { 
            width: 100%; 
            max-width: 650px; 
            display: flex; 
            flex-direction: column; 
            padding: 2rem 1rem;
            position: relative;
        }
        .header { 
            padding-bottom: 1.5rem; 
            border-bottom: 1px solid rgba(0, 245, 255, 0.1); 
            margin-bottom: 1rem; 
            text-align: center;
        }
        .header h1 { 
            font-size: 1.25rem; 
            font-weight: 700; 
            letter-spacing: -0.02em;
            background: linear-gradient(90deg, #00f5ff, #0090ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .chat-content { 
            flex: 1; 
            overflow-y: auto; 
            display: flex; 
            flex-direction: column; 
            gap: 1rem; 
            padding: 1rem 0.5rem;
            mask-image: linear-gradient(to top, transparent, black 2%);
        }
        .chat-content::-webkit-scrollbar { width: 3px; }
        .chat-content::-webkit-scrollbar-thumb { background: rgba(0, 245, 255, 0.1); border-radius: 2px; }
        .msg { 
            max-width: 85%; 
            padding: 0.8rem 1.1rem; 
            border-radius: 1rem; 
            font-size: 0.95rem; 
            line-height: 1.5; 
            animation: slideUp 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        }
        @keyframes slideUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .msg.user { 
            background: rgba(0, 245, 255, 0.08); 
            border: 1px solid rgba(0, 245, 255, 0.15);
            color: #00f5ff;
            align-self: flex-end; 
            border-bottom-right-radius: 4px; 
        }
        .msg.bot { 
            background: rgba(255, 255, 255, 0.03); 
            border: 1px solid rgba(255, 255, 255, 0.05); 
            color: #e0e0e0;
            align-self: flex-start; 
            border-bottom-left-radius: 4px;
        }
        .msg.bot ul, .msg.bot ol { 
            padding-left: 1.5rem; 
            margin: 0.75rem 0;
            list-style-position: outside;
        }
        .msg.bot li { 
            margin-bottom: 0.5rem; 
            padding-left: 0.2rem;
        }
        .msg.bot li::marker {
            color: #00f5ff;
            font-weight: bold;
        }
        .msg.bot p { margin-bottom: 0.75rem; }
        .msg.bot p:last-child { margin-bottom: 0; }
        .msg.bot strong { color: #00f5ff; }
        .msg.bot a { color: #00f5ff; }
        .msg.interrupted { border-left: 3px solid #ff4b4b; color: rgba(224, 224, 224, 0.6); font-style: italic; }
        .msg.typing::after { content: "▋"; animation: blink 1s infinite; margin-left: 2px; color: #00f5ff; }
        @keyframes blink { 0%, 50% { opacity: 1; } 51%, 100% { opacity: 0; } }
        
        .footer-controls {
            margin-top: 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .status-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0 0.5rem;
        }

        .visualizer { display: flex; align-items: flex-end; gap: 3px; height: 15px; opacity: 0; transition: opacity 0.3s ease; }
        .visualizer.active { opacity: 1; }
        .bar { 
            width: 3px; 
            background: #00f5ff; 
            border-radius: 1px; 
            height: 3px; 
            transition: height 0.08s ease;
            box-shadow: 0 0 5px rgba(0, 245, 255, 0.3);
        }

        .status { font-size: 0.7rem; color: rgba(0, 245, 255, 0.4); font-weight: 500; text-transform: uppercase; letter-spacing: 0.1em; }
        
        .input-bar {
            display: flex;
            gap: 0.6rem;
            align-items: center;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 0.4rem 0.6rem;
            border-radius: 1rem;
            transition: all 0.3s ease;
        }
        .input-bar:focus-within {
            background: rgba(255, 255, 255, 0.05);
            border-color: rgba(0, 245, 255, 0.2);
            box-shadow: 0 0 15px rgba(0, 245, 255, 0.05);
        }
        #textInput {
            flex: 1;
            background: transparent;
            border: none;
            color: #fff;
            padding: 0.5rem 0.4rem;
            font-size: 0.95rem;
            outline: none;
        }
        #textInput::placeholder { color: rgba(255, 255, 255, 0.15); }

        .icon-btn {
            background: transparent;
            border: none;
            color: rgba(255, 255, 255, 0.4);
            width: 36px;
            height: 36px;
            border-radius: 0.75rem;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .icon-btn:hover { color: #fff; background: rgba(255, 255, 255, 0.05); }
        .icon-btn.active { color: #00f5ff; background: rgba(0, 245, 255, 0.1); }
        .icon-btn.active svg { filter: drop-shadow(0 0 5px rgba(0, 245, 255, 0.5)); }
        .icon-btn.primary { background: #00f5ff; color: #000; }
        .icon-btn.primary:hover { background: #00d8e0; transform: scale(1.05); }
        .icon-btn svg { width: 1.2rem; height: 1.2rem; }

        .btn-clear { background: transparent; color: rgba(255, 255, 255, 0.2); border: none; font-size: 0.65rem; cursor: pointer; text-transform: uppercase; letter-spacing: 0.08em; align-self: center; margin-top: 0.5rem; }
        .placeholder { flex: 1; display: flex; align-items: center; justify-content: center; color: rgba(255, 255, 255, 0.15); font-size: 0.85rem; text-align: center; }
    </style>
</head>
<body>
    <div class="app-container">
        <div class="header">
            <h1>Aura Agent</h1>
        </div>
        <div class="chat-content" id="chatArea">
            <div class="placeholder" id="emptyState">Type a message or press the microphone to start talking.</div>
        </div>
        
        <div class="footer-controls">
            <div class="status-row">
                <div class="visualizer" id="viz">
                    <div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div>
                    <div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div>
                </div>
                <div class="status" id="statusMsg">Ready</div>
            </div>

            <div class="input-bar">
                <button class="icon-btn" id="toggleBtn" title="Voice Chat">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="23"></line><line x1="8" y1="23" x2="16" y2="23"></line></svg>
                </button>
                <input type="text" id="textInput" placeholder="Type or talk..." autocomplete="off">
                <button class="icon-btn primary" id="sendBtn">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
                </button>
            </div>
            
            <button class="btn-clear" onclick="clearChat()">Clear history</button>
        </div>
    </div>

    <script>
        let running = false;
        let ws = null;
        let aCtx = null;
        let stream = null;
        let proc = null;
        let aQueue = [];
        let playing = false;
        let pCtx = null;
        let activeSrc = null;
        let pendingShutdown = false;
        let typingMsg = null;
        let scheduledTime = 0;

        const chatArea = document.getElementById('chatArea');
        const emptyState = document.getElementById('emptyState');
        const toggleBtn = document.getElementById('toggleBtn');
        const sendBtn = document.getElementById('sendBtn');
        const textInput = document.getElementById('textInput');
        const statusMsg = document.getElementById('statusMsg');
        const bars = document.querySelectorAll('.bar');
        const viz = document.getElementById('viz');

        const MIC_ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="23"></line><line x1="8" y1="23" x2="16" y2="23"></line></svg>`;
        const STOP_ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect></svg>`;

        toggleBtn.onclick = () => running ? stop() : start();
        sendBtn.onclick = () => sendTextMessage();
        textInput.onkeypress = (e) => { if (e.key === 'Enter') sendTextMessage(); };

        async function sendTextMessage() {
            const text = textInput.value.trim();
            if (!text) return;
            
            if (!ws || ws.readyState !== 1) await startWebSocketOnly();

            if (ws && ws.readyState === 1) {
                if (playing || typingMsg) {
                    killAudio();
                    ws.send(JSON.stringify({type: 'interrupt'}));
                }
                ws.send(JSON.stringify({type: 'chat', content: text}));
                addMsg('user', text);
                textInput.value = '';
                statusMsg.textContent = 'Thinking';
            }
        }

        async function startWebSocketOnly() {
            if (!pCtx) pCtx = new (window.AudioContext || window.webkitAudioContext)();
            const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${proto}//${location.host}/ws/chat`);
            ws.binaryType = 'arraybuffer';
            return new Promise((resolve) => {
                ws.onopen = () => { statusMsg.textContent = 'Connected'; resolve(); };
                ws.onmessage = (e) => e.data instanceof ArrayBuffer ? queueAudio(e.data) : onMsg(JSON.parse(e.data));
                ws.onclose = () => { if (!running) statusMsg.textContent = 'Ready'; };
            });
        }

        let audioBuffer = [];
        const CHUNK_SIZE = 4096;

        async function start() {
            try {
                if (!pCtx) pCtx = new (window.AudioContext || window.webkitAudioContext)();
                if (pCtx.state === 'suspended') await pCtx.resume();

                stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, sampleRate: 16000 } });
                aCtx = new AudioContext({ sampleRate: 16000 });
                
                // AudioWorklet for smoother processing
                const workletCode = `
                  class PCMProcessor extends AudioWorkletProcessor {
                    process(inputs, outputs, parameters) {
                      const input = inputs[0];
                      if (input.length > 0) {
                        this.port.postMessage(input[0]);
                      }
                      return true;
                    }
                  }
                  registerProcessor('pcm-processor', PCMProcessor);
                `;
                const blob = new Blob([workletCode], { type: 'application/javascript' });
                await aCtx.audioWorklet.addModule(URL.createObjectURL(blob));

                const src = aCtx.createMediaStreamSource(stream);
                const node = new AudioWorkletNode(aCtx, 'pcm-processor');
                
                node.port.onmessage = (e) => {
                    if (!running || !ws || ws.readyState !== 1) return;
                    
                    // Buffer chunks (Worklet is ~128 samples, we want ~4096)
                    const chunks = e.data;
                    for (let i = 0; i < chunks.length; i++) audioBuffer.push(chunks[i]);
                    
                    if (audioBuffer.length >= CHUNK_SIZE) {
                        const d = new Float32Array(audioBuffer.slice(0, CHUNK_SIZE));
                        audioBuffer = audioBuffer.slice(CHUNK_SIZE);
                        
                        updateViz(d);
                        const p = new Int16Array(d.length);
                        for (let i = 0; i < d.length; i++) p[i] = Math.max(-32768, Math.min(32767, d[i] * 32768));
                        ws.send(p.buffer);
                    }
                };

                src.connect(node);
                node.connect(aCtx.destination); // Keep alive
                proc = node;

                if (!ws || ws.readyState !== 1) {
                    await startWebSocketOnly();
                }
                
                running = true; 
                toggleBtn.innerHTML = STOP_ICON; 
                toggleBtn.classList.add('active'); 
                statusMsg.textContent = 'Listening'; 
                viz.classList.add('active');
            } catch (err) { console.error(err); alert('Microphone access needed.'); }
        }

        function stop(clearUI = false) {
            running = false;
            // Clear audio buffer
            audioBuffer = [];
            
            if (proc) {
                 proc.disconnect();
                 proc.port.onmessage = null; // Remove event listener
            }
            if (aCtx) aCtx.close();
            if (stream) stream.getTracks().forEach(t => t.stop());
            // WEBSOCKET REMAINS OPEN FOR SESSION PERSISTENCE
            toggleBtn.innerHTML = MIC_ICON;
            toggleBtn.classList.remove('active');
            statusMsg.textContent = 'Ready';
            viz.classList.remove('active');
            resetViz();
            killAudio();
            if (clearUI) {
                if (ws) ws.close();
                clearChatUI();
            }
        }
        
        window.onbeforeunload = () => { if (ws && ws.readyState === 1) ws.send(JSON.stringify({type: 'clear'})); if (ws) ws.close(); };

        function killAudio() { 
            aQueue = []; 
            playing = false; scheduledTime = 0;
            if (activeSrc) { try { activeSrc.stop(); } catch(e) {} activeSrc = null; } 
        }

        function onMsg(data) {
            switch (data.type) {
                case 'processing': statusMsg.textContent = 'Thinking'; break;
                case 'user_message': 
                    // Only add if it's not the last text message we just sent manually
                    if (emptyState.style.display === 'flex' || chatArea.lastElementChild.textContent !== data.content) {
                        addMsg('user', data.content); 
                    }
                    break;
                case 'typing': showTyping(data.content); statusMsg.textContent = 'Speaking'; break;
                case 'assistant_message': doneTyping(data.content); statusMsg.textContent = (running ? 'Listening' : 'Ready'); break;
                case 'interrupt': 
                    killAudio(); 
                    if (typingMsg) { typingMsg.classList.add('interrupted'); doneTyping(typingMsg.textContent + " ... [Interrupted]"); }
                    break;
                case 'clear': killAudio(); stop(true); statusMsg.textContent = 'Cleared'; break;
                case 'shutdown': pendingShutdown = true; if (!playing) stop(true); break;
            }
        }

        function clearChatUI() { chatArea.innerHTML = ''; chatArea.appendChild(emptyState); emptyState.style.display = 'flex'; typingMsg = null; }

        async function queueAudio(buf) { 
            try { const audioBuf = await pCtx.decodeAudioData(buf); aQueue.push(audioBuf); if (!playing) play(); } catch (err) { console.error("Audio decode error", err); }
        }

        async function play() {
            if (aQueue.length === 0) { playing = false; if (pendingShutdown) setTimeout(() => stop(true), 500); return; }
            playing = true;
            const audioBuf = aQueue.shift();
            try {
                const src = pCtx.createBufferSource();
                src.buffer = audioBuf;
                src.connect(pCtx.destination);
                activeSrc = src;
                const now = pCtx.currentTime;
                if (scheduledTime < now) scheduledTime = now + 0.02;
                src.start(scheduledTime);
                scheduledTime += audioBuf.duration;
                src.onended = () => play();
            } catch (e) { play(); }
        }

        function addMsg(role, text) {
            emptyState.style.display = 'none';
            const m = document.createElement('div');
            m.className = `msg ${role == 'user' ? 'user' : 'bot'}`;
            if (role === 'bot') {
                m.innerHTML = marked.parse(text);
            } else {
                m.textContent = text;
            }
            chatArea.appendChild(m);
            chatArea.scrollTop = chatArea.scrollHeight;
            return m;
        }

        function showTyping(text) {
            if (!typingMsg) typingMsg = addMsg('bot', '');
            typingMsg.classList.add('typing');
            typingMsg.innerHTML = marked.parse(text);
            chatArea.scrollTop = chatArea.scrollHeight;
        }

        function doneTyping(text) {
            if (typingMsg) { typingMsg.classList.remove('typing'); typingMsg.innerHTML = marked.parse(text); typingMsg = null; }
            else if (text) { addMsg('bot', text); }
            chatArea.scrollTop = chatArea.scrollHeight;
        }

        function updateViz(data) {
            const step = Math.floor(data.length / bars.length);
            for (let i = 0; i < bars.length; i++) {
                let sum = 0;
                for (let j = 0; j < step; j++) sum += Math.abs(data[i * step + j]);
                bars[i].style.height = Math.max(3, Math.min(18, (sum / step) * 120)) + 'px';
            }
        }

        function resetViz() { bars.forEach(b => b.style.height = '3px'); }
        function clearChat() { if (ws && ws.readyState === 1) ws.send(JSON.stringify({type: 'clear'})); stop(true); }

        window.addEventListener('DOMContentLoaded', () => {
             // Auto-connect on load
             startWebSocketOnly();
        });
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index(): return HTML_PAGE

@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    await websocket.accept()
    logger.info("Connection accepted")
    vad = SileroVADWrapper()
    response_task: Optional[asyncio.Task] = None
    
    # Session-specific state
    dg_client = DeepgramClient()
    session_agent = SimpleAgent(use_rag=True)

    session_history: List[dict] = []
    
    # Generate Session ID
    session_id = str(uuid.uuid4())
    # Stash session_id in the first history item as metadata (hidden) or just keep track
    # We'll put it in a metadata dict in the first item if we want to retrieve it later easily
    # or just use scope. But passing it to functions is cleaner.
    # For now, let's just append a metadata item that won't be rendered
    session_history.append({"session_id": session_id, "role": "system", "content": "Session Init"})

    
    # Send welcome message
    welcome_text = "Hi, I am Aura from BharatLogic. How can I help you today?"
    try:
        await websocket.send_json({"type": "typing", "content": welcome_text})
        # Update agent history
        from langchain_core.messages import AIMessage
        session_agent.history.append(AIMessage(content=welcome_text))
        session_history.append({"role": "assistant", "content": welcome_text})
        
        # Audio for welcome message (disabled as per request)
        # for audio_chunk in text_to_speech(welcome_text, dg_client):
        #     await websocket.send_bytes(audio_to_bytes(audio_chunk))
        
        await websocket.send_json({"type": "assistant_message", "content": welcome_text})
    except Exception as e:
        logger.error(f"Error sending welcome message: {e}")
    
    try:
        await websocket.send_json({"type": "listening"})
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                logger.info("Websocket message: Disconnect received")
                break
                
            if "bytes" in message:
                data = message["bytes"]
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                
                # Process VAD
                events = vad.process(audio_chunk)
                for event_type, payload in events:
                    if event_type == "start_speech":
                        # Always send interrupt to stop active TTS on client
                        await websocket.send_json({"type": "interrupt"})
                        if response_task and not response_task.done():
                            logger.info("In-progress response task cancelled")
                            response_task.cancel()
                    
                    elif event_type == "end_speech":
                        logger.info("Speech segment complete, start inference")
                        audio_tuple = (16000, payload)
                        response_task = asyncio.create_task(handle_inference(websocket, session_agent, session_history, dg_client, process_audio_sync, audio_tuple))
            
            elif "text" in message:
                import json
                try:
                    payload = json.loads(message["text"])
                    if payload.get("type") == "chat":
                        content = payload.get("content")
                        logger.info(f"Received text chat: {content}")
                        # Interrupt previous response if any
                        if response_task and not response_task.done():
                            response_task.cancel()
                        response_task = asyncio.create_task(handle_inference(websocket, session_agent, session_history, dg_client, process_text_chat, content))
                    
                    elif payload.get("type") == "interrupt":
                        if response_task and not response_task.done():
                            logger.info("Manual interrupt received, cancelling task")
                            response_task.cancel()
                            
                    elif payload.get("type") == "clear":
                        logger.warning("Backend memory reset confirmed by UI command")
                        session_history.clear()
                        session_agent.clear_history()
                        await websocket.send_json({"type": "clear"})
                    elif payload.get("type") == "shutdown":
                        logger.warning("Backend received shutdown signal")
                        await websocket.send_json({"type": "shutdown"})
                except Exception as e:
                    logger.error(f"Text parse error: {e}")

                    
    except WebSocketDisconnect: 
        logger.info("Client disconnected")
    except Exception as e: 
        if "Cannot call \"receive\" once a disconnect message has been received" in str(e):
            logger.debug("Websocket receive suppressed after disconnect")
        else:
            logger.error(f"WS error: {e}")

    finally:
        if response_task: 
            logger.info("Cleaning up session: cancelling active response task")
            response_task.cancel()
        session_history.clear()
        session_agent.clear_history()
        logger.info("Session state cleared")

async def handle_inference(websocket: WebSocket, session_agent: SimpleAgent, session_history: List[dict], dg_client: DeepgramClient, func, *args):
    """Handles the async-sync inference pipeline with comprehensive error handling."""
    stop_flag = threading.Event()
    try:
        await websocket.send_json({"type": "processing"})
        typing_count = 0
        async for event_type, content in run_sync_gen_in_thread(func, *args, session_agent, session_history, dg_client, stop_flag=stop_flag):
            try:
                if event_type == "chat_user": 
                    await websocket.send_json({"type": "user_message", "content": content})
                elif event_type == "typing": 
                    typing_count += 1
                    await websocket.send_json({"type": "typing", "content": content})
                elif event_type == "clear": 
                    await websocket.send_json({"type": "clear"})
                elif event_type == "shutdown": 
                    await websocket.send_json({"type": "shutdown"})
                elif event_type == "audio":
                    sr, arr = content
                    await websocket.send_bytes(audio_to_bytes((sr, arr)))
                elif event_type == "chat_assistant": 
                    logger.info(f"Final response sent. Typing events emitted: {typing_count}")
                    await websocket.send_json({"type": "assistant_message", "content": content})
                elif event_type == "error": 
                    await websocket.send_json({"type": "error", "message": content})
            except Exception as send_error:
                logger.error(f"Error sending {event_type}: {send_error}")
                # If we can't send, the connection is likely broken, so break the loop
                break
    except asyncio.CancelledError: 
        logger.info("Response task cancelled, signalling background thread")
        stop_flag.set()
    except Exception as e: 
        logger.error(f"Inference pipeline failure: {e}")
        try:
            await websocket.send_json({"type": "error", "message": "An error occurred processing your request."})
        except:
            pass  # Connection already closed

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)