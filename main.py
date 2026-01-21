import io
from dotenv import load_dotenv
load_dotenv()
import queue
import time
import wave
import asyncio
import threading
import numpy as np
from typing import Generator, AsyncGenerator, Tuple, List, Optional
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.cors import CORSMiddleware
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions, SpeakOptions, SpeakWSOptions, SpeakWebSocketEvents
from loguru import logger
import uvicorn
import uuid
from pymongo import MongoClient
from datetime import datetime
import os

from pysilero_vad import SileroVoiceActivityDetector
from agent import SimpleAgent

logger.remove()
logger.add(
    lambda msg: print(msg),
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
)

MONGO_URI = os.getenv("MONGO_URI")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["voice_bot_db"]
conversations_collection = db["conversations"]
chat_logs_collection = db["chat_logs"]

def save_conversation_summary(session_id: str, summary_data: dict, chat_history: List[dict]):
    """Saves the conversation summary and full chat logs to MongoDB.
    Only saves if name AND (phone OR email) are present."""
    try:
        name = summary_data.get("name")
        phone = summary_data.get("phone")
        email = summary_data.get("email")
        
        # Skip save if name is missing or no contact info (Commented out to save all chats)
        # if not name or (not phone and not email):
        #     logger.info(f"Skipping save for session {session_id}: missing name or contact info")
        #     return
        
        # 1. Save Summary
        summary_doc = {
            "session_id": session_id,
            "timestamp": datetime.utcnow(),
            "name": name,
            "phone": phone,
            "email": email,
            "summary": summary_data.get("summary", "")
        }
        conversations_collection.insert_one(summary_doc)
        
        # 2. Save Full Chat Logs
        chat_log_doc = {
            "session_id": session_id,
            "timestamp": datetime.utcnow(),
            "chat_history": chat_history
        }
        chat_logs_collection.insert_one(chat_log_doc)
        
        logger.info(f"Summary and Full Chat Logs saved for session {session_id}")
    except Exception as e:
        logger.error(f"Failed to save conversation data: {e}")


def audio_to_bytes(audio: Tuple[int, np.ndarray]) -> bytes:
    """Convert audio tuple to WAV bytes with 44-byte header."""
    sample_rate, audio_data = audio
    if audio_data.dtype != np.int16:
        if audio_data.dtype in [np.float32, np.float64]:
            audio_data = (audio_data * 32767).astype(np.int16)
        else:
            audio_data = audio_data.astype(np.int16)
    
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

class DeepgramTTSManager:
    """Manages TTS with persistent connection - sequential processing."""
    
    def __init__(self, dg_client: DeepgramClient):
        self.dg_client = dg_client
        self._cancelled = False
        self._connection = None
        self._audio_queue = None
        self._is_connected = False
        self._lock = threading.Lock()
        self._speak_lock = threading.Lock()  # Ensures one speak at a time
        
    def connect(self):
        """Establish persistent TTS connection."""
        if self._is_connected:
            return True
            
        with self._lock:
            if self._is_connected:
                return True
                
            try:
                self._audio_queue = queue.Queue()
                self._connection = self.dg_client.speak.websocket.v("1")
                
                audio_q = self._audio_queue
                
                def on_binary_data(self_dg, data, **kwargs):
                    audio_q.put(("audio", data))
                
                def on_close(self_dg, close, **kwargs):
                    logger.debug("TTS: Connection closed by server")
                    audio_q.put(("close", None))
                
                self._connection.on(SpeakWebSocketEvents.AudioData, on_binary_data)
                self._connection.on(SpeakWebSocketEvents.Close, on_close)
                
                options = SpeakWSOptions(
                    model="aura-2-andromeda-en",
                    encoding="linear16",
                    sample_rate=16000,
                )
                
                if self._connection.start(options):
                    self._is_connected = True
                    logger.info("TTS: Persistent connection established")
                    return True
                else:
                    logger.error("TTS: Failed to start connection")
                    return False
            except Exception as e:
                logger.error(f"TTS connection error: {e}")
                return False
    
    def speak_sync(self, text: str) -> bytes:
        """Send text and return audio bytes - SEQUENTIAL, one at a time."""
        if not text.strip():
            return b''
        
        # Lock ensures only one speak() runs at a time
        with self._speak_lock:
            if self._cancelled:
                self._cancelled = False
                return b''
            
            if not self._is_connected:
                if not self.connect():
                    return b''
            
            # Clear any old audio in queue
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except:
                    break
            
            logger.debug(f"TTS: Sending text: '{text[:40]}...'")
            
            try:
                self._connection.send_text(text)
                self._connection.flush()
            except Exception as e:
                logger.error(f"TTS send error: {e}")
                self._is_connected = False
                # Retry with new connection
                if self.connect():
                    try:
                        self._connection.send_text(text)
                        self._connection.flush()
                    except:
                        return b''
                else:
                    return b''
            
            start_time = time.time()
            last_audio_time = start_time
            started_receiving = False
            accumulated_bytes = bytearray()
            
            while True:
                if self._cancelled:
                    logger.debug("TTS cancelled")
                    self._cancelled = False
                    return b''
                
                try:
                    msg_type, data = self._audio_queue.get(timeout=0.1)
                    if msg_type == "audio":
                        started_receiving = True
                        last_audio_time = time.time()
                        accumulated_bytes.extend(data)
                    elif msg_type == "close":
                        self._is_connected = False
                        break
                except queue.Empty:
                    now = time.time()
                    # Wait for complete audio (0.5s silence = done)
                    if started_receiving and (now - last_audio_time > 0.5):
                        break
                    if not started_receiving and (now - start_time > 5.0):
                        logger.warning("TTS: No audio received, timeout")
                        return b''
            
            logger.debug(f"TTS: Received {len(accumulated_bytes)} bytes")
            return bytes(accumulated_bytes)
    
    def speak(self, text: str):
        """Generator wrapper for speak_sync."""
        audio_bytes = self.speak_sync(text)
        if audio_bytes:
            try:
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                yield (16000, audio_array.reshape(1, -1))
            except Exception as e:
                logger.error(f"TTS audio error: {e}")
    
    def cancel(self):
        """Cancel current TTS."""
        self._cancelled = True
        if self._audio_queue:
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except:
                    break
    
    def close(self):
        """Close persistent connection."""
        self._cancelled = True
        with self._lock:
            if self._connection and self._is_connected:
                try:
                    self._connection.finish()
                except:
                    pass
                self._is_connected = False
        logger.info("TTS Manager closed")

def clean_text_for_tts(text: str) -> str:
    """Removes Markdown characters and extra punctuation for cleaner speech."""
    import re
    text = re.sub(r'(\*\*|\*|__|_)', '', text)
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'^\s*([-+*]|\d+\.)\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'(`|```[a-z]*\n|```)', '', text)
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def transcribe_audio(audio: Tuple[int, np.ndarray], dg_client: DeepgramClient) -> str:
    """Transcribe audio using Deepgram Nova-3 REST API."""
    audio_bytes = audio_to_bytes(audio)
    
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
    """Convert text to speech using Deepgram Aura TTS WebSocket API for streaming."""
    if not text.strip():
        return

    logger.info(f"TTS Start: '{text[:30]}...'")
    import queue
    audio_queue = queue.Queue()

    async def run_ws_tts():
        try:
            # Create a live connection
            dg_connection = dg_client.speak.websocket.v("1")
            import time
            last_audio_time = [None] # Use list for mutable closure access or use nonlocal

            def on_binary_data(self, data, **kwargs):
                # logger.debug(f"TTS WS: Received {len(data)} bytes")
                audio_queue.put(data)
                last_audio_time[0] = time.time()

            # Wait for the stream to close
            close_event = asyncio.Event()
            
            def on_close(self, close, **kwargs):
                logger.debug("TTS WS: Connection closed by server")
                close_event.set()

            dg_connection.on(SpeakWebSocketEvents.AudioData, on_binary_data)
            dg_connection.on(SpeakWebSocketEvents.Close, on_close)
            
            options = SpeakWSOptions(
                model="aura-2-andromeda-en",
                encoding="linear16",
                sample_rate=16000,
            )
            
            if dg_connection.start(options) is False:
                logger.error("Failed to start Deepgram TTS WS")
                audio_queue.put(None)
                return

            logger.debug(f"TTS WS: Sending text: '{text[:30]}...'")
            dg_connection.send_text(text)
            dg_connection.flush()
            
            # Idle timeout logic (mimicking Twilio example)
            # Wait for audio to start, then wait for it to stop (silence gap)
            start_wait = time.time()
            IDLE_TIMEOUT = 0.5  # 500ms silence = done
            MAX_WAIT = 10.0     # Max time to wait for ANYTHING
            
            while True:
                await asyncio.sleep(0.05)
                now = time.time()
                
                # Check for explicit close
                if close_event.is_set():
                    logger.debug("TTS WS: Closed event received early")
                    break

                # If we have received audio, check for idle silence
                if last_audio_time[0] is not None:
                    if (now - last_audio_time[0]) >= IDLE_TIMEOUT:
                        logger.debug("TTS WS: Idle timeout reached, assuming complete")
                        break
                
                # If we haven't received ANY audio yet, check global timeout
                elif (now - start_wait) >= MAX_WAIT:
                    logger.warning("TTS WS: Max wait reached without completion")
                    break

            # Cleanup
            dg_connection.finish()
            audio_queue.put(None)

        except Exception as e:
            logger.error(f"TTS WS Error: {e}")
            audio_queue.put(None)

    # Run async TTS in a separate thread to yield chunks as they arrive
    tts_thread = threading.Thread(target=lambda: asyncio.run(run_ws_tts()))
    tts_thread.start()

    accumulated_bytes = bytearray()
    
    # Threshold for buffering (e.g., 0.25 seconds of 16kHz mono 16-bit audio = 8000 bytes)
    # Reduces overhead of WAV headers and browser decoding calls
    BUFFER_THRESHOLD = 3200

    while True:
        chunk = audio_queue.get()
        if chunk is None:
            # End of stream: yield remaining buffer if any
            if accumulated_bytes:
                try:
                    logger.debug(f"TTS: Yielding final buffer {len(accumulated_bytes)} bytes")
                    audio_array = np.frombuffer(accumulated_bytes, dtype=np.int16).astype(np.float32)
                    audio_array = np.clip(audio_array, -32767.0, 32767.0)
                    out_array = audio_array.astype(np.int16).reshape(1, -1)
                    yield (16000, out_array)
                except Exception as e:
                    logger.error(f"Error processing final TTS chunk: {e}")
            break
        
        accumulated_bytes.extend(chunk)

        if len(accumulated_bytes) >= BUFFER_THRESHOLD:
            # We now prefer to wait for the whole sentence to ensure smoothness
            # preventing "choppy" playback within a sentence.
            # But if it gets TOO big (e.g. > 3 seconds), we yield to prevent massive latency.
            if len(accumulated_bytes) > 96000: # ~3 seconds
                 try:
                    logger.debug(f"TTS: Yielding large buffer {len(accumulated_bytes)} bytes")
                    audio_array = np.frombuffer(accumulated_bytes, dtype=np.int16).astype(np.float32)
                    audio_array = np.clip(audio_array, -32767.0, 32767.0)
                    out_array = audio_array.astype(np.int16).reshape(1, -1)
                    yield (16000, out_array)
                    accumulated_bytes = bytearray()
                 except Exception as e:
                    logger.error(f"Error processing TTS chunk: {e}")
                    accumulated_bytes = bytearray()

    tts_thread.join()

# def process_chat_common(text: str, session_agent: SimpleAgent, session_history: List[dict], dg_client: DeepgramClient, executor, stop_flag: threading.Event = None, suppress_tts: bool = False) -> Generator:
def process_chat_common(text: str, session_agent: SimpleAgent, session_history: List[dict], dg_client: DeepgramClient, executor, tts_manager: DeepgramTTSManager = None, stop_flag: threading.Event = None, suppress_tts: bool = False) -> Generator:
    """Shared pipeline: Agent -> parallel per-sentence TTS with word-count fallback."""
    
    clean_text = text.lower().strip().strip('.').strip('!').strip('?')
    end_phrases = ["goodbye", "bye", "have a great day", "take care"]
    
    # 1. Immediate End Check
    if any(p == clean_text or clean_text.startswith(p + " ") or clean_text.endswith(" " + p) for p in end_phrases):
        logger.info(f"Immediate end phrase detected: {clean_text}")
        try:
            if len(session_agent.history) > 1:
                summary = session_agent.generate_summary()
                session_id = session_history[0].get("session_id") if session_history and "session_id" in session_history[0] else str(uuid.uuid4())
                save_conversation_summary(session_id, summary, session_history)
        except Exception as e:
            logger.error(f"Error saving summary on immediate goodbye: {e}")

        session_history.clear()
        session_agent.clear_history()
        yield ("shutdown", None)
        return

    if clean_text in ["clear history", "clear"]:
        logger.info("Manual text-chat clear triggered")
        try:
            if len(session_agent.history) > 1:
                summary = session_agent.generate_summary()
                session_id = session_history[0].get("session_id") if session_history and "session_id" in session_history[0] else str(uuid.uuid4())
                save_conversation_summary(session_id, summary, session_history)
        except Exception as e:
            logger.error(f"Error saving summary on text clear: {e}")

        session_history.clear()
        session_agent.clear_history()
        yield ("clear", None)
        return

    session_history.append({"role": "user", "content": text})
    full_response = ""
    buffer = ""
    
    # Queue management for parallel streaming
    # We use a deque of Queues to maintain sentence order:
    # [Queue(Sentence1), Queue(Sentence2), ...]
    from collections import deque
    import queue
    import concurrent.futures
    
    audio_queues = deque() 
    
    # def tts_stream_feeder(text_input, output_q):
    #     """Feeds audio chunks from generator to a queue."""
    #     try:
    #         clean_txt = clean_text_for_tts(text_input)
    #         if clean_txt:
    #             for chunk in text_to_speech(clean_txt, dg_client):
    #                 output_q.put(("audio", chunk))
    #     except Exception as e:
    #         logger.error(f"TTS Stream Error: {e}")
    #     finally:
    #         output_q.put(("done", None))
    def tts_stream_feeder(text_input, output_q):
        """Feeds audio chunks from generator to a queue - uses sync method for parallel processing."""
        try:
            clean_txt = clean_text_for_tts(text_input)
            if clean_txt and tts_manager:
                # Get audio bytes directly (non-blocking for queue)
                audio_bytes = tts_manager.speak_sync(clean_txt)
                if audio_bytes:
                    try:
                        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                        output_q.put(("audio", (16000, audio_array.reshape(1, -1))))
                    except Exception as e:
                        logger.error(f"TTS audio conversion error: {e}")
        except Exception as e:
            logger.error(f"TTS Stream Error: {e}")
        finally:
            output_q.put(("done", None))

    def poll_audio_queues():
        """Helper to drain ready audio chunks from the active queue."""
        chunks = []
        # While we have active queues
        while audio_queues:
            current_q = audio_queues[0] # Look at the first one (in order)
            try:
                while True:
                    # Non-blocking get
                    item_type, item_data = current_q.get_nowait()
                    if item_type == "audio":
                        chunks.append(item_data)
                    elif item_type == "done":
                        # This sentence is fully consumed
                        audio_queues.popleft()
                        break # Break inner loop, check next queue in outer loop
            except queue.Empty:
                # Current queue has no more data right now, but isn't done
                break 
        return chunks

    is_first_chunk = True
    
    try:
        executor_stream = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        
        for event in session_agent.stream(
            {"messages": [{"role": "user", "content": text}]}, 
            stream_mode="messages"
        ):
            message, _ = event
            if hasattr(message, 'content') and message.content:
                chunk_text = message.content
                full_response += chunk_text
                buffer += chunk_text
                
                # Yield regular typing update
                # yield ("typing", chunk_text) # Optional: if needed by UI
                
                # Check for punctuation to splice sentences
                # Check for punctuation OR word count to splice sentences
                words_in_buffer = len(buffer.split())
                has_sentence_end = any(p in buffer for p in [".", "?", "!"])
                has_pause_punct = "," in buffer and words_in_buffer >= 8
                should_force_split = words_in_buffer >= 15  # Force split at 15 words

                if has_sentence_end or has_pause_punct or should_force_split:
                    import re
                    
                    if has_sentence_end:
                        # Split by sentence-ending punctuation
                        parts = re.split(r'([.?!])\s*', buffer)
                    elif has_pause_punct:
                        # Split by comma for natural pause
                        parts = re.split(r'(,)\s*', buffer)
                    else:
                        # Force split at word boundary (take first 12 words)
                        words = buffer.split()
                        first_part = ' '.join(words[:12])
                        remainder = ' '.join(words[12:])
                        parts = [first_part, '', remainder] if remainder else [first_part, '']
                    
                    if len(parts) > 1:
                        # parts like: ["Hello", ".", " How are you", "?", ""]
                        processed_buffer = ""
                        
                        idx = 0
                        while idx < len(parts) - 1:
                            sentence = parts[idx] + (parts[idx+1] if idx+1 < len(parts) else "")
                            idx += 2
                            
                            if sentence.strip() and len(sentence.strip()) > 2:
                                is_first_chunk = False
                                # Start streaming TTS for this sentence
                                q = queue.Queue()
                                audio_queues.append(q)
                                executor_stream.submit(tts_stream_feeder, sentence, q)
                            
                            processed_buffer += sentence
                        
                        # Any remainder goes back to buffer
                        if idx < len(parts):
                            buffer = parts[idx]
                        else:
                            buffer = ""

            # INTERLEAVED POLLING: Check for audio while LLM generates
            ready_chunks = poll_audio_queues()
            for ac in ready_chunks:
                yield ("audio", ac)
                
            if stop_flag and stop_flag.is_set():
                break
        
        # Final buffer processing
        if buffer.strip():
            if not suppress_tts:
                q = queue.Queue()
                audio_queues.append(q)
                executor_stream.submit(tts_stream_feeder, buffer.strip(), q)
        
        # Drain remaining audio
        while audio_queues:
            if stop_flag and stop_flag.is_set(): 
                logger.info("Parallel TTS processing halted due to interruption")
                break
            
            ready_chunks = poll_audio_queues()
            for ac in ready_chunks:
                yield ("audio", ac)
            
            # Since we're done generating text, we can afford a small sleep to prevent busy loop 
            # while waiting for final audio
            if audio_queues:
                import time
                time.sleep(0.05)

        executor_stream.shutdown(wait=False)

    except Exception as e:
        logger.error(f"Inference Loop Error: {e}")
        yield ("error", str(e))

    session_history.append({"role": "assistant", "content": full_response})
    yield ("chat_assistant", full_response)

    # 2. End-of-response Goodbye Check (if not already handled)
    ai_clean = full_response.lower()
    if any(p in ai_clean for p in end_phrases):
        logger.info("End phrase detected in AI response.")
        try:
            if len(session_agent.history) > 1:
                summary = session_agent.generate_summary()
                session_id = session_history[0].get("session_id") if session_history and "session_id" in session_history[0] else str(uuid.uuid4())
                save_conversation_summary(session_id, summary, session_history)
        except Exception as e:
            logger.error(f"Error saving summary on AI goodbye: {e}")

        session_history.clear()
        session_agent.clear_history()
        yield ("shutdown", None)

# def process_audio_sync(audio: Tuple[int, np.ndarray], session_agent: SimpleAgent, session_history: List[dict], dg_client: DeepgramClient, executor, stop_flag: threading.Event = None) -> Generator:
def process_audio_sync(audio: Tuple[int, np.ndarray], session_agent: SimpleAgent, session_history: List[dict], dg_client: DeepgramClient, executor, tts_manager: DeepgramTTSManager = None, stop_flag: threading.Event = None) -> Generator:
    """Voice pipeline: STT -> Shared Logic with TTS."""
    transcript = transcribe_audio(audio, dg_client)
    if not transcript.strip():
        return
    logger.info(f'Transcribed: "{transcript}"')
    
    yield ("chat_user", transcript)
    # yield from process_chat_common(transcript, session_agent, session_history, dg_client, executor, stop_flag=stop_flag, suppress_tts=False)
    yield from process_chat_common(transcript, session_agent, session_history, dg_client, executor, tts_manager=tts_manager, stop_flag=stop_flag, suppress_tts=False)

# def process_text_chat(text: str, session_agent: SimpleAgent, session_history: List[dict], dg_client: DeepgramClient, executor, stop_flag: threading.Event = None) -> Generator:
def process_text_chat(text: str, session_agent: SimpleAgent, session_history: List[dict], dg_client: DeepgramClient, executor, tts_manager: DeepgramTTSManager = None, stop_flag: threading.Event = None) -> Generator:   
    """Text pipeline: Shared Logic without TTS."""
    # yield from process_chat_common(text, session_agent, session_history, dg_client, executor, stop_flag=stop_flag, suppress_tts=True)
    yield from process_chat_common(text, session_agent, session_history, dg_client, executor, tts_manager=tts_manager, stop_flag=stop_flag, suppress_tts=True)
async def run_sync_gen_in_thread(gen_func, *args, stop_flag: threading.Event = None, **kwargs):
    """Bridge sync generator to async iterator with cancellation support."""
    q = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def worker():
        try:
            for item in gen_func(*args, **kwargs, stop_flag=stop_flag):
                if stop_flag and stop_flag.is_set():
                    break
                loop.call_soon_threadsafe(q.put_nowait, {"type": "data", "item": item})
            loop.call_soon_threadsafe(q.put_nowait, {"type": "done"})
        except Exception as e:
            logger.error(f"Sync generator worker error: {e}")
            loop.call_soon_threadsafe(q.put_nowait, {"type": "error", "error": str(e)})
    
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    
    while True:
        try:
            msg = await q.get()
            if msg["type"] == "data": 
                yield msg["item"]
            elif msg["type"] == "done": 
                break
            elif msg["type"] == "error": 
                yield ("error", msg["error"])
                break
        except asyncio.CancelledError:
            if stop_flag: stop_flag.set()
            raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Lifespan: Pre-loading RAG vector store...")
    from rag.retriever import get_relevant_context
    # Run a dummy search to trigger singleton initialization and index loading
    try:
        get_relevant_context("initialization warmup")
    except Exception as e:
        logger.error(f"Lifespan: RAG warmup failed: {e}")
        
    logger.info("Aura Voice Agent Active - Port 8000")
    yield
    logger.info("Shutting down: closing resources...")
    mongo_client.close()
    logger.info("MongoDB connection closed.")

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
        * { margin: 0; padding: 0; box-sizing: border-box; font-family: inherit; }
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
            font-family: inherit;
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

        /* Session Ended Banner (Bottom) */
        .ended-overlay {
            background: rgba(10, 22, 40, 0.95);
            border-top: 1px solid rgba(0, 245, 255, 0.3);
            display: none;
            flex-direction: row;
            align-items: center;
            justify-content: space-between;
            padding: 0.8rem 1.2rem;
            z-index: 100;
            animation: slideInBottom 0.3s ease-out;
            position: relative;
            margin-top: -10px;
            margin-bottom: 10px;
            border-radius: 0.75rem;
            box-shadow: 0 -5px 20px rgba(0, 0, 0, 0.3);
        }
        @keyframes slideInBottom { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        
        .ended-content {
            display: flex;
            align-items: center;
            gap: 1rem;
            width: 100%;
            justify-content: space-between;
        }
        .ended-text h2 { color: #00f5ff; font-size: 0.9rem; margin-bottom: 2px; }
        .ended-text p { color: rgba(255, 255, 255, 0.5); font-size: 0.75rem; }
        
        .btn-restart {
            background: #00f5ff;
            color: #000;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s ease;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-size: 0.75rem;
            white-space: nowrap;
            font-family: inherit;
        }
        .btn-restart:hover { transform: translateY(-1px); box-shadow: 0 3px 10px rgba(0, 245, 255, 0.2); }

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
            <!-- Session Ended Banner -->
            <div class="ended-overlay" id="endedOverlay">
                <div class="ended-content">
                    <div class="ended-text">
                        <h2>Chat Ended</h2>
                    </div>
                    <button class="btn-restart" onclick="resetSession()">Start New Chat</button>
                </div>
            </div>

            <div class="hint-row" style="text-align: center; margin-bottom: 0.5rem;">
                <span style="opacity: 0.6; font-size: 0.75rem; color: #00f5ff;">Please turn on microphone to start voice chat.</span>
            </div>
            <div class="status-row">
                <div class="visualizer" id="viz">
                    <div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div>
                    <div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div>
                </div>
                <div class="status" id="statusMsg">Ready</div>
            </div>

            <div class="input-bar" id="inputBar">
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
        let sessionEnded = false;
        let decodeQueue = [];
        let decoding = false;
        let tts_cancel = false; 

        // Open all links in new tabs
        document.addEventListener('click', function(e) {
            const target = e.target.closest('a');
            if (target && target.href && (target.href.startsWith('http') || target.href.startsWith('https'))) {
                target.setAttribute('target', '_blank');
                target.setAttribute('rel', 'noopener noreferrer');
            }
        });

        const chatArea = document.getElementById('chatArea');
        const emptyState = document.getElementById('emptyState');
        const toggleBtn = document.getElementById('toggleBtn');
        const sendBtn = document.getElementById('sendBtn');
        const textInput = document.getElementById('textInput');
        const statusMsg = document.getElementById('statusMsg');
        const bars = document.querySelectorAll('.bar');
        const viz = document.getElementById('viz');
        const endedOverlay = document.getElementById('endedOverlay');
        const inputBar = document.getElementById('inputBar');

        const MIC_ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="23"></line><line x1="8" y1="23" x2="16" y2="23"></line></svg>`;
        const STOP_ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect></svg>`;

        toggleBtn.onclick = () => { if(!sessionEnded) running ? stop() : start(); };
        sendBtn.onclick = () => { if(!sessionEnded) sendTextMessage(); };
        textInput.onkeypress = (e) => { if (e.key === 'Enter' && !sessionEnded) sendTextMessage(); };

        async function sendTextMessage() {
            const text = textInput.value.trim();
            if (!text || sessionEnded) return;
            
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
                ws.onclose = () => { 
                    if (!running && !sessionEnded) statusMsg.textContent = 'Ready'; 
                };
            });
        }

        let audioBuffer = [];
        const CHUNK_SIZE = 4096;

        async function start() {
            if (sessionEnded) return;
            try {
                if (!pCtx) pCtx = new (window.AudioContext || window.webkitAudioContext)();
                if (pCtx.state === 'suspended') await pCtx.resume();

                stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, sampleRate: 16000 } });
                aCtx = new AudioContext({ sampleRate: 16000 });
                
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
                    if (!running || !ws || ws.readyState !== 1 || sessionEnded) return;
                    
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
                node.connect(aCtx.destination);
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
            audioBuffer = [];
            
            if (proc) {
                 proc.disconnect();
                 proc.port.onmessage = null;
            }
            if (aCtx) aCtx.close();
            if (stream) stream.getTracks().forEach(t => t.stop());
            
            toggleBtn.innerHTML = MIC_ICON;
            toggleBtn.classList.remove('active');
            statusMsg.textContent = (sessionEnded ? 'Chat Ended' : 'Ready');
            viz.classList.remove('active');
            resetViz();
            killAudio();
            if (clearUI) {
                clearChatUI();
            }
        }
        
        window.onbeforeunload = () => { if (ws && ws.readyState === 1) ws.send(JSON.stringify({type: 'clear'})); if (ws) ws.close(); };

        function killAudio() { 
                    aQueue = []; 
                    decodeQueue = [];
                    playing = false; 
                    scheduledTime = 0;
                    decoding = false;
                    if (activeSrc) { 
                try { 
                    activeSrc.stop(); 
                    activeSrc.disconnect();
                } catch(e) {} 
                activeSrc = null; 
            }
            if (pCtx && pCtx.state === 'running') {
                scheduledTime = pCtx.currentTime;
            }
        }

        function onMsg(data) {
            switch (data.type) {
                case 'processing': 
                    statusMsg.textContent = 'Thinking'; 
                    tts_cancel = false;
                    break;
                case 'user_message': 
                    if (emptyState.style.display === 'flex' || chatArea.lastElementChild.textContent !== data.content) {
                        addMsg('user', data.content); 
                    }
                    break;
                case 'typing': showTyping(data.content); statusMsg.textContent = 'Speaking'; break;
                case 'assistant_message': doneTyping(data.content); statusMsg.textContent = (running ? 'Listening' : 'Ready'); break;
                case 'interrupt': 
                    tts_cancel = true;
                    killAudio();
                    if (typingMsg) { 
                        typingMsg.classList.add('interrupted'); 
                        doneTyping(typingMsg.textContent + " ... [Interrupted]"); 
                    }
                    setTimeout(() => { killAudio(); }, 100);
                    break;
                case 'clear': killAudio(); stop(true); statusMsg.textContent = 'Cleared'; break;
                case 'shutdown': 
                    pendingShutdown = true; 
                    if (!playing) handleShutdownUI();
                    break;
            }
        }

        function handleShutdownUI() {
            sessionEnded = true;
            statusMsg.textContent = 'Chat Ended';
            endedOverlay.style.display = 'flex';
            inputBar.style.opacity = '0.5';
            textInput.disabled = true;
            stop(false); // Stop mic if running
        }

        function resetSession() {
            sessionEnded = false;
            pendingShutdown = false;
            endedOverlay.style.display = 'none';
            inputBar.style.opacity = '1';
            textInput.disabled = false;
            clearChat(); // This will clear backend and UI
        }

        function clearChatUI() { 
            chatArea.innerHTML = ''; 
            chatArea.appendChild(emptyState); 
            emptyState.style.display = 'flex'; 
            typingMsg = null; 
        }

        async function queueAudio(buf) { 
            if (tts_cancel) {
                return;
            }
            if (decodeQueue.length > 20) {
                console.warn("Audio queue overflow, clearing old data");
                decodeQueue = [];
                killAudio();
            }
            decodeQueue.push(buf);
            processDecodeQueue();
        }

        async function processDecodeQueue() {
            if (decoding || decodeQueue.length === 0) return;
            decoding = true;
            
            while (decodeQueue.length > 0) {
                const buf = decodeQueue.shift();
                try { 
                    const bufCopy = buf.slice(0);
                    const audioBuf = await pCtx.decodeAudioData(bufCopy); 
                    aQueue.push(audioBuf); 
                    if (!playing) play(); 
                } catch (err) { 
                    console.error("Audio decode error", err); 
                }
            }
            
            decoding = false;
        }
        async function play() {
            if (aQueue.length === 0) { 
                playing = false; 
                scheduledTime = 0;
                if (pendingShutdown) {
                    setTimeout(() => handleShutdownUI(), 800);
                }
                return; 
            }
            playing = true;
            const audioBuf = aQueue.shift();
            try {
                const src = pCtx.createBufferSource();
                src.buffer = audioBuf;
                src.connect(pCtx.destination);
                activeSrc = src;
                
                const now = pCtx.currentTime;
                const startTime = Math.max(now + 0.01, scheduledTime);
                src.start(startTime);
                scheduledTime = startTime + audioBuf.duration;
                
                src.onended = () => {
                    activeSrc = null;
                    play();
                };
            } catch (e) { 
                console.error("Playback error:", e);
                activeSrc = null;
                play(); 
            }
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
    ws_lock = asyncio.Lock()
    shutdown_event = asyncio.Event()
    response_task: Optional[asyncio.Task] = None
    # dg_client = DeepgramClient()
    # session_agent = SimpleAgent(use_rag=True)
    # session_history: List[dict] = []
    # session_id = str(uuid.uuid4())
    # session_history.append({"session_id": session_id, "role": "system", "content": "Session Init"})
    dg_client = DeepgramClient()
    tts_manager = DeepgramTTSManager(dg_client)  # <-- ADD THIS LINE
    tts_manager.connect()  # <-- ADD THIS LINE (pre-connect to eliminate first delay)
    session_agent = SimpleAgent(use_rag=True)
    session_history: List[dict] = []
    session_id = str(uuid.uuid4())
    session_history.append({"session_id": session_id, "role": "system", "content": "Session Init"})
    # State for real-time STT
    transcript_buffer = []
    last_speech_time = None
    last_activity_time = asyncio.get_event_loop().time()
    
    # Initialize Deepgram Live Transcription
    dg_connection = dg_client.listen.asynclive.v("1")

    async def on_transcript(self, result, **kwargs):
        nonlocal last_speech_time
        try:
            transcript = result.channel.alternatives[0].transcript
            if not transcript:
                return

            # Update speech time
            last_speech_time = asyncio.get_event_loop().time()
            
            # Use 'user_message' to match frontend expected type
            # (REMOVED: partial transcripts are no longer sent to UI)
            # async with ws_lock:
            #     await websocket.send_json({"type": "user_message", "content": transcript})
            
            if result.is_final:
                transcript_buffer.append(transcript)
                logger.debug(f"Live Transcript (is_final): {transcript}")
        except Exception as e:
            logger.error(f"Error in on_transcript callback: {e}")

    async def on_error(self, error, **kwargs):
        logger.error(f"Deepgram Live Error: {error}")

    dg_connection.on(LiveTranscriptionEvents.Transcript, on_transcript)
    dg_connection.on(LiveTranscriptionEvents.Error, on_error)

    options = LiveOptions(
        model="nova-2",
        language="en",
        encoding="linear16",
        sample_rate=16000,
        channels=1,
        punctuate=True,
        smart_format=True,
    )
    
    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=5)
    
    try:
        await dg_connection.start(options)
        
        # Welcome message with TTS (only once)
        welcome_text = "Hi, I am Aura from BharatLogic. How can I help you today?"
        from langchain_core.messages import AIMessage
        
        # Send text to UI
        async with ws_lock:
            await websocket.send_json({"type": "assistant_message", "content": welcome_text})
        
        # Add to history
        session_agent.history.append(AIMessage(content=welcome_text))
        session_history.append({"role": "assistant", "content": welcome_text})
        
        # Send TTS audio
        for audio_chunk in tts_manager.speak(welcome_text):
            async with ws_lock:
                await websocket.send_bytes(audio_to_bytes(audio_chunk))

        while not shutdown_event.is_set():
            # Check for silence finalization with dynamic threshold and VAD awareness
            if last_speech_time and transcript_buffer:
                silence_duration = asyncio.get_event_loop().time() - last_speech_time
                
                # Determine threshold: shorter if user finished a punctuation-ended sentence
                current_text = " ".join(transcript_buffer).strip()
                threshold = 1 if current_text.endswith(('.', '?', '!',',')) else 1.5
                
                # Only finalize if silence threshold is met AND VAD confirms user isn't speaking
                if silence_duration >= threshold and not vad.is_speaking:
                    final_text = current_text
                    transcript_buffer.clear()
                    last_speech_time = None
                    
                    if final_text:
                        logger.info(f"Finalized speech: {final_text}")
                        if response_task and not response_task.done():
                            response_task.cancel()
                        
                        # Send finalized message to UI exactly once
                        async with ws_lock:
                            await websocket.send_json({"type": "user_message", "content": final_text})

                        # call: process_chat_common(final_text, session_agent, session_history, dg_client, stop_flag, suppress_tts=False)
                        response_task = asyncio.create_task(handle_inference(websocket, session_agent, session_history, dg_client, executor, tts_manager, process_chat_common, final_text, ws_lock=ws_lock, shutdown_event=shutdown_event, suppress_tts=False))
            # Receive message with timeout (0.1s for silence check, 5s for keep-alive)
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=0.1)
                last_activity_time = asyncio.get_event_loop().time()
            except asyncio.TimeoutError:
                # Send KeepAlive JSON every 5s of silence to keep Deepgram connection alive
                if asyncio.get_event_loop().time() - last_activity_time > 5.0:
                    try:
                        # await dg_connection.send(b'\x00' * 320)
                        import json
                        await dg_connection.send(json.dumps({"type": "KeepAlive"}))
                        last_activity_time = asyncio.get_event_loop().time()
                    except: pass
                continue

            if message["type"] == "websocket.disconnect":
                logger.info("Websocket message: Disconnect received")
                break
                
            if "bytes" in message:
                audio_bytes = message["bytes"]
                # 1. Feed to Deepgram for STT
                try:
                    await dg_connection.send(audio_bytes)
                except Exception as e:
                    logger.error(f"Error sending audio to Deepgram: {e}")
                
                # 2. Feed to Silero VAD for robust interruption detection
                audio_chunk = np.frombuffer(audio_bytes, dtype=np.int16)
                vad_events = vad.process(audio_chunk)
                for event_type, _ in vad_events:
                    if event_type == "start_speech":
                        # Human speech detected - trigger immediate interrupt
                        await websocket.send_json({"type": "interrupt"})
                        if response_task and not response_task.done():
                            logger.info("In-progress response task cancelled via Silero VAD interrupt")
                            response_task.cancel()
            
            elif "text" in message:
                import json
                try:
                    payload = json.loads(message["text"])
                    if payload.get("type") == "chat":
                        content = payload.get("content")
                        logger.info(f"Received text chat: {content}")
                        if response_task and not response_task.done():
                            response_task.cancel()
                        response_task = asyncio.create_task(handle_inference(websocket, session_agent, session_history, dg_client, executor, tts_manager, process_chat_common, content, ws_lock=ws_lock, shutdown_event=shutdown_event, suppress_tts=True))
                    elif payload.get("type") == "interrupt":
                        tts_manager.cancel()
                        if response_task and not response_task.done():
                            logger.info("Manual interrupt received, cancelling task")
                            response_task.cancel()
                            
                    elif payload.get("type") == "clear":
                        logger.warning("Backend memory reset confirmed by UI command")
                        try:
                            if len(session_agent.history) > 1:
                                summary_data = session_agent.generate_summary()
                                save_conversation_summary(session_id, summary_data, session_history)
                        except Exception as e:
                            logger.error(f"Error saving summary on UI clear: {e}")
                            
                        session_history.clear()
                        session_agent.clear_history()
                        async with ws_lock:
                            await websocket.send_json({"type": "clear"})

                        # Re-initialize session
                        session_id = str(uuid.uuid4())
                        session_history.append({"session_id": session_id, "role": "system", "content": "Session Init"})
                        
                        # Welcome message (text only, no TTS on clear to avoid double)
                        welcome_text = "Hi, I am Aura from BharatLogic. How can I help you today?"
                        from langchain_core.messages import AIMessage
                        session_agent.history.append(AIMessage(content=welcome_text))
                        session_history.append({"role": "assistant", "content": welcome_text})
                        async with ws_lock:
                            await websocket.send_json({"type": "assistant_message", "content": welcome_text})
                    elif payload.get("type") == "shutdown":
                        logger.warning("Backend received shutdown signal")
                        async with ws_lock:
                            await websocket.send_json({"type": "shutdown"})
                        shutdown_event.set()
                        break 
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
        if executor:
            executor.shutdown(wait=False)
            logger.info("Session thread pool executor shut down")

        tts_manager.close()

        if response_task: 
            logger.info("Cleaning up session: cancelling active response task")
            response_task.cancel()
        
        try:
            await dg_connection.finish()
            logger.info("Deepgram Live connection finished")
        except:
            pass
        
        # Save summary before clearing history (on any disconnect/crash)
        try:
            if len(session_agent.history) > 1:  # More than just system message
                logger.info("Generating summary before session cleanup...")
                summary_data = session_agent.generate_summary()
                save_conversation_summary(session_id, summary_data, session_history)
        except Exception as e:
            logger.error(f"Error saving summary on disconnect: {e}")
        
        session_history.clear()
        session_agent.clear_history()
        logger.info("Session state cleared")

# async def handle_inference(websocket: WebSocket, session_agent: SimpleAgent, session_history: List[dict], dg_client: DeepgramClient, executor, func, *args, ws_lock: asyncio.Lock = None, shutdown_event: asyncio.Event = None, suppress_tts: bool = False):
async def handle_inference(websocket: WebSocket, session_agent: SimpleAgent, session_history: List[dict], dg_client: DeepgramClient, executor, tts_manager: DeepgramTTSManager, func, *args, ws_lock: asyncio.Lock = None, shutdown_event: asyncio.Event = None, suppress_tts: bool = False): 
    """Handles the async-sync inference pipeline with comprehensive error handling."""
    stop_flag = threading.Event()
    try:
        async with ws_lock:
            await websocket.send_json({"type": "processing"})
        typing_count = 0
        # Signature: (text, session_agent, session_history, dg_client, executor, stop_flag=None, suppress_tts=False)
        async for event_type, content in run_sync_gen_in_thread(
            func, *args, session_agent, session_history, dg_client, executor, tts_manager=tts_manager, stop_flag=stop_flag, suppress_tts=suppress_tts
        ):
            try:
                if event_type == "chat_user": 
                    async with ws_lock:
                        await websocket.send_json({"type": "user_message", "content": content})
                elif event_type == "typing": 
                    typing_count += 1
                    async with ws_lock:
                        await websocket.send_json({"type": "typing", "content": content})
                elif event_type == "clear": 
                    async with ws_lock:
                        await websocket.send_json({"type": "clear"})
                elif event_type == "shutdown": 
                    logger.info("Shutdown detected in inference stream")
                    async with ws_lock:
                        await websocket.send_json({"type": "shutdown"})
                    if shutdown_event:
                        shutdown_event.set()
                elif event_type == "audio":
                    sr, arr = content
                    # logger.debug(f"Sending audio event (SR={sr}, bytes={len(arr.tobytes())})")
                    async with ws_lock:
                        await websocket.send_bytes(audio_to_bytes((sr, arr)))
                elif event_type == "chat_assistant": 
                    logger.info(f"Final response sent. Typing events emitted: {typing_count}")
                    async with ws_lock:
                        await websocket.send_json({"type": "assistant_message", "content": content})
                elif event_type == "error": 
                    async with ws_lock:
                        await websocket.send_json({"type": "error", "message": content})
            except Exception as send_error:
                logger.error(f"Error sending {event_type}: {send_error}")
                break
    except asyncio.CancelledError: 
        logger.info("Response task cancelled, signalling background thread")
        stop_flag.set()
    except Exception as e: 
        logger.error(f"Inference pipeline failure: {e}")
        try:
            async with ws_lock:
                await websocket.send_json({"type": "error", "message": "An error occurred processing your request."})
        except:
            pass  

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)