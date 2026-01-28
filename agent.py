import asyncio
import uuid
import os
from dotenv import load_dotenv
from loguru import logger

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    room_io,
)
from livekit.plugins import openai, deepgram, silero

from src.session_summary import generate_session_summary
from src.db import save_conversation_summary

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
STT_MODEL = os.getenv("STT_MODEL", "nova-2")
TTS_MODEL = os.getenv("TTS_MODEL", "aura-asteria-en")

summary_llm = openai.LLM(model=LLM_MODEL)

class SessionState:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.conversation_log = []
        self.summary_saved = False
        self.finalize_event = asyncio.Event() 

try:
    from src.rag.retriever import get_relevant_context
except ImportError:
    def get_relevant_context(q, k=3):
        return None

@function_tool
async def search_company_info(_context: RunContext, user_query: str):
    result = get_relevant_context(user_query, k=5)
    if result:
        return {"context": result, "found": True}
    return {"context": "No specific info found.", "found": False}

@function_tool
async def control_website(ctx: RunContext, action: str, target: str):
    """
    Controls the website UI.
    Actions: 
    - 'navigate': To change pages (target is the URL or path like '/portfolio')
    - 'scroll': To scroll to a section (target is a CSS selector or section name)
    - 'highlight': To highlight a specific element (target is a CSS selector)
    """
    import json
    logger.info(f"Website Control Tool Called: {action} -> {target}")
    
    try:
        room = getattr(ctx, 'room', None)
        if room is None:
            room = ctx.session.room_io.room

        if not room or not room.local_participant:
            logger.error("Room or LocalParticipant not available in tool context")
            return {"error": "Room not connected", "status": "failed"}

        payload = {
            "type": "WEBSITE_CONTROL",
            "action": action,
            "target": target
        }
        
        await room.local_participant.publish_data(
            json.dumps(payload),
            reliable=True
        )
        
        logger.info(f"SUCCESS: Website Control command sent: {action}")
        return {"status": "success", "message": f"Command '{action}' for '{target}' sent to website."}
    
    except Exception as e:
        logger.error(f"FAILED to send website control command: {e}")
        return {"error": str(e), "status": "failed"}

@function_tool
async def end_conversation(_context: RunContext):
    return {"status": "ended"}

SYSTEM_PROMPT = """
You are Aura, an AI assistant for BharatLogic.
CRITICAL RULE FOR USING search_company_info:
- When the user asks ANY question about BharatLogic (services, pricing, technologies, team, etc.)
- You MUST call search_company_info with the user's EXACT WORDS.
- Keep responses to 1-2 sentences.

WEBSITE CONTROL RULES:
- If the user wants to see 'Portfolio', call control_website(action='navigate', target='portfolio.html')
- If the user wants to see 'Services', call control_website(action='navigate', target='services.html')
- If the user wants to see 'Contact', call control_website(action='navigate', target='contact.html')
- If the user wants to go 'Home', call control_website(action='navigate', target='index.html')

SCROLLING, HIGHLIGHTING & CLICKING (on Services page):
- If the user asks about AI or NLP, call control_website(action='highlight', target='#ai-services')
- If they ask to start the AI demo, call control_website(action='click', target='#ai-demo-btn')
- If they ask about Cloud health or status, call control_website(action='click', target='#cloud-check-btn')
- If they ask about mobile apps, call control_website(action='highlight', target='#mobile-apps')
- If they want to see a general section, use 'scroll' with the section ID.

IMPORTANT: When the user says any farewell phrase like "bye", say goodbye and call the 'end_conversation' tool.
"""

async def finalize_session(state: SessionState):
    if state.summary_saved:
        state.finalize_event.set()
        return

    if len(state.conversation_log) < 2:
        state.finalize_event.set()
        return

    logger.info("Generating session summary...")

    try:
        summary_json = await generate_session_summary(
            summary_llm,
            state.conversation_log
        )

        save_conversation_summary(
            session_id=state.session_id,
            summary_data=summary_json,
            conversation_log=state.conversation_log
        )

        state.summary_saved = True
        logger.info(f"Session summary saved: {state.session_id}")
    except Exception as e:
        logger.error(f"Failed to save summary: {e}")
    finally:
        state.finalize_event.set()


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    logger.info(f"Connected to room: {ctx.room.name}")

    state = SessionState()

    @ctx.add_shutdown_callback
    async def on_shutdown():
        logger.info("Shutdown callback triggered, finalizing session...")
        await finalize_session(state)
        logger.info("Shutdown callback complete.")

    agent = Agent(
        instructions=SYSTEM_PROMPT,
        tools=[search_company_info, end_conversation, control_website],
    )

    session = AgentSession(
        vad=silero.VAD.load(
            min_silence_duration=0.1,
            activation_threshold=0.5,
            prefix_padding_duration=0.1
        ),
        stt=deepgram.STT(model=STT_MODEL),
        llm=openai.LLM(model=LLM_MODEL),
        tts=deepgram.TTS(model=TTS_MODEL),
    )

    @session.on("conversation_item_added")
    def on_item(event):
        role = event.item.role
        content = event.item.content

        if isinstance(content, list) and content:
            text = str(content[0])
        elif isinstance(content, str):
            text = content
        else:
            return

        state.conversation_log.append({
            "role": role,
            "text": text
        })

    @session.on("function_calls_finished")
    def on_function_done(event):
        for call in event.function_calls:
            if call.name == "end_conversation":

                async def close_flow():
                    await finalize_session(state)
                    await ctx.room.disconnect()

                asyncio.create_task(close_flow())

    await session.start(
        agent=agent,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=True,
            audio_output=True,
            text_output=True
        )
    )

    await session.generate_reply(
        instructions="Say exactly: Hello! I am Aura from BharatLogic. How can I help you today?"
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(entrypoint_fnc=entrypoint)
    )