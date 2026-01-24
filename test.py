import asyncio
import uuid
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

from session_summary import generate_session_summary
from db_writer import save_conversation_summary

load_dotenv()

import logging
from loguru import logger

logging.getLogger("pymongo").handlers = []
logging.getLogger("pymongo").propagate = False
logging.getLogger("pymongo").setLevel(logging.WARNING)

# 🔹 Separate LLM for summary (CRITICAL FIX)
summary_llm = openai.LLM(model="gpt-4o-mini")


class SessionState:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.conversation_log = []
        self.summary_saved = False
        self.finalize_event = asyncio.Event()  # Track when finalize completes


try:
    from rag.retriever import get_relevant_context
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
async def end_conversation(_context: RunContext):
    return {"status": "ended"}


SYSTEM_PROMPT = """
You are Aura, an AI assistant for BharatLogic.
CRITICAL RULE FOR USING search_company_info:
- When the user asks ANY question about BharatLogic (services, pricing, technologies, team, etc.)
- You MUST call search_company_info with the user's EXACT WORDS
- Do NOT rephrase or summarize - use their exact sentence
- Example: If user says "Tell me about your AI services", call search_company_info(user_query="Tell me about your AI services")
Keep responses to 1-2 sentences.

IMPORTANT: When the user says any farewell phrase like "bye", "goodbye", "see you", 
"take care", "have a great day", etc., say a warm goodbye message and then call 
the 'end_conversation' tool.
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

    # Register shutdown callback - this BLOCKS job exit until complete
    @ctx.add_shutdown_callback
    async def on_shutdown():
        logger.info("Shutdown callback triggered, finalizing session...")
        await finalize_session(state)
        logger.info("Shutdown callback complete.")

    agent = Agent(
        instructions=SYSTEM_PROMPT,
        tools=[search_company_info, end_conversation],
    )

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(model="nova-2"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=deepgram.TTS(model="aura-asteria-en"),
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
