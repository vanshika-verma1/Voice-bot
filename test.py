import asyncio
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
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

try:
    from rag.retriever import get_relevant_context
except ImportError:
    def get_relevant_context(q, k=3):
        return None


@function_tool
async def search_company_info(_context: RunContext, query: str):
    """Search BharatLogic's knowledge base. Pass user's FULL sentence."""
    logger.info(f"🔍 RAG Query: '{query}'")
    result = get_relevant_context(query, k=3)
    if result:
        return {"context": result, "found": True}
    return {"context": "No specific info found.", "found": False}


@function_tool
async def end_conversation(_context: RunContext):
    """
    End the conversation. Call this when the user says goodbye, bye, 
    see you, take care, have a great day, or any farewell phrase.
    """
    logger.info("👋 Ending conversation...")
    return {"status": "ended", "message": "Conversation ended by user request."}


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


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    logger.info(f"🚀 Connected to room: {ctx.room.name}")

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
        
        if content:
            # Handle content that might be a list
            if isinstance(content, list):
                text = content[0] if content else ""
            else:
                text = str(content)
            
            if text:
                logger.info(f"{'👤 USER' if role == 'user' else '🤖 AGENT'}: {text}")

    @session.on("function_calls_finished")
    def on_function_done(event):
        for call in event.function_calls:
            if call.name == "end_conversation":
                logger.info("📋 Session ending due to farewell...")
                # Give time for final message to play, then disconnect
                asyncio.create_task(delayed_disconnect(ctx, 3))

    async def delayed_disconnect(ctx, delay):
        await asyncio.sleep(delay)
        logger.info("👋 Disconnecting from room...")
        await ctx.room.disconnect()

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
        instructions="Say: Hello! I am Aura from BharatLogic. How can I help you today?"
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))