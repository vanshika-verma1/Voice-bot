import json
from livekit.agents.llm.chat_context import ChatContext


SUMMARY_SYSTEM_PROMPT = """
You are a conversation summarizer.

Return STRICT JSON only:

{
  "name": "",
  "phone": "",
  "email": "",
  "summary": ""
}

Rules:
- Extract name / phone / email ONLY if user explicitly said them
- summary must be 2–3 concise lines
- If data is missing, use empty string
- DO NOT explain
- DO NOT add markdown
""".strip()


async def generate_session_summary(llm, conversation_log: list) -> dict:
    # 1️⃣ Create chat context
    chat_ctx = ChatContext()

    # 2️⃣ Add system prompt (CORRECT)
    chat_ctx.add_message(
        role="system",
        content=[SUMMARY_SYSTEM_PROMPT],
    )

    # 3️⃣ Add conversation messages (CORRECT)
    for item in conversation_log:
        chat_ctx.add_message(
            role=item["role"],
            content=[item["text"]],
        )

    # 4️⃣ Call LLM
    stream = llm.chat(chat_ctx=chat_ctx)

    # 5️⃣ Collect streamed response
    text = ""
    async with stream:
        async for chunk in stream:
            if chunk.delta and chunk.delta.content:
                text += chunk.delta.content

    text = text.strip()
    print(text)
    # 6️⃣ Parse JSON safely
    try:
        return json.loads(text)
    except Exception:
        return {
            "name": "",
            "phone": "",
            "email": "",
            "summary": text,
        }
