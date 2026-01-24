from pymongo import MongoClient
from datetime import datetime
import os

MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client["voice_bot_db"]

conversations = db["conversations"]
chat_logs = db["chat_logs"]

def save_conversation_summary(
    session_id: str,
    summary_data: dict,
    conversation_log: list
):
    conversations.insert_one({
        "session_id": session_id,
        "timestamp": datetime.utcnow(),
        "name": summary_data.get("name", ""),
        "phone": summary_data.get("phone", ""),
        "email": summary_data.get("email", ""),
        "summary": summary_data.get("summary", "")
    })

    chat_logs.insert_one({
        "session_id": session_id,
        "timestamp": datetime.utcnow(),
        "chat_history": conversation_log
    })
