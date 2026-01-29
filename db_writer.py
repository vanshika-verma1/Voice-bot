from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

client = None
db = None
conversations = None
chat_logs = None

def _get_db():
    global client, db, conversations, chat_logs
    if client is None:
        client = MongoClient(MONGO_URI)
        db = client["voice_bot_db"]
        conversations = db["conversations"]
        chat_logs = db["chat_logs"]
    return conversations, chat_logs

def save_conversation_summary(
    session_id: str,
    summary_data: dict,
    conversation_log: list
):
    try:
        convos, logs = _get_db()
        
        convos.insert_one({
            "session_id": session_id,
            "timestamp": datetime.utcnow(),
            "name": summary_data.get("name", ""),
            "phone": summary_data.get("phone", ""),
            "email": summary_data.get("email", ""),
            "summary": summary_data.get("summary", "")
        })

        logs.insert_one({
            "session_id": session_id,
            "timestamp": datetime.utcnow(),
            "chat_history": conversation_log
        })
        
        logger.info(f"Saved conversation: {session_id}")
    except Exception as e:
        logger.error(f"Failed to save conversation: {e}")
