from pymongo import MongoClient
from datetime import datetime
import os
from loguru import logger

_mongo_client: MongoClient | None = None
_db = None

def get_db():
    """Get or initialize the MongoDB database instance."""
    global _mongo_client, _db
    if _db is None:
        mongo_uri = os.getenv("MONGO_URI")
        if not mongo_uri:
            logger.error("MONGO_URI not found in environment variables")
            return None
        
        try:
            _mongo_client = MongoClient(mongo_uri)
            _db = _mongo_client["voice_bot_db"]
            logger.info("✅ MongoDB connected")
        except Exception as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            return None
    return _db

def connect_db():
    """Explicitly connect to the database (called during app startup)."""
    return get_db()

def close_db():
    """Close the database connection."""
    global _mongo_client, _db
    if _mongo_client:
        _mongo_client.close()
        _mongo_client = None
        _db = None
        logger.info("🛑 MongoDB connection closed")

def save_conversation_summary(
    session_id: str,
    summary_data: dict,
    conversation_log: list
):
    """Save conversation summary and logs to MongoDB."""
    db = get_db()
    if db is None:
        logger.error("Database not available. Skipping save.")
        return

    try:
        conversations = db["conversations"]
        chat_logs = db["chat_logs"]

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
        logger.info(f"Successfully saved session data for: {session_id}")
    except Exception as e:
        logger.error(f"Failed to save session data: {e}")