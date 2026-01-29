from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

mongo_client: MongoClient | None = None
_db = None


def connect_db():
    global mongo_client, _db
    mongo_client = MongoClient(MONGO_URI)
    _db = mongo_client["voice_bot_db"]
    print("✅ MongoDB connected")


def get_db():
    """Get the database instance"""
    return _db


def close_db():
    global mongo_client
    if mongo_client:
        mongo_client.close()
        print("🛑 MongoDB connection closed")
