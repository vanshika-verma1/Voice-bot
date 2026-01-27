from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

mongo_client: MongoClient | None = None
db = None

def connect_db():
    global mongo_client, db
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client["voice_bot_db"]
    print("✅ MongoDB connected")

def close_db():
    global mongo_client
    if mongo_client:
        mongo_client.close()
        print("🛑 MongoDB connection closed")