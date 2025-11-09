import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection

# Load environment variables if present
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "appdb")

client = MongoClient(DATABASE_URL)
db = client[DATABASE_NAME]


def _get_collection(name: str) -> Collection:
    return db[name]


def create_document(collection_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Insert a document with automatic timestamps and return inserted doc."""
    now = datetime.utcnow()
    payload = {**data, "created_at": now, "updated_at": now}
    col = _get_collection(collection_name)
    result = col.insert_one(payload)
    inserted = col.find_one({"_id": result.inserted_id})
    # Convert ObjectId to str for consistency
    if inserted and "_id" in inserted:
        inserted["id"] = str(inserted.pop("_id"))
    return inserted or {}


def get_documents(
    collection_name: str, filter_dict: Optional[Dict[str, Any]] = None, limit: int = 50
) -> List[Dict[str, Any]]:
    """Query documents from a collection."""
    col = _get_collection(collection_name)
    cursor = col.find(filter_dict or {}).limit(limit)
    docs: List[Dict[str, Any]] = []
    for d in cursor:
        d["id"] = str(d.pop("_id"))
        docs.append(d)
    return docs
