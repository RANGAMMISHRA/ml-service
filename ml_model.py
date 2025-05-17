# ml_service/ml_model.py
from sentence_transformers import SentenceTransformer, util
from pymongo import MongoClient
from bson import ObjectId
import os

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_db():
    client = MongoClient(os.getenv("MONGO_URI"))
    return client["docdb"]

def get_embeddings(texts):
    return model.encode(texts, convert_to_tensor=True)

def get_similar_docs(doc_id, db=None):
    db = db or get_db()
    target_doc = db.documents.find_one({"_id": ObjectId(doc_id)})
    if not target_doc or not target_doc.get("content"):
        return []

    target_embedding = get_embeddings([target_doc["content"]])[0]
    docs = list(db.documents.find({"_id": {"$ne": ObjectId(doc_id)}, "content": {"$ne": ""}}))
    corpus = [doc["content"] for doc in docs]
    if not corpus:
        return []

    embeddings = get_embeddings(corpus)
    similarities = util.pytorch_cos_sim(target_embedding, embeddings)[0]

    results = []
    for i, score in enumerate(similarities):
        results.append({
            "filename": docs[i]["filename"],
            "s3_url": docs[i]["s3_url"],
            "tags": docs[i].get("tags", []),
            "score": round(score.item(), 2)
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:5]
