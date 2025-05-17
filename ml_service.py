# ml_service/ml_service.py
from flask import Flask, jsonify
#from ml_model import get_similar_docs
from ml_model import get_similar_docs



from bson import ObjectId
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from flask_cors import CORS

load_dotenv()
app = Flask(__name__)
CORS(app)

client = MongoClient(os.getenv("MONGO_URI"))
db = client["docdb"]

@app.route("/recommend/<doc_id>")
def recommend(doc_id):
    try:
        results = get_similar_docs(doc_id, db)
        return jsonify(results)
    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
