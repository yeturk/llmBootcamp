from fastapi import FastAPI
from pydantic import BaseModel
from embeddings import get_embedding
from vector_qdrant import client, init_qdrant, COLLECTION_NAME
from qdrant_client.models import PointStruct

app = FastAPI()

init_qdrant()

class Document(BaseModel):
    id: int
    text: str

class SearchQuery(BaseModel):
    query: str

@app.post("/add_document")
def add_document(doc: Document):
    embedding = get_embedding(doc.text)

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[PointStruct(id=doc.id, vector=embedding, payload={"text": doc.text})],
    )
    return {"status": "ok"}

@app.post("/search")
def search_docs(search: SearchQuery):
    query_vec = get_embedding(search.query)

    results = client.search(
        collection_name=COLLECTION_NAME, 
        query_vector=query_vec, 
        limit=5
    )
    return {
        "results": [
            {"id": r.id, "score": r.score, "text": r.payload.get("text")}
            for r in results
        ]
    }
