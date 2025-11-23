from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from embeddings import get_embedding

client = QdrantClient(host="qdrant", port=6333)
COLLECTION_NAME = "documents"

def init_qdrant():
    """
    Eğer koleksiyon yoksa oluştur.
    Gemini text-embedding-004 boyutu -> 768
    """
    collections = client.get_collections().collections
    if not any(c.name == COLLECTION_NAME for c in collections):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
