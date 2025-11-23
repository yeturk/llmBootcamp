__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import DirectoryLoader
from uuid import uuid4, os
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model     = "text-embedding-004",
    task_type = "RETRIEVAL_DOCUMENT"
)

url = "http://qdrant:6333"
#url = "http://localhost:6333"


# Returns QdrantVectorStore object + ingesting documents
def ingest_from_docs(upload_dir: str = "/tmp/uploads"):
    # Read pdf content
    UPLOAD_DIR = upload_dir
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    raw_documents = DirectoryLoader(path=UPLOAD_DIR).load()

    # Split raw pdf content into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size    = 1500,
        chunk_overlap = 200
    )

    docs = text_splitter.split_documents(raw_documents)

    QdrantVectorStore.from_documents(
        docs,
        embeddings,
        url             = url,
        prefer_grpc     = True,
        collection_name = "vbo-de-bootcamp",
    )

    print("Ingested docs")


def get_retreiver():
    vector_store = QdrantVectorStore.from_existing_collection(
        collection_name = "vbo-de-bootcamp",
        embedding       = embeddings,
        url             = url,
        prefer_grpc     = True
    )

    bootcamp_retriever = vector_store.as_retriever(
        search_type   = "mmr", 
        search_kwargs = {"k": 3, "fetch_k": 10} 
    )

    # Search type = MMR (Maximal Marginal Relevance)
    # MMR = hem en alakalı hem de birbirinden çeşitli sonuçlar getirir.
        # k=3 → 3 adet belge dönecek
        # fetch_k=10 → önce 10 belge retrieve edilir sonra MMR uygulanır
    
    return bootcamp_retriever