import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# -----------------------------
# Global Settings
# -----------------------------
PDF_DIR = "app/backend/pdfs"
COLLECTION_NAME = "gtu-staj1"

app = FastAPI()

# Allow Streamlit to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Utility Functions
# -----------------------------

def get_vectorstore():
    """Qdrant bağlanır ve hazır vector store döner."""
    client       = QdrantClient(host="qdrant", port=6333)
    embeddings   = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    vector_store = Qdrant(
        client          = client,
        collection_name = COLLECTION_NAME,
        embeddings      = embeddings,
    )
    return vector_store


def ingest_pdfs_in_folder():
    """PDF klasöründeki tüm PDF'leri ingest eder."""
    print("🔍 Initial PDF ingestion started...")

    if not os.path.exists(PDF_DIR):
        print("⚠ PDFs directory not found:", PDF_DIR)
        return

    filenames = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]

    if len(filenames) == 0:
        print("⚠ No PDF found in:", PDF_DIR)
        return

    vector_store = get_vectorstore()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = 1500,
        chunk_overlap = 200
    )

    for filename in filenames:
        path = f"{PDF_DIR}/{filename}"
        print(f"📄 Loading: {path}")

        loader = PyPDFLoader(path)
        docs = loader.load()

        chunks = splitter.split_documents(docs)

        print(f"📌 Adding {len(chunks)} chunks to Qdrant...")
        vector_store.add_documents(documents=chunks)

    print("✅ Initial ingestion completed.")


def ensure_collection_exists():
    client = QdrantClient(host="qdrant", port=6333)

    collections = client.get_collections().collections
    names = [c.name for c in collections]

    if COLLECTION_NAME not in names:
        print(f"📌 Collection `{COLLECTION_NAME}` not found. Creating it...")

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "size": 768,
                "distance": "Cosine"
            }
        )
        print("✅ Collection created.")
    else:
        print(f"✔ Collection `{COLLECTION_NAME}` already exists.")


# -----------------------------
# FastAPI STARTUP
# -----------------------------
@app.on_event("startup")
def startup_event():
    print("🚀 FastAPI starting... Running initial ingestion.")
    ensure_collection_exists()
    ingest_pdfs_in_folder()


# -----------------------------
# API ENDPOINTS
# -----------------------------
@app.get("/")
def home():
    return {"message": "GTU Staj Chatbot Backend is running!"}


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Kullanıcının yüklediği PDF dosyasını ingest eder."""
    save_path = f"{PDF_DIR}/{file.filename}"

    # PDF klasörü yoksa oluştur
    os.makedirs(PDF_DIR, exist_ok=True)

    with open(save_path, "wb") as f:
        f.write(await file.read())

    ingest_pdfs_in_folder()  # YENİ PDF’İ ingest et

    return {"status": "success", "filename": file.filename}


@app.post("/chat")
async def chat_api(query: str):
    """Kullanıcı sorusu → Retrieval → Gemini → Cevap"""
    vector_store = get_vectorstore()

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Sen GTÜ Bilgisayar Mühendisliği Staj Süreçleri konusunda uzman bir asistansın. "
         "Kullanıcıya sadece PDF içindeki bilgilere dayanarak cevap ver."),
        ("human", "Soru: {query}\n\nİlgili Bilgiler:\n{context}\n\nCevap:"),
    ])

    chain = prompt | model
    response = chain.invoke({"query": query, "context": context})

    return {"answer": response.content}
