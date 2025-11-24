import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

# History-aware RAG
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# -----------------------------
# Global Settings
# -----------------------------
PDF_DIR = "app/backend/pdfs"
UPLOAD_DIR = "app/backend/uploads"
COLLECTION_NAME = "gtu-staj1"

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chat history store
store = {}

# -----------------------------
# Utility Functions
# -----------------------------

def get_vectorstore():
    """Qdrant'a bağlanır ve vector store döner."""
    client = QdrantClient(host="qdrant", port=6333)
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    vector_store = QdrantVectorStore.from_existing_collection(
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        url="http://qdrant:6333",
    )
    return vector_store


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Session için chat history döner."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def ingest_file(file_path: str):
    """Tek bir dosyayı ingest eder (PDF veya TXT)."""
    print(f"📄 Loading: {file_path}")
    
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        print("❌ Unsupported file type")
        return

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    vector_store = get_vectorstore()
    print(f"📌 Adding {len(chunks)} chunks to Qdrant...")
    vector_store.add_documents(documents=chunks)
    print("✅ File ingested successfully.")


def ingest_pdfs_in_folder():
    """PDF klasöründeki tüm PDF'leri ingest eder."""
    print("🔍 Initial PDF ingestion started...")

    if not os.path.exists(PDF_DIR):
        print("⚠️ PDFs directory not found:", PDF_DIR)
        return

    filenames = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]

    if len(filenames) == 0:
        print("⚠️ No PDF found in:", PDF_DIR)
        return

    for filename in filenames:
        path = f"{PDF_DIR}/{filename}"
        ingest_file(path)

    print("✅ Initial ingestion completed.")


def ensure_collection_exists():
    """Collection yoksa oluşturur."""
    client = QdrantClient(host="qdrant", port=6333)
    collections = client.get_collections().collections
    names = [c.name for c in collections]

    if COLLECTION_NAME not in names:
        print(f"📌 Collection `{COLLECTION_NAME}` not found. Creating it...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"size": 768, "distance": "Cosine"}
        )
        print("✅ Collection created.")
    else:
        print(f"✔ Collection `{COLLECTION_NAME}` already exists.")


# -----------------------------
# RAG Chain Setup
# -----------------------------

def get_rag_chain():
    """History-aware RAG chain döner."""
    vector_store = get_vectorstore()
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})

    llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

    # Contextualize question prompt
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Sohbet geçmişi ve kullanıcının son sorusuna bakarak, "
         "soruyu bağımsız hale getir. Cevap VERME, sadece soruyu yeniden formüle et."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # QA prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Sen GTÜ Bilgisayar Mühendisliği Staj Süreçleri konusunda uzman bir asistansın. "
         "Kullanıcıya sadece verilen PDF bilgilerine dayanarak yardımcı ol. "
         "Eğer bilmiyorsan, bilmediğini söyle.\n\nContext: {context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    qa_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return qa_chain


# -----------------------------
# FastAPI STARTUP
# -----------------------------
@app.on_event("startup")
def startup_event():
    print("🚀 FastAPI starting...")
    ensure_collection_exists()
    ingest_pdfs_in_folder()


# -----------------------------
# API ENDPOINTS
# -----------------------------

@app.get("/")
def home():
    return {"message": "GTU Staj Chatbot Backend is running!"}


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


@app.post("/chat")
async def chat_api(request: ChatRequest):
    """Streaming chat response with history."""
    qa_chain = get_rag_chain()

    def generate():
        try:
            response = qa_chain.invoke(
                {"input": request.message},
                config={"configurable": {"session_id": request.session_id}}
            )
            # Stream karakter karakter
            answer = response.get("answer", "")
            for char in answer:
                yield char
        except Exception as e:
            yield f"❌ Hata: {str(e)}"

    return StreamingResponse(generate(), media_type="text/plain")


@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    """Kullanıcının yüklediği PDF/TXT dosyasını ingest eder."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    save_path = f"{UPLOAD_DIR}/{file.filename}"
    
    with open(save_path, "wb") as f:
        f.write(await file.read())
    
    # Dosyayı ingest et
    ingest_file(save_path)
    
    return {"status": "success", "filename": file.filename}