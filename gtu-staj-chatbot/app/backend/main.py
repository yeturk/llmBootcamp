import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import logging

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

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def get_qdrant_client():
    """Qdrant client döner."""
    return QdrantClient(host="qdrant", port=6333)


def get_embeddings():
    """Embeddings döner."""
    return GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        task_type="retrieval_document"
    )


def get_vectorstore():
    """Qdrant'a bağlanır ve vector store döner."""
    try:
        client = get_qdrant_client()
        embeddings = get_embeddings()
        
        vector_store = QdrantVectorStore.from_existing_collection(
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
            url="http://qdrant:6333",
            prefer_grpc=False,
        )
        logger.info("✅ Vector store connected successfully")
        return vector_store
    except Exception as e:
        logger.error(f"❌ Vector store connection error: {e}")
        raise


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Session için chat history döner."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
        logger.info(f"📝 New session created: {session_id}")
    return store[session_id]


def ingest_file(file_path: str):
    """Tek bir dosyayı ingest eder (PDF veya TXT)."""
    logger.info(f"📄 Loading file: {file_path}")
    
    try:
        # Dosya yükleme
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            logger.error("❌ Unsupported file type")
            return False

        docs = loader.load()
        logger.info(f"📖 Loaded {len(docs)} pages/documents")

        # Chunk size - daha iyi context
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Daha küçük chunk = daha iyi retrieval
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        logger.info(f"✂️ Split into {len(chunks)} chunks")

        # Metadata ekle - ÖNEMLİ: Dosya adını metadata'ya ekle
        for i, chunk in enumerate(chunks):
            chunk.metadata["source_file"] = os.path.basename(file_path)
            chunk.metadata["chunk_id"] = i
            # İçeriğin preview'ını da ekle
            chunk.metadata["preview"] = chunk.page_content[:100]

        # Qdrant'a ekle
        client = get_qdrant_client()
        embeddings = get_embeddings()
        
        QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            url="http://qdrant:6333",
            prefer_grpc=False,
            collection_name=COLLECTION_NAME,
        )
        
        logger.info(f"✅ Successfully ingested {len(chunks)} chunks from {file_path}")
        logger.info(f"📊 Sample chunk: {chunks[0].page_content[:200]}...")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error ingesting file {file_path}: {e}")
        return False


def ingest_pdfs_in_folder():
    """PDF klasöründeki tüm PDF'leri ingest eder."""
    logger.info("🔍 Initial PDF ingestion started...")

    if not os.path.exists(PDF_DIR):
        logger.warning(f"⚠️ PDFs directory not found: {PDF_DIR}")
        return

    filenames = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]

    if len(filenames) == 0:
        logger.warning(f"⚠️ No PDF found in: {PDF_DIR}")
        return

    success_count = 0
    for filename in filenames:
        path = f"{PDF_DIR}/{filename}"
        if ingest_file(path):
            success_count += 1

    logger.info(f"✅ Initial ingestion completed. {success_count}/{len(filenames)} files processed.")


def ensure_collection_exists():
    """Collection yoksa oluşturur."""
    try:
        client = get_qdrant_client()
        collections = client.get_collections().collections
        names = [c.name for c in collections]

        if COLLECTION_NAME not in names:
            logger.info(f"📌 Collection `{COLLECTION_NAME}` not found. Creating it...")
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={"size": 768, "distance": "Cosine"}
            )
            logger.info("✅ Collection created.")
        else:
            logger.info(f"✓ Collection `{COLLECTION_NAME}` already exists.")
            
            # Collection stats
            collection_info = client.get_collection(COLLECTION_NAME)
            logger.info(f"📊 Collection has {collection_info.points_count} documents")
            
    except Exception as e:
        logger.error(f"❌ Error ensuring collection exists: {e}")
        raise


# -----------------------------
# RAG Chain Setup - ESNEKLEŞTİRİLMİŞ PROMPT
# -----------------------------

def get_rag_chain():
    """History-aware RAG chain döner."""
    vector_store = get_vectorstore()
    
    # Retriever - DAHA FAZLA CHUNK + DAHA BÜYÜK FETCH
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 12,  # 10'dan 12'ye çıktı - daha fazla bilgi
            "fetch_k": 40,  # 30'dan 40'a - daha geniş arama
            "lambda_mult": 0.5  # 0.6'dan 0.5'e - daha fazla çeşitlilik
        }
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,  # Daha yaratıcı sentez için artırdık
    )

    # Contextualize question prompt
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Sohbet geçmişine ve kullanıcının son sorusuna bakarak, "
         "soruyu bağımsız ve aranabilir bir şekilde yeniden formüle et. "
         "SADECE soruyu yeniden yaz, cevap verme."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 🔥 ÇOK İYİLEŞTİRİLMİŞ PROMPT - BİLGİ SENTEZİ + SAMİMİ TON
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         """Sen GTÜ Bilgisayar Mühendisliği öğrencilerine staj konusunda yardımcı olan samimi bir asistansın.

**ÖNEMLİ: Birden fazla doküman parçasından bilgi topluyorsan, onları BİRLEŞTİRİP ÖZETLEMELİSİN!**

Örnek: 
- Bir dokümanda "her staj en az 20 gün" yazıyor
- Başka dokümanda "toplamda 40 gün yapmalısınız" yazıyor
→ Sen şöyle cevapla: "2 staj yapman gerekiyor, her biri en az 20 gün olmak üzere toplamda 40 gün"

**Yanıt Tarzın:**
1. **Direkt cevap ver** - Uzatma, en önemli bilgiyi ilk cümlede söyle
2. **Karmaşık maddeleri basitleştir** - "Madde 2.3'e göre..." yerine "Şöyle ki..."
3. **Samimi ol** - "yapman gerekir" yerine "yapmalısın"
4. **Gereksiz detay verme** - Öğrenci ne soruyorsa ona odaklan

**Örnek İyi Cevaplar:**

Soru: "Zorunlu stajım kaç gün?"
İYİ: "2 staj yapmalısın, her biri en az 20 gün. Toplamda 40 gün tutuyor."
KÖTÜ: "Öğretim planında yer alan stajlar madde 9'a göre en az 40 iş günü olmak üzere..."

Soru: "Hangi belgeler gerekli?"
İYİ: "3 belge lazım: 1) Kabul belgesi (firmadan), 2) Müstehaklık belgesi (e-devletten), 3) Müfredat durumu (Proliz'den)"
KÖTÜ: "Staj başvurusu için aşağıda belirtilen evraklar gerekmektedir..."

**Kritik Kurallar:**
✅ Birden fazla belgeden gelen bilgiyi BİRLEŞTİR ve özetle
✅ İlk cümlede direkt cevabı ver, sonra detay ekle

**Context (Dokümanlar):**
{context}

Öğrencinin sorusunu yukarıdaki kurallara göre yanıtla:"""),
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
    logger.info("🚀 FastAPI starting...")
    ensure_collection_exists()
    ingest_pdfs_in_folder()
    logger.info("✅ Startup completed")


# -----------------------------
# API ENDPOINTS
# -----------------------------

@app.get("/")
def home():
    return {
        "message": "GTU Staj Chatbot Backend is running!",
        "collection": COLLECTION_NAME,
        "endpoints": ["/chat", "/upload-file", "/collection-info", "/search-test"]
    }


@app.get("/collection-info")
def collection_info():
    """Collection hakkında bilgi döner."""
    try:
        client = get_qdrant_client()
        collection = client.get_collection(COLLECTION_NAME)
        return {
            "collection_name": COLLECTION_NAME,
            "points_count": collection.points_count,
            "status": collection.status,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 🔥 YENİ: Test endpoint - retrieval testi için
@app.get("/search-test")
def search_test(query: str):
    """Retrieval testi yapar - hangi dokümanları buluyor gösterir."""
    try:
        vector_store = get_vectorstore()
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        
        docs = retriever.get_relevant_documents(query)
        
        return {
            "query": query,
            "found_docs": len(docs),
            "results": [
                {
                    "source": doc.metadata.get("source_file", "unknown"),
                    "chunk_id": doc.metadata.get("chunk_id", -1),
                    "content_preview": doc.page_content[:200]
                }
                for doc in docs
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


@app.post("/chat")
async def chat_api(request: ChatRequest):
    """Streaming chat response with history."""
    logger.info(f"💬 Chat request: {request.message[:50]}... (session: {request.session_id})")
    
    qa_chain = get_rag_chain()

    def generate():
        try:
            response = qa_chain.invoke(
                {"input": request.message},
                config={"configurable": {"session_id": request.session_id}}
            )
            
            answer = response.get("answer", "")
            logger.info(f"✅ Response generated: {len(answer)} chars")
            
            # Debug: Hangi kaynaklardan bilgi alındı
            source_docs = response.get("context", [])
            logger.info(f"📚 Used {len(source_docs)} source documents")
            
            # Kaynak dosyaları logla
            sources = set(doc.metadata.get("source_file", "unknown") for doc in source_docs)
            logger.info(f"📁 Sources: {sources}")
            
            # Stream karakter karakter
            for char in answer:
                yield char
                
        except Exception as e:
            error_msg = f"❌ Hata: {str(e)}"
            logger.error(error_msg)
            yield error_msg

    return StreamingResponse(generate(), media_type="text/plain")


@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    """Kullanıcının yüklediği PDF/TXT dosyasını ingest eder."""
    logger.info(f"📤 File upload request: {file.filename}")
    
    # Dosya tipi kontrolü
    if not (file.filename.endswith('.pdf') or file.filename.endswith('.txt')):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")
    
    # Upload dizini oluştur
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    save_path = f"{UPLOAD_DIR}/{file.filename}"
    
    try:
        # Dosyayı kaydet
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"💾 File saved: {save_path} ({len(content)} bytes)")
        
        # Dosyayı ingest et
        success = ingest_file(save_path)
        
        if success:
            # Collection stats
            client = get_qdrant_client()
            collection = client.get_collection(COLLECTION_NAME)
            
            return {
                "status": "success",
                "filename": file.filename,
                "size_bytes": len(content),
                "message": "Dosya başarıyla yüklendi ve sisteme eklendi",
                "collection_points": collection.points_count
            }
        else:
            raise HTTPException(status_code=500, detail="Dosya işlenirken hata oluştu")
            
    except Exception as e:
        logger.error(f"❌ Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))