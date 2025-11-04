__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from uuid import uuid4
from dotenv import load_dotenv
import shutil

load_dotenv()

# Read pdf content
raw_documents = PyPDFDirectoryLoader(path="/home/yunus/projects/llmBootcamp/week06/canli_aillm2/pdfs").load()

# Split raw pdf content into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=120
)

split_documents = text_splitter.split_documents(raw_documents)

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

CHROMA_PATH = "/home/yunus/projects/llmBootcamp/week06/canli_aillm2/chromadb"
shutil.rmtree(CHROMA_PATH, ignore_errors=True)  # varsa sil, önceden oluşturulduysa

vector_store = Chroma(
    collection_name    = "italia-guide",
    embedding_function = embeddings,
    persist_directory  = CHROMA_PATH, 
)

# Add items to vector store
uuids = [str(uuid4()) for _ in range(len(split_documents))]

vector_store.add_documents(documents=split_documents, ids=uuids)

print(f"Loaded {len(raw_documents)} documents")
print(f"Split into {len(split_documents)} chunks")
print("Embedding and saving to Chroma...")