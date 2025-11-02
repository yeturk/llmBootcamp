import shutil
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004", task_type="RETRIEVAL_DOCUMENT")

CHROMA_PATH = "/home/yunus/projects/llmBootcamp/week06/04_adding_external_tool/chromadb"
shutil.rmtree(CHROMA_PATH, ignore_errors=True)  # varsa sil, önceden oluşturulduysa

vector_store = Chroma(
    collection_name="italia-guide",
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH 
)