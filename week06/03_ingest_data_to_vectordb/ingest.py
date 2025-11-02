__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# Python’un dahili sqlite3 modülünü kaldırıp, onun yerine pysqlite3’ü kullan.

from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from uuid import uuid4
import shutil # to delete existing chromaDB folder
from dotenv import load_dotenv

load_dotenv()

# Read pdf content
raw_documents = PyPDFDirectoryLoader(path="/home/yunus/projects/llmBootcamp/week06/03_ingest_data_to_vectordb/pdfs").load()

# Split raw pdf content into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,    # her parçanın maksimum uzunluğu 1500 karakter
    chunk_overlap=200   # parçalar arasında 200 karakterlik örtüşme
)
# print(type(text_splitter))

split_documents = text_splitter.split_documents(raw_documents)
# print(type(split_documents))

# Bu kısım PDF’ten elde edilen metinleri vektör temsillere çevirir.
# Yani her chunk, artık yüksek boyutlu bir sayı listesine dönüşür.
embeddings = GoogleGenerativeAIEmbeddings( model     = "text-embedding-004",
                                           task_type = "RETRIEVAL_DOCUMENT")

CHROMA_PATH = "/home/yunus/projects/llmBootcamp/week06/03_ingest_data_to_vectordb/chromadb"
shutil.rmtree(CHROMA_PATH, ignore_errors=True)  # varsa sil, önceden oluşturulduysa

vector_store = Chroma(
    collection_name     = "italia-guide",   # bu koleksiyonun adı (veri seti etiketi)
    embedding_function  = embeddings,
    persist_directory   = CHROMA_PATH,      # vektörlerin ve metadata’nın kalıcı olarak saklandığı dizin
)

# Add items to vector store
uuids = [str(uuid4()) for _ in range(len(split_documents))]

vector_store.add_documents(documents=split_documents, ids=uuids)

if __name__=='__main__':
    print(type(raw_documents))  # <class 'list'>
    print("-" * 80)
    print(raw_documents[:3])
    print("-" * 80)
    print(len(raw_documents))
    print("-" * 80)
    print(len(split_documents))
    print("-" * 80)
    print(split_documents[:5])


print("-" * 80)