# 🔹 Gerekli kütüphaneleri içe aktar
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
import shutil # to delete existing chromaDB folder
from pprint import pprint
from dotenv import load_dotenv

load_dotenv()

# 🔹 PDF dosyanin bulunduğu klasörün yolunu belirt
# Burada sadece klasör verilir, tek PDF bile olsa o klasörün içine koymalisin
pdf_klasoru = "/home/yunus/projects/llmBootcamp/week06/delstaj/delpdfs"   # senin dosyan bu dizinde

# 🔹 PDF'leri yükle
loader = PyPDFDirectoryLoader(path=pdf_klasoru)
# print(type(loader))     # <class 'langchain_community.document_loaders.pdf.PyPDFDirectoryLoader'>

raw_documents = loader.load()
print(f"📄 Toplam belge sayisi: {len(raw_documents)}")
# print(type(raw_documents))          # <class 'list'>
# print(type(raw_documents[11]))      # <class 'langchain_core.documents.base.Document'>
# print(len(raw_documents))           # 28
# print(raw_documents[11])

print("-" * 80)

# 🔹 2. Metinleri parçalara böl
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,   # her parçanin maksimum karakter sayisi
    chunk_overlap=300  # parçalar arasi çakişma miktari (önceki parçalardan 200 karakter alinir)
)

split_documents = text_splitter.split_documents(raw_documents)

print(f"🔹 Toplam parça sayisi: {len(split_documents)}")
# print("İlk parçadan bir örnek:\n")
# print(split_documents[72].page_content[:400])
print("-" * 80)

# 🔹 3. Google Embeddings oluştur
embeddings = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",
    task_type="RETRIEVAL_DOCUMENT"
)

CHROMA_PATH = "/home/yunus/projects/llmBootcamp/week06/delstaj/delchromadb"
shutil.rmtree(CHROMA_PATH, ignore_errors=True)  # varsa sil, önceden oluşturulduysa

print("6️⃣ Chroma vektör veritabani oluştur")
# 6️⃣ Chroma vektör veritabani oluştur
vector_store = Chroma(
    collection_name     = "gtuStajBelgesi",
    embedding_function  = embeddings,
    persist_directory   = CHROMA_PATH  # kalici olarak diske yaz
)

# 7️⃣ UUID (benzersiz kimlik) üret
uuids = [str(uuid4()) for _ in range(len(split_documents))]

print("8️⃣ Belgeleri vektör veritabanina ekle")
# 8️⃣ Belgeleri vektör veritabanina ekle
vector_store.add_documents(documents=split_documents, ids=uuids)

print("✅ Belgeler başariyla Chroma veritabanina eklendi!")
print(f"Toplam belge: {len(vector_store.get()['ids'])}")
print("-" * 80)

##### SEARCHING (search.py)
results = vector_store.similarity_search("When should the internship form be filled out?", k=2)

for res in results:
    print(f"🔹 {res.page_content[:400]} \n")
    print(f"📄 Kaynak: {res.metadata['source']} (Sayfa: {res.metadata.get('page', 'N/A')}) \n")
    print("-" * 80)