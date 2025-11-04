# ingest_texts.py
# Author: Yunus Emre
# Description: Ingests scraped EU AI Act text data into a Chroma vector store.

import os
import shutil
from uuid import uuid4
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# -------------------------------------------------------------------
# 1. Load environment variables
# -------------------------------------------------------------------
load_dotenv()

CHROMA_PATH = "./chroma_eu_ai"
shutil.rmtree(CHROMA_PATH, ignore_errors=True)  # her çalıştırmada temizle

# -------------------------------------------------------------------
# 2. Load local text file
# -------------------------------------------------------------------
print("📖 Loading scraped text file...")
loader = TextLoader("eu_ai_act_texts.txt", encoding="utf-8")
raw_documents = loader.load()
print(f"✅ Loaded {len(raw_documents)} document(s).")

# -------------------------------------------------------------------
# 3. Split into chunks
# -------------------------------------------------------------------
print("✂️ Splitting document into text chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,   # her chunk maksimum 1000 karakter
    chunk_overlap=110  # her chunk arasında 110 karakterlik örtüşme
)
split_documents = text_splitter.split_documents(raw_documents)
print(f"✅ Split into {len(split_documents)} chunks.")

# -------------------------------------------------------------------
# 4. Create embeddings and save to Chroma
# -------------------------------------------------------------------
print("🧠 Creating embeddings with Google Generative AI...")
embeddings = GoogleGenerativeAIEmbeddings(
    model     = "text-embedding-004",
    task_type = "RETRIEVAL_DOCUMENT"
)

# -------------------------------------------------------------------
# 5. ChromaDB Oluştur ve Belgeleri Ekle
# -------------------------------------------------------------------
vector_store = Chroma(
    collection_name     = "eu-ai-act",       # koleksiyon adı
    embedding_function  = embeddings,
    persist_directory   = CHROMA_PATH,       # vektör verisinin kalıcı olarak tutulacağı dizin
)

# Her chunk için benzersiz UUID üret
uuids = [str(uuid4()) for _ in range(len(split_documents))]

# Chroma’ya ekle
vector_store.add_documents(documents=split_documents, ids=uuids)

print("-" * 80)
print(f"✅ Ingestion complete. Vector store saved to: {CHROMA_PATH}")
print("-" * 80)

# -------------------------------------------------------------------
# 6. Debug Bilgisi (isteğe bağlı)
# -------------------------------------------------------------------
if __name__ == '__main__':
    print(type(raw_documents))       # <class 'list'>
    print("-" * 80)
    # print(raw_documents[:1])
    # print("-" * 80)
    print(f"Total raw docs: {len(raw_documents)}")
    print(f"Total split chunks: {len(split_documents)}")
    print("-" * 80)