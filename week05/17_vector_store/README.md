# 🗂️ Vector Stores in LangChain (5-Minute Explanation)

### 🔹 What is a Vector Store?

* A **database specialized for embeddings**.
* Stores documents as vectors (from an embedding model).
* Allows **similarity search**: given a query, return the most semantically related documents.

---

### 🔹 Why Do We Need Vector Stores?

* Normal databases search by **keywords**.
* Vector stores search by **meaning**.
* Core use cases:

  * **Semantic search**
  * **RAG (Retrieval-Augmented Generation)**
  * **Deduplication / clustering**

---

### 🔹 How They Work

1. **Text → Embedding** (vector representation).
2. **Stored in Vector Store** (with metadata).
3. **Query → Embedding**.
4. **Similarity Search** (cosine similarity, dot product, etc.).
5. Return top-k relevant documents.

---

### 🔹 Common Vector Stores

* Open-source: **Chroma**, **Qdrant**, **Weaviate**, **FAISS**.
* Cloud: Pinecone, Milvus, Vespa, Redis Vector.

---

# 🧑‍💻 Example: Chroma + Gemini Embeddings

```python
# Install
!pip install langchain==0.3.26 langchain-google-genai chromadb

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# 1. Embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 2. Example documents
docs = [
    Document(page_content="Vector stores allow semantic search using embeddings."),
    Document(page_content="Chroma is an open-source vector database."),
    Document(page_content="LangChain helps build LLM applications."),
]

# 3. Build the vector store
vectorstore = Chroma.from_documents(docs, embeddings)

# 4. Query
query = "How can I search by meaning instead of keywords?"
results = vectorstore.similarity_search(query, k=2)

print("🔎 Query:", query)
for i, r in enumerate(results, 1):
    print(f"{i}. {r.page_content}")

```


# 🎯 Key Takeaways

* **Vector store = embedding database**.
* Enables **semantic search** instead of keyword search.
* **Chroma** is the easiest entry point → great for demos & RAG prototypes.

---