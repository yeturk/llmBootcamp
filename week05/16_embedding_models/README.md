# 🧩 Embedding Models in LangChain

### 🔹 What are Embeddings?

* **Embeddings** = numerical vector representations of text.
* They capture **semantic meaning** → similar texts have vectors close together.
* Example:

  * *“dog”* and *“puppy”* → vectors are near.
  * *“dog”* and *“car”* → vectors are far.

---

### 🔹 Why Do We Need Embeddings?

* LLMs don’t “search” by words, they search by **meaning**.
* Embeddings power:

  * **Semantic Search** (find similar docs).
  * **Clustering & Classification**.
  * **Retrieval-Augmented Generation (RAG)**.

---

### 🔹 How LangChain Handles Embeddings

* Provides a **standard interface** → you can plug in different providers:

  * **Google** → `GoogleGenerativeAIEmbeddings` (Gemini).
  * **OpenAI** → `OpenAIEmbeddings`.
  * **Hugging Face**, **Cohere**, etc.
* Once you have embeddings, store them in a **vector database** (Chroma, Qdrant, Pinecone, etc.).

---

# 🧑‍💻 Example: Gemini Embeddings + Chroma Search

```python
# Install libraries
!pip install langchain==0.3.26 langchain-google-genai chromadb

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# 1. Initialize embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 2. Create sample documents
docs = [
    Document(page_content="LangChain helps developers build LLM apps."),
    Document(page_content="Embeddings map text into vectors for semantic search."),
    Document(page_content="Chroma is a vector database to store embeddings."),
]

# 3. Build a vectorstore
vectorstore = Chroma.from_documents(docs, embeddings)

# 4. Query the database
query = "How do we find similar meaning text?"
results = vectorstore.similarity_search(query, k=2)

print("🔎 Query:", query)
for i, r in enumerate(results, 1):
    print(f"{i}. {r.page_content}")
```

---

### ✅ Expected Output

```
🔎 Query: How do we find similar meaning text?
1. Embeddings map text into vectors for semantic search.
2. Chroma is a vector database to store embeddings.
```

---

# 🎯 Key Takeaways

* **Embedding models** = turn text into vectors.
* They are the **backbone of semantic search & RAG**.
* In LangChain, you choose an embedding provider → store vectors in a DB → query with similarity search.

