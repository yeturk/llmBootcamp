
# 🔎 Retrieval in LangChain (5-Minute Explanation)

* **What is Retrieval?**
  Retrieval is the process of **finding relevant information** from a knowledge base (like a vector database or document store) to help the LLM answer user queries.

* **Why is it important?**
  LLMs don’t have access to your private or up-to-date data. Retrieval lets you provide context from external sources (e.g., PDFs, database, website).

* **How does it work?**

  1. **Embed Documents** → Convert text into embeddings (vectors).
  2. **Store in Vector DB** → Save embeddings in a vector database like Chroma or Qdrant.
  3. **Query** → Convert user query into an embedding.
  4. **Similarity Search** → Retrieve the most similar documents.
  5. **Pass to LLM** → Combine retrieved docs + prompt → LLM generates the final answer.

* **LangChain Abstraction**:
  Retrieval is handled by a **Retriever** object, usually created from a vectorstore.

---

# 🧑‍💻 Example: Simple Retrieval with Chroma + Gemini 2.5 Flash

```python
# Install needed libraries (Colab-friendly)
!pip install langchain==0.3.26 langchain-google-genai chromadb

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# 1. Initialize embeddings and LLM
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key="YOUR_API_KEY")

# 2. Create some sample documents
docs = [
    Document(page_content="LangChain helps developers build applications with LLMs."),
    Document(page_content="Retrieval allows LLMs to access external knowledge bases."),
    Document(page_content="Chroma is a lightweight open-source vector database.")
]

# 3. Build a vectorstore and retriever
vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# 4. User query
query = "What is retrieval in LangChain?"

# 5. Get relevant documents
relevant_docs = retriever.invoke(query)
print("Retrieved docs:", [d.page_content for d in relevant_docs])

# 6. Combine docs + question into a prompt
context = "\n".join([d.page_content for d in relevant_docs])
final_prompt = f"Answer the question using the context:\n\n{context}\n\nQuestion: {query}"

# 7. Ask the LLM
response = llm.invoke(final_prompt)
print("\nAI Response:", response.content)
```

---

# ✅ Key Takeaways

* **Retriever** = interface to pull relevant docs.
* **Vectorstore** = where embeddings live.
* **Retrieval-Augmented Generation (RAG)** = retrieval + LLM answer.

---

⚡ In a real course, this would be the **bridge to Week 6: RAG**, showing how retrieval powers chatbot-style applications.

---


1. **Retriever only pulls snippets** that are most similar.
2. If those snippets are **too shallow**, the LLM doesn’t have enough to give a good, direct explanation.
3. The model then says “not available in context” — which is technically correct.

---

### How to Fix (for Teaching)

You can show students how to:

* **Improve the documents** (give richer context).
* **Add instructions in the final prompt** so the LLM synthesizes even if docs are incomplete.

---

### Improved Example Prompt

```python
# Combine docs + query with stronger instruction
context = "\n".join([d.page_content for d in relevant_docs])
final_prompt = f"""
You are an AI tutor. Use the provided context to explain the concept clearly. 
If the context is incomplete, combine it with your own knowledge.

Context:
{context}

Question: What is retrieval in LangChain?
"""

response = llm.invoke(final_prompt)
print("\nAI Response:", response.content)
```

---

### Why This is Good for Students

* They **see the limitation** of raw retrieval.
* They **learn prompting fixes**.
* It prepares them for **Week 6: RAG** — where retrieval chains handle this more elegantly.

---

👉 Do you want me to adjust the example into a **proper `create_retrieval_chain` demo** (so the response is more accurate and less dependent on manual prompt crafting)?
