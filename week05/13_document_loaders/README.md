# What are “Document Loaders”?

Document Loaders **ingest external data** and normalize it into LangChain’s `Document` objects:

```python
Document(page_content="text...", metadata={"source": "...", "title": "...", ...})
```

Once data is in this uniform shape, you can:

* **Clean/transform** (HTML → text, remove boilerplate)
* **Split** into chunks (for RAG)
* **Embed & index** (vector stores)
* **Query** with LLMs

---

## Common Loader Types (when to use)

* **Local files**

  * `TextLoader`: Plain `.txt`, `.md` (fastest path).
  * `CSVLoader`: Tabular text → one `Document` per row (or merged).
  * `PyPDFLoader`: PDFs page-by-page; preserves page metadata.
  * `Unstructured*` family: robust for messy office docs (needs extra deps).
* **Web & APIs**

  * `WebBaseLoader`: Fetches web pages, parses with `bs4`.
  * Specialized SaaS loaders (GDrive, Slack, Notion, etc.) via integrations.
* **Directories**

  * `DirectoryLoader`: Recursively load multiple file types with a glob.

> Rule of thumb: **Prefer the simplest loader** that produces clean text and rich metadata. You can always post-process with splitters and cleaners.

---

## Loader Pitfalls & Tips

* **HTML noise**: Use `WebBaseLoader` (it sanitizes via BeautifulSoup) or add your own cleaner.
* **Chunking matters**: Split after loading to \~500–1,000 chars with overlap (100–150) for better retrieval.
* **Metadata is gold**: Keep `source`, `page`, `url`, etc. for citations and filtering.
* **Batching**: Many loaders can read dozens/hundreds of files; monitor memory and persist embeddings.
* **PDF quality varies**: If text extraction is poor, try different loaders (e.g., `PyPDFLoader` vs `UnstructuredPDFLoader`).

---

# Minimal End-to-End Example (Loader → Split → Embed → Retrieve → Answer)

**What it does**

1. Loads a local `.txt` and a web page
2. Splits to chunks
3. Embeds with **GoogleGenerativeAIEmbeddings** (`text-embedding-004`)
4. Stores in **Chroma**
5. Answers a question with **Gemini 2.5 Flash** via a tiny LCEL chain

> Light dependencies only. Works in Colab/local.

```python
# !pip install -q langchain==0.3.26 langchain-google-genai==2.* chromadb==0.5.* beautifulsoup4 pypdf

import os
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Loaders
from langchain_community.document_loaders import TextLoader, WebBaseLoader, PyPDFLoader

# Vector store + embeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# --------- 0) Config (Gemini) ----------
os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"

# LLM & Embeddings (Gemini family)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
emb = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")  # Google embeddings

# --------- 1) Load documents ----------
docs = []

# a) Local txt
Path("sample.txt").write_text("LangChain Document Loaders streamline data ingestion into LLM apps.")
docs += TextLoader("sample.txt").load()

# b) Web page (keeps URL metadata; requires bs4)
web_docs = WebBaseLoader("https://python.langchain.com/docs/concepts/document_loaders/").load()
docs += web_docs

# (Optional) c) PDF example if you have a local PDF
# pdf_docs = PyPDFLoader("example.pdf").load()
# docs += pdf_docs

print(f"Loaded {len(docs)} raw documents.")
print("Sample metadata:", docs[0].metadata)

# --------- 2) Split into chunks ----------
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
chunks = splitter.split_documents(docs)
print(f"Produced {len(chunks)} chunks. First chunk preview:\n", chunks[0].page_content[:200])

# --------- 3) Embed & index (Chroma) ----------
vs = Chroma.from_documents(documents=chunks, embedding=emb, persist_directory="chroma_doc_loader_demo")
retriever = vs.as_retriever(search_kwargs={"k": 3})

# --------- 4) Tiny LCEL RAG chain ----------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise tutor. Use the provided context to answer. If unsure, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])
parser = StrOutputParser()

def format_docs(docs_):
    return "\n\n".join(
        f"[{i+1}] Source: {d.metadata.get('source','unknown')}\n{d.page_content}"
        for i, d in enumerate(docs_)
    )

# LCEL graph: user question -> retrieve -> prompt -> Gemini -> string
from langchain_core.runnables import RunnableParallel, RunnableLambda

chain = (
    {"context": RunnableLambda(lambda x: format_docs(retriever.get_relevant_documents(x["question"]))),
     "question": RunnablePassthrough := RunnableLambda(lambda x: x["question"])}
    | prompt
    | llm
    | parser
)

# --------- 5) Ask a question ----------
q = "What are LangChain document loaders and why use them?"
print("\n=== ANSWER ===")
print(chain.invoke({"question": q}))

# (Optional) Streaming version
print("\n=== STREAM (token-by-token) ===")
for chunk in chain.stream({"question": q}):
    print(chunk, end="", flush=True)
print()
```

**What to point out to students**

* `TextLoader`, `WebBaseLoader`, `PyPDFLoader` each return `Document` with **useful metadata** (`source`, `page`, `url`).
* We **split after loading** to keep chunks retrieval-friendly.
* We use **Google embeddings** (`text-embedding-004`) to align with your rule.
* LCEL (`|`) lets us wire **retriever → prompt → model** clearly.
* The **same chain** supports `.invoke()` and `.stream()` without changes.

---

## Quick Checklist

1. Show 3 loaders (txt, web, pdf) and inspect `metadata`.
2. Demonstrate splitting and explain **chunk size/overlap**.
3. Embed with **Google embeddings**; store in Chroma.
4. Build a one-line LCEL chain and **ask one question**.
5. Repeat the question with **`.stream()`** to show token flow.

If you’d like, I can deliver a **1-slide diagram** for “Load → Split → Embed → Retrieve → Generate” matching your previous streaming slide.
