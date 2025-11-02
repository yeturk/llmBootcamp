# What is LCEL?
- The LangChain Expression Language (LCEL) takes a declarative approach to building new Runnables from existing Runnables.

* **Composable graph for LLM apps** using the `|` pipe operator (aka **Runnables**).
* **Deterministic wiring**, not just prompt strings: you build flows like
  `input → prompt → model → parser`.
* Works the same for **sync / async / batch / streaming** with zero code changes to the graph.
* **Type-aware I/O** (dict in, string out, etc.), easy to test each stage.
* Built-ins for **fallbacks / retries / parallelism / routing**.
* Portable across providers—just swap the model node.

---

# LCEL Example

```python
# pip install langchain==0.3.26 langchain-google-genai==2.1.9 # in Colab/local
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

# 1) Model (Gemini 2.5 Flash) – keep the bootcamp rule
os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 2) Prompt (uses LCEL placeholders)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise tutor."),
    ("human", "Explain {topic} in exactly 2 short bullet points.")
])

# 3) Parser (turns model messages into string)
parser = StrOutputParser()

# 4) Chain with LCEL pipes
chain = prompt | llm | parser

# --- A) Single call
print("=== INVOKE ===")
print(chain.invoke({"topic": "LangChain Expression Language"}))

# --- B) Streaming (same graph, different call)
print("\n=== STREAM ===")
for chunk in chain.stream({"topic": "LangChain Expression Language"}):
    print(chunk, end="", flush=True)
print()

# --- C) Batch (same graph)
print("\n=== BATCH ===")
for out in chain.batch([
    {"topic": "RAG"},
    {"topic": "Prompt Engineering"},
]):
    print("-", out.strip())
```

**What this shows**

* **LCEL = pipes**: `prompt | llm | parser`
* Same chain supports **`.invoke()`**, **`.stream()`**, **`.batch()`**—no rewiring.
* Clean separation of concerns: prompt template, model, output parser.

---

