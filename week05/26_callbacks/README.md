## 🚀 Callbacks in LangChain

Callbacks let you **observe** what happens inside LangChain as your chain or agent runs.
Think of them as **hooks**: they give you visibility into events like

* When a model is called
* What prompt is sent
* Intermediate results
* Errors or token usage

---

## ✨ Why use them?

* **Debugging** → See exactly what’s happening step by step
* **Logging** → Save inputs/outputs for analysis
* **Monitoring** → Track token usage, latency, costs

---

## 🔑 Core Idea

Without callbacks: you only see the **final output**.
With callbacks: you see **all intermediate steps** as they happen.

---

## 🧑‍💻 Example in LangChain

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.callbacks import StdOutCallbackHandler

# 1. Create a callback handler (prints to console)
handler = StdOutCallbackHandler()

# 2. Attach it to an LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    callbacks=[handler]
)

# 3. Run a request
response = llm.invoke("Explain LangChain in one sentence.")
print("=== Final Answer ===")
print(response.content)
```

---

## ✅ Example Output

```
[chain/start] Entering LLM call
[chain/end] Finished LLM call
=== Final Answer ===
LangChain is a framework for building applications powered by large language models.
```

---

## 📌 Takeaway

* Callbacks = **event listeners** inside LangChain
* Useful for **debugging, monitoring, and logging**
* Can send data to console, files, dashboards, or custom tools

