## 🚀 Async in LangChain

When working with LLMs, API calls can be **slow** (1–10 seconds).
If you need to run **multiple requests at once**, using **async/await** helps you do it **concurrently**.

LangChain supports **asynchronous methods** (like `.ainvoke()`, `.abatch()`, `.astream()`).

---

## ✨ Why use async?

* **Speed** → Run many LLM calls in parallel.
* **Efficiency** → Don’t block your program while waiting.
* **Scalability** → Useful for chatbots, batch Q\&A, pipelines.

---

## 🔑 Core Idea

Instead of:

```python
answer1 = llm.invoke("What is AI?")
answer2 = llm.invoke("What is ML?")
```

(run **sequentially**)

We can run **concurrently**:

```python
answer1, answer2 = await asyncio.gather(
    llm.ainvoke("What is AI?"),
    llm.ainvoke("What is ML?")
)
```

---

## 🧑‍💻 Example in LangChain

```python
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Define async workflow
async def main():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    # Run two calls concurrently
    results = await asyncio.gather(
        llm.ainvoke("Explain AI in one sentence."),
        llm.ainvoke("Explain ML in one sentence."),
    )

    # Print results
    for r in results:
        print("=== Response ===")
        print(r.content)

# 2. Run event loop
asyncio.run(main())
```

---

## ✅ Output (example)

```
=== Response ===
AI is the science of building machines that can mimic human intelligence.
=== Response ===
ML is a subset of AI where machines learn from data instead of being explicitly programmed.
```

---

## 📌 Takeaway

* `.invoke()` → synchronous
* `.ainvoke()` → asynchronous
* `asyncio.gather()` → run multiple calls in parallel
* Best for **batch tasks, chat apps, and pipelines**

