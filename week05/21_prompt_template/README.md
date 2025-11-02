## 🚀 Prompt Templates in LangChain

When working with LLMs, we often need to send prompts that follow a **consistent structure**. Instead of writing full prompt strings each time, **PromptTemplate** lets you define a reusable template with placeholders for dynamic inputs.

### ✨ Why use them?

* **Reusability** → Define once, use with different inputs.
* **Consistency** → Ensures prompts follow the same format.
* **Flexibility** → Insert variables (like `{question}`, `{context}`) dynamically.

---

## 🔑 Core Idea

```text
"Answer the following question: {question}"
```

Here, `{question}` is a placeholder. When filled with `"What is LangChain?"`, the final prompt becomes:

```text
"Answer the following question: What is LangChain?"
```

---

## 🧑‍💻 Example in LangChain

```python
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Define a template with a placeholder
template = "You are a helpful tutor. Answer the following question clearly:\nQuestion: {question}"

prompt = PromptTemplate(
    input_variables=["question"],
    template=template,
)

# 2. Fill it with a user question
final_prompt = prompt.format(question="What is Retrieval-Augmented Generation (RAG)?")
print(final_prompt)

# 3. Send it to an LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
response = llm.invoke(final_prompt)
print(response.content)
```

---

## ✅ Output (example)

```
You are a helpful tutor. Answer the following question clearly:
Question: What is Retrieval-Augmented Generation (RAG)?

RAG is a method where an AI retrieves relevant documents from a knowledge base and uses them to provide more accurate answers...
```

---

### 📌 Takeaway

**Prompt Templates** are the foundation of structured prompting in LangChain.
They help you:

* Keep prompts clean
* Reuse them across your app
* Dynamically inject user input
