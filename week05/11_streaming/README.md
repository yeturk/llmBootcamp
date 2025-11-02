## 🔹 LangChain Streaming (Quick Summary)

* **Streaming** means receiving the LLM response **incrementally (token by token)** instead of waiting for the entire output.
* **`.invoke()`** → returns the **full response at once**.
* **`.stream()`** → yields **chunks of text progressively**.
* **Benefits:**

  * Creates a “typing effect” in chatbots and apps.
  * Faster perceived response time for users.
  * Enables real-time interactivity in UIs.

---

## 🔹 Minimal Example with Gemini 2.5 Flash

```python
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize Gemini 2.5 Flash
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key="YOUR_GOOGLE_API_KEY"
)

# Stream response token by token
for chunk in llm.stream("Explain artificial intelligence in one sentence."):
    print(chunk.content, end="", flush=True)

print("\n--- Done ---")
```

👉 Output will appear **piece by piece** (although in Colab it may look like one block, under the hood it’s streamed).

---

⚡ Done in under 5 minutes: theory + a working Colab-friendly example.

Do you want me to also create a **visual one-slide diagram** (showing “invoke = full response” vs. “stream = token flow”) that you can drop into your deck?
