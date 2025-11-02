## 🚀 Example Selectors

When using **few-shot prompting**, you may have many examples.
But including **all of them** in every prompt can be inefficient.

👉 **Example Selectors** automatically choose the *most relevant* examples for the current input.

---

## ✨ Why use them?

* **Scalability** → No need to hardcode examples.
* **Relevance** → Selects examples closest to the user query.
* **Efficiency** → Keeps prompts short, avoids token waste.

---

## 🔑 Core Idea

Instead of showing *all* examples:

```
English: cat → French: chat  
English: house → French: maison  
English: tree → French: arbre  
English: dog → French: ?
```

With Example Selector, if the input is `"fish"`, it might select **animal-related examples** only:

```
English: cat → French: chat  
English: dog → French: chien  
English: fish → French: ?
```

---

## 🧑‍💻 Example in LangChain

```python
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import InMemoryStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# 1. Define examples
examples = [
    {"english": "cat", "french": "chat"},
    {"english": "dog", "french": "chien"},
    {"english": "house", "french": "maison"},
    {"english": "tree", "french": "arbre"},
]

# 2. Example formatting template
example_template = "English: {english} -> French: {french}"
example_prompt = PromptTemplate(
    input_variables=["english", "french"], 
    template=example_template
)

# 3. Build Example Selector (semantic search with embeddings)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    embeddings,
    FAISS,  # vector database
    k=2,    # pick 2 most similar examples
)

# 4. Few-shot prompt with dynamic example selection
dynamic_prompt = FewShotPromptTemplate(
    example_selector=selector,
    example_prompt=example_prompt,
    prefix="Translate English words into French.",
    suffix="English: {input} -> French:",
    input_variables=["input"],
)

# 5. Format final prompt
final_prompt = dynamic_prompt.format(input="bird")
print("=== Prompt Sent to LLM ===")
print(final_prompt)

# 6. Send to model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
response = llm.invoke(final_prompt)
print("=== LLM Response ===")
print(response.content)
```

---

## ✅ Output (example)

```
=== Prompt Sent to LLM ===
Translate English words into French.
English: cat -> French: chat
English: dog -> French: chien
English: bird -> French:

=== LLM Response ===
oiseau
```

---

## 📌 Takeaway

**Example Selectors** make few-shot prompts **dynamic**:

* They pick only the most relevant examples
* They reduce token usage
* They improve LLM accuracy on varied inputs
