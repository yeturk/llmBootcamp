## 🚀 Few-Shot Prompting

LLMs learn **from examples inside the prompt**.
Instead of giving only instructions, you show the model **a few examples** of input–output pairs, so it generalizes better.

This is called **Few-Shot Prompting**.

---

## ✨ Why use it?

* **Guide model behavior** → Helps it follow your desired format.
* **Improve accuracy** → Reduces ambiguity.
* **Cheaper than fine-tuning** → No need to retrain a model.

---

## 🔑 Core Idea

Instead of just asking:

```
Translate this word to French: dog
```

We provide **examples**:

```
English: cat → French: chat  
English: house → French: maison  
English: dog → French: ?
```

The model sees the pattern and fills in correctly.

---

## 🧑‍💻 Example in LangChain

```python
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Define example input-output pairs
examples = [
    {"english": "cat", "french": "chat"},
    {"english": "house", "french": "maison"},
]

# 2. Template for formatting each example
example_template = "English: {english} -> French: {french}"
example_prompt = PromptTemplate(
    input_variables=["english", "french"], 
    template=example_template
)

# 3. Few-shot template with prefix and suffix
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Translate English words into French.",
    suffix="English: {input} -> French:",
    input_variables=["input"],
)

# 4. Format final prompt
final_prompt = few_shot_prompt.format(input="dog")
print("=== Prompt Sent to LLM ===")
print(final_prompt)

# 5. Send to model
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
English: house -> French: maison
English: dog -> French:

=== LLM Response ===
chien
```

---

## 📌 Takeaway

**Few-Shot Prompting** = teaching by example.
It is most useful when:

* You need the LLM to follow a **specific style/format**
* The task is **repetitive** (classification, translation, summarization, etc.)
* You don’t want to (or can’t) fine-tune a model
