## 🚀 Output Parsers in LangChain

When an LLM responds, it usually returns **free-form text**.
But in real applications, we often need **structured outputs** (like JSON, lists, or numbers).

**OutputParser** helps us transform the raw text into a structured format.

---

## ✨ Why use them?

* **Reliability** → Get data in a predictable structure.
* **Automation** → Easy to pass LLM outputs to downstream systems.
* **Error handling** → Parsers can validate and retry if output is malformed.

---

## 🔑 Core Idea

Instead of:

```
"The capital of France is Paris."
```

We want structured output:

```json
{"country": "France", "capital": "Paris"}
```

---

## 🧑‍💻 Example in LangChain

```python
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Define schema for expected output
schemas = [
    ResponseSchema(name="country", description="The country being asked about"),
    ResponseSchema(name="capital", description="The capital city of the country"),
]

parser = StructuredOutputParser.from_response_schemas(schemas)
format_instructions = parser.get_format_instructions()

# 2. Build a prompt with instructions
template = """Extract the country and its capital from the text.

Text: {text}

{format_instructions}
"""

prompt = PromptTemplate(
    input_variables=["text"],
    partial_variables={"format_instructions": format_instructions},
    template=template,
)

# 3. Generate response
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
final_prompt = prompt.format(text="The capital of France is Paris.")
response = llm.invoke(final_prompt)

# 4. Parse into structured data
structured_output = parser.parse(response.content)
print(structured_output)
```

---

## ✅ Output (example)

```python
{'country': 'France', 'capital': 'Paris'}
```

---

## 📌 Takeaway

**Output Parsers** make LLM outputs machine-readable.
They are essential when your application needs to:

* Extract fields from text
* Work with APIs and databases
* Ensure consistent, structured results

