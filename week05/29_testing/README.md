# 🚀 Testing in LangChain (Unit → Contract → Integration)

When you ship LLM apps, don’t rely on manual spot checks. Add **tests** to catch regressions in:

* **Prompt formatting** (does the template fill correctly?)
* **Parsing/contract** (does the output match the schema?)
* **End-to-end behavior** (does the chain run as expected?)

**Testing ≠ Evaluation**:

* *Testing* checks **correctness deterministically** (schemas, edge cases, errors).
* *Evaluation* compares **quality** (semantics, helpfulness) across prompts/models.
* Use **LangSmith** to run larger regression suites and track results over time (optional but powerful).

---

## 🧩 What to test

1. **Prompt Unit Test** – inputs → formatted string.
2. **Parser Contract Test** – given a candidate output → parse succeeds/fails clearly.
3. **Chain Integration Test** – run the chain; use a **fake LLM** for determinism, and a **live test** behind a flag.

---

## 🧪 Minimal Testable Chain (with a Fake LLM)

```python
# Colab-friendly; Python 3.12; langchain==0.3.26

from dataclasses import dataclass
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# -------- Chain pieces --------
TEMPLATE = """Extract the country and capital from the text.
Text: {text}

{format_instructions}
"""

schemas = [
    ResponseSchema(name="country", description="Country name"),
    ResponseSchema(name="capital", description="Capital city"),
]
parser = StructuredOutputParser.from_response_schemas(schemas)
FORMAT_INSTR = parser.get_format_instructions()

prompt = PromptTemplate(
    input_variables=["text"],
    partial_variables={"format_instructions": FORMAT_INSTR},
    template=TEMPLATE,
)

# Real LLM (used only if you want a live integration run)
def real_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# A tiny fake LLM for deterministic tests
@dataclass
class FakeLLM:
    # content to return from .invoke(...)
    content: str
    class _Resp:
        def __init__(self, content): self.content = content
    def invoke(self, _input):  # LangChain models support .invoke
        return FakeLLM._Resp(self.content)

def build_chain(llm):
    # Compose: prompt -> llm -> parser
    def run(text: str):
        formatted = prompt.format(text=text)
        raw = llm.invoke(formatted).content
        return parser.parse(raw)
    return run

# -------- Unit tests (simple asserts) --------

# 1) Prompt Unit Test
formatted = prompt.format(text="The capital of France is Paris.")
assert "Text: The capital of France is Paris." in formatted
assert "country" in FORMAT_INSTR and "capital" in FORMAT_INSTR  # sanity check

# 2) Parser Contract Test (valid JSON)
fake_ok = FakeLLM(content='{"country": "France", "capital": "Paris"}')
chain_ok = build_chain(fake_ok)
out = chain_ok("The capital of France is Paris.")
assert out == {"country": "France", "capital": "Paris"}

# 2b) Parser Contract Test (invalid JSON → expect a clean error)
fake_bad = FakeLLM(content="Paris is the capital of France.")
chain_bad = build_chain(fake_bad)
error_happened = False
try:
    chain_bad("The capital of France is Paris.")
except Exception as e:
    error_happened = True
assert error_happened, "Parser should fail on malformed output."

# 3) Integration Test (optional live call)
RUN_LIVE = False  # flip to True to try a real request
if RUN_LIVE:
    live_chain = build_chain(real_llm())
    live_out = live_chain("The capital of Japan is Tokyo.")
    # Basic shape check
    assert set(live_out.keys()) == {"country", "capital"}
    print("Live test passed:", live_out)

print("All tests passed ✅")
```

**How to use in class**

* Run the cell: students see assertions pass.
* Flip `RUN_LIVE=True` to demonstrate a real LLM integration test (optional).

---

## 🔗 Where LangSmith fits

* Keep the **fast unit/contract tests** in code (like above).
* For **regression suites** (dozens/hundreds of test cases, CSV/JSON datasets, multiple prompts/models), use **LangSmith** to:

  * Log runs and test results
  * Compare prompt/chain variants over time
  * Track failures & diffs when prompts change

**Rule of thumb**

* **Local tests** → fast feedback, CI.
* **LangSmith** → experiment tracking & large-scale regression testing.

---

## 📌 Takeaway

* Make LLM apps testable by **separating prompt, LLM, and parser**.
* Use a **Fake LLM** for deterministic unit/contract tests.
* Gate **live tests** behind a flag.
* Scale up with **LangSmith** for repeatable, shareable regression suites.
