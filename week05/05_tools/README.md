## Overview
The tool abstraction in LangChain associates a Python function with a schema that defines the function's name, description and expected arguments.

Tools can be passed to chat models that support tool calling allowing the model to request the execution of a specific function with specific inputs.

Tools are a way to encapsulate a function and its schema in a way that can be passed to a chat model.

Create tools using the @tool decorator, which simplifies the process of tool creation, supporting the following:
- Automatically infer the tool's name, description and expected arguments, while also supporting customization.
- Defining tools that return artifacts (e.g. images, dataframes, etc.)
- Hiding input arguments from the schema (and hence from the model) using injected tool arguments.

The key methods to execute the function associated with the tool:

- invoke: Invokes the tool with the given arguments.
- ainvoke: Invokes the tool with the given arguments, asynchronously. Used for async programming with Langchain.

In LangChain, a **Tool** is **a wrapper around a function or an external capability that an agent can use**.

---

### 📦 Think of a Tool as a "Skill"

Imagine you have an AI assistant.
You want it to do more than just chat — like:

* Search the web 🕵️
* Use a calculator 🧮
* Query a database 📊
* Call an API 🛰️
* Send an email 📧

Each of these **capabilities** is a *tool*.
LangChain lets you wrap any Python function (or API) and expose it to the agent as a **tool** it can choose to use.

## Create tools using @tool decorator
- create main.py in 05_tools directory
```python
import os
from dotenv import load_dotenv
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

logger = logging.getLogger(__name__)  # 👈 Uses the module name
logger.setLevel(logging.INFO)

if not logger.handlers:
    ch = logging.StreamHandler()
    fh = logging.FileHandler("05_tools.log")

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

# Example usage
logger.info("This log message includes the module name.")

@tool
def multiply(a: int, b: int) -> int:
   """Multiply two numbers."""
   return a * b

def main():
    load_dotenv()
    logger.info("Environment variables loaded from .env file (if present).")
    api_key = os.getenv("GOOGLE_API_KEY")
    # ... (API key check) ...
    if api_key:
        # Print last 4 character of key
        logger.info(f"GOOGLE_API_KEY: ********{api_key[:4]}")
    else:
        logger.error("Error: GOOGLE_API_KEY not found in environment variables.")
        return # Exit if no API key

    # Use tool directly
    logger.info(f"Using tool directly {multiply.invoke({"a": 2, "b": 3})} - name: {multiply.name} - Description: {multiply.description} - Args: {multiply.args}")

if __name__=='__main__':
    main()
```
- Run 
```bash
uv run week_05/05_tools/main.py 
```
- Output
```
2025-07-29 15:54:54,909 - __main__ - INFO - This log message includes the module name.
2025-07-29 15:54:54,911 - __main__ - INFO - Environment variables loaded from .env file (if present).
2025-07-29 15:54:54,911 - __main__ - INFO - GOOGLE_API_KEY: ********AIza
2025-07-29 15:54:55,044 - __main__ - INFO - Using tool directly 6 - name: multiply - Description: Multiply two numbers. - Args: {'a': {'title': 'A', 'type': 'integer'}, 'b': {'title': 'B', 'type': 'integer'}}
```



## 🛠️ Tools in Practice

A LangChain tool is just a **function + some metadata**, wrapped using the `@tool` decorator or `Tool` class.

---

## 🧠 How Does the Agent Know to Use Tools?

LangChain’s **agents** are like smart decision-makers.

When you give the agent:

* A prompt ("What is 10 times 5?")
* And a list of tools

It will:

1. **Interpret the user request**
2. **Pick a tool if needed**
3. **Call the tool with the right input**
4. **Use the result to respond**

This is similar to how humans solve problems:

> “I don’t know the answer directly, but I can Google it” or “I’ll use a calculator.”

---

### 🧠 Agent + Tool Example

Let’s use LangChain's `AgentExecutor` with tools:

```python
from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI

# rest above

tools = [multiply]

llm = ChatGoogleGenerativeAI(
        api_key=api_key,
        model="gemini-2.5-flash", 
        temperature=0 
    )
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

agent.run("What is 8 times 7?")
```

The agent will:

* Read your question
* Decide to use the `multiply` tool
* Feed it `(8, 7)`
* Get the result `56`
* Return the answer: `"The result is 56"`

---

## 🔍 Why Tools Matter

Without tools, a language model is limited to what it has learned during training.

**With tools**, it becomes:

* **Interactive** (calls APIs)
* **Up-to-date** (search tools)
* **Actionable** (sends emails, executes code, updates a database)

---

## 🧩 Summary

| Concept         | Meaning in LangChain                     |
| --------------- | ---------------------------------------- |
| **Tool**        | A wrapped function the agent can use     |
| **Agent**       | A smart planner that uses tools to act   |
| **Tool Input**  | A string or dictionary given to the tool |
| **Tool Output** | Returned value used in the final answer  |

---


