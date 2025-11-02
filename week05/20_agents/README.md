# 🕹️ Agents in LangChain

## 1. What is an Agent?

* Normally, chains run in a **fixed order** (retriever → prompt → LLM).
* An **Agent** is more flexible: the LLM decides **what action to take next** based on the user’s request.
* It can:

  * **Use tools** (e.g., a retriever, calculator, search API).
  * **Decide step-by-step** what to call.
  * **Loop** until it finds an answer.

📌 Think of an agent as giving your LLM a **toolbox + reasoning loop** instead of a single script.

---

## 2. Why Use Agents?

* Useful when tasks require **multiple steps** (e.g., look up → calculate → explain).
* Make apps **dynamic**: the model picks the right tool at runtime.
* Great for chatbots, assistants, task automation.

---

## 3. How Agents Work

1. User gives a query.
2. Agent decides which **tool(s)** to use.
3. Calls tools (retriever, calculator, API, DB).
4. Collects results, reasons again.
5. Produces final answer.

---

### 🕹️ Agents in LangChain — with @tool Decorator
Why the decorator?

- Lets you define a tool with a docstring as its description.
- No need to manually build Tool(...) objects.
- Cleaner to scale as you add more tools.

## 4. Colab Demo — Wiki Lookup Agent

```python
!pip install langchain==0.3.26 langchain-google-genai

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool

# 1. LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key="YOUR_API_KEY")

# 2. Fake "Wiki" database
wiki_data = {
    "python": "Python was created by Guido van Rossum in 1991.",
    "langchain": "LangChain is a framework for building LLM-powered applications."
}

def wiki_lookup(query: str) -> str:
    for k, v in wiki_data.items():
        if k in query.lower():
            return v
    return "I could not find that."

# 3. Define tool
tools = [
    Tool(
        name="Wiki",
        func=wiki_lookup,
        description="Look up facts about Python or LangChain."
    )
]

# 3) Initialize the agent with VERBOSE mode to print the trace
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type="zero-shot-react-description",
    verbose=True     # <-- prints Thought/Action/Observation steps
)

# 5. Ask a question
response = agent.invoke("Who created Python?")
print("🤖 Agent:", response["output"])
```
## 5. Expected Output
```
--- Q1: Who created Python? ---


> Entering new AgentExecutor chain...
Action: wiki
Action Input: python
Observation: Python was created by Guido van Rossum in 1991.
Thought:I now know the final answer
Final Answer: Guido van Rossum created Python.

> Finished chain.
Final: Guido van Rossum created Python.

--- Q2: What is 45 plus 55? ---


> Entering new AgentExecutor chain...
Action: add_numbers
Action Input: 45 plus 55
Observation: The sum is 100
Thought:I now know the final answer
Final Answer: The sum is 100

> Finished chain.
Final: The sum is 100
```

---

## 6. Teaching Takeaways

* **Chains** = fixed sequence.
* **Agents** = flexible, tool-using, reasoning loops.
* In practice, agents power **assistant-like apps** (search, DB queries, APIs).
* This demo shows the simplest agent calling one tool.



# 🔍 How does the LLM decide which tool to use?

1. **Prompting / Instructions**

   * When we initialize the agent, LangChain sends the LLM a **system prompt** that describes all available tools.
   * Example (simplified):

     ```
     You are an agent. You can use these tools:
     - Wiki: Look up facts about Python or LangChain.
     - add_numbers: Add numbers mentioned in the query.
     ```
   * The LLM is told: *“If a tool seems useful, call it. Otherwise, answer directly.”*

---

2. **User Query Analysis**

   * The LLM looks at the **user’s request** and matches it to a tool’s description.
   * Query: *“Who created Python?”* → mentions “Python” (matches Wiki’s domain).
   * Query: *“What is 45 plus 55?”* → mentions numbers + math words (matches calculator).

---

3. **Reasoning Step (ReAct pattern)**

   * The LLM doesn’t just blurt an answer. It writes out steps internally like:

     * *“The user asked about Python. I should look it up in the Wiki tool.”*
     * *“The user asked to add numbers. I should call the calculator.”*

---

4. **Tool Invocation**

   * LangChain then runs the chosen tool function with the LLM’s input.
   * The tool returns a result (e.g., `"Python was created by Guido van Rossum in 1991."`).
   * The LLM sees that result and uses it to craft the final answer.

---

## 📌 Key Teaching Line

> *“The LLM doesn’t have hard-coded logic. Instead, it reads the **tool descriptions** in the system prompt, then decides — in natural language — which tool to call. That’s why writing clear tool descriptions is so important.”*

---

## Example Trace (simplified for class)

For: `"What is 45 plus 55?"`

LLM might think:

```
Thought: The user wants a math calculation.
Action: add_numbers
Action Input: "45 plus 55"
Observation: "The sum is 100"
Final Answer: The sum is 100
```

---