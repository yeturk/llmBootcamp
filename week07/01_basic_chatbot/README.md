# Lesson 1: Build a Basic Chatbot

This lesson follows the official LangGraph tutorial to build a basic chatbot using StateGraph, nodes, and edges.

## Learning Objectives

- Understand LangGraph's StateGraph architecture
- Learn about nodes, edges, and state management
- Build your first production-ready chatbot
- Master the official LangGraph development pattern

## Key Concepts

### StateGraph
LangGraph uses a `StateGraph` to define the structure of your chatbot as a "state machine". It consists of:
- **Nodes**: Functions that perform actions (like calling the LLM)
- **Edges**: Connections that define the flow between nodes
- **State**: The data that flows through the graph

### State Management
Our chatbot state contains:
```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
```

The `add_messages` annotation tells LangGraph to append new messages rather than overwrite the entire conversation history. Other annotations: 
```
- add_messages → append chat history

- replace → overwrite

- concat → merge lists/strings

- collect → accumulate values

- custom reducer → anything you want
```
### Graph Structure
```
START → chatbot → END
```

This simple flow processes user messages through the LLM and returns responses.

## Code Structure

### 1. State Definition
```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
```

### 2. Chatbot Node Function
```python
def chatbot(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}
```

### 3. Graph Building
```python
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()
```

## Running the Example

```bash
# Basic conversation
uv run week_07/01_basic_chatbot/main.py --message "Hello! Can you help me with Python?"

# Ask for coding help
uv run week_07/01_basic_chatbot/main.py --message "Explain how decorators work in Python"

# General questions
uv run week_07/01_basic_chatbot/main.py --message "What's the difference between lists and tuples?"
```

## 🔍 What You'll See

The chatbot will:
1. Process your message through the LangGraph StateGraph
2. Show your input message
3. Display the chatbot's response
4. Show the complete conversation state with both messages

Example output:
```
================================================================================
BASIC CHATBOT - OFFICIAL LANGGRAPH TUTORIAL
================================================================================
👤 User: Hello! Can you help me with Python?
--------------------------------------------------
🤖 Chatbot: Of course! I'd be happy to help you with Python...

📝 Complete Conversation State:
1. 👤 User: Hello! Can you help me with Python?...
2. 🤖 Chatbot: Of course! I'd be happy to help you with Python...
```

## 📈 What's Next?

In the next lesson, we'll add tools to our chatbot so it can search the web for current information it doesn't know.

## 🔗 Official Documentation

- [LangGraph Basic Chatbot Tutorial](https://langchain-ai.github.io/langgraph/tutorials/get-started/1-build-basic-chatbot/)
- [LangGraph StateGraph Documentation](https://langchain-ai.github.io/langgraph/concepts/low_level/)
- [LangGraph Concepts](https://langchain-ai.github.io/langgraph/concepts/why-langgraph/)

---
**Note**: This is the foundation lesson. Each subsequent lesson builds upon this basic chatbot structure, adding tools, memory, human-in-the-loop controls, and more advanced features.