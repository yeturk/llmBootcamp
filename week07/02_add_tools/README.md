# Lesson 2: Add Tools

This lesson follows the official LangGraph tutorial to add tools to our basic chatbot, enabling web search capabilities.

## 🎯 Learning Objectives

- Add tools to a LangGraph chatbot
- Understand conditional edges and routing
- Learn the ToolNode pattern
- Enable web search for current information
- Master tool integration with LLMs

## 🔧 Key Concepts

### Tools Integration
LangGraph makes it easy to add tools to your chatbot:
- **Bind tools to LLM**: `llm.bind_tools(tools)` gives the LLM access to tools
- **ToolNode**: Official prebuilt node for executing tools
- **Conditional routing**: Let the LLM decide when to use tools

### Conditional Edges
Instead of fixed edges, we use conditional edges that route based on the LLM's decision:
```python
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,  # Router function
    {
        "tools": "tools",  # If LLM wants tools
        "end": END,        # If no tools needed
    }
)
```

### Router Function
The `route_tools` function examines the LLM's response:
- If it contains `tool_calls` → route to tools
- If no tool calls → end conversation

## 💻 Code Structure

### 1. Tool Setup
```python
# Add Tavily search tool
search_tool = TavilySearch(
    max_results=2,
    description="Search the web for current information"
)
tools = [search_tool]

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)
```

### 2. Chatbot Node (Enhanced)
```python
def chatbot(state: State):
    # LLM can now decide to use tools
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}
```

### 3. Router Function
```python
def route_tools(state: State):
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return "end"
```

### 4. Graph with Tools
```python
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_conditional_edges("chatbot", route_tools, {...})
graph_builder.add_edge("tools", "chatbot")  # Tools → back to chatbot
```

## 🚀 Running the Example

```bash
# Questions that need current information (will use search)
uv run week_07/02_add_tools/main.py --message "What are the latest developments in AI this week?"

# Current events
uv run week_07/02_add_tools/main.py --message "What's happening in the stock market today?"

# Questions that don't need search (will answer directly)
uv run week_07/02_add_tools/main.py --message "Explain how Python decorators work"

# Weather information
uv run week_07/02_add_tools/main.py --message "What's the weather like in Tokyo right now?"
```

## 🔍 What You'll See

The chatbot will intelligently decide whether to use tools:

### With Tools (Current Information)
```
👤 User: What are the latest AI developments?
🤖 Chatbot: Planning to use 1 tool(s)
🛠️ Tools: Executing search...
   📋 Result: Recent AI developments include...
🤖 Chatbot: Based on the latest information, here are the key AI developments...
```

### Without Tools (General Knowledge)
```
👤 User: Explain Python decorators
🤖 Chatbot: Python decorators are a powerful feature that allows...
```

## 📈 Key Features Added

- ✅ **Web Search Integration**: Tavily search for current information
- ✅ **Conditional Routing**: LLM decides when to use tools
- ✅ **Tool Execution**: Automatic tool calling and result processing
- ✅ **Smart Decision Making**: Uses tools only when needed
- ✅ **Processing Transparency**: Shows the tool usage process

## 🆚 vs Lesson 1

| Feature | Lesson 1 | Lesson 2 |
|---------|----------|----------|
| Knowledge | Fixed training data | Current web information |
| Graph flow | Simple: START→chatbot→END | Conditional: chatbot⇄tools |
| Capabilities | General knowledge only | Current events, real-time data |
| Complexity | Basic StateGraph | Conditional edges + ToolNode |

## 📈 What's Next?

In the next lesson, we'll add memory to our chatbot so it can remember previous conversations and maintain context across multiple interactions.

## 🔗 Official Documentation

- [LangGraph Add Tools Tutorial](https://langchain-ai.github.io/langgraph/tutorials/get-started/2-add-tools/)
- [LangGraph ToolNode Documentation](https://langchain-ai.github.io/langgraph/reference/prebuilt/)
- [Tavily Search Integration](https://langchain-ai.github.io/langgraph/concepts/tools/)

---
**Note**: This lesson builds directly on Lesson 1. The chatbot can now access current information while maintaining the same clean architecture.