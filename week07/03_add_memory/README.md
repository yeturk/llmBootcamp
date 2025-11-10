# Lesson 3: Add Memory

This lesson follows the official LangGraph tutorial to add persistent memory to our chatbot using checkpointers.

## 🎯 Learning Objectives

- Understand LangGraph's checkpointer system
- Add persistent conversation memory
- Learn thread-based conversation management
- Master state persistence across interactions
- Enable multi-turn conversations with context

## 🔧 Key Concepts

### Checkpointers
LangGraph solves memory limitations through **persistent checkpointing**:
- **SqliteSaver**: For development with true persistence across sessions
- **PostgresSaver**: For production environments
- **Thread-based**: Each conversation has a unique thread_id

### Memory Persistence
When you provide a checkpointer and thread_id:
1. LangGraph automatically saves state after each step
2. Subsequent calls with the same thread_id load the saved state
3. The chatbot remembers previous conversations

### Thread Management
```python
# Each conversation has a unique thread
config = {"configurable": {"thread_id": "conversation-123"}}
graph.invoke(inputs, config)  # Remembers previous messages
```

## 💻 Code Structure

### 1. Checkpointer Setup
```python
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

# Create persistent memory checkpointer
checkpointer = SqliteSaver(sqlite3.connect("checkpoint.db", check_same_thread=False))

# Compile graph with memory
graph = graph_builder.compile(checkpointer=checkpointer)
```

### 2. Thread Configuration
```python
# Configure conversation thread
config = {"configurable": {"thread_id": thread_id}}

# All operations use the same thread for memory
response = graph.invoke(inputs, config)
```

### 3. Persistent State
```python
# State automatically persists between calls
# Each message is remembered in the conversation thread
class State(TypedDict):
    messages: Annotated[list, add_messages]  # Accumulates over time
```

## 🚀 Running the Example

### Single Message Mode
```bash
# First message in a conversation
uv run week_07/03_add_memory/main.py --message "Hi, my name is Alice"

# Continue the same conversation (use the thread ID from output)
uv run week_07/03_add_memory/main.py --message "What's my name?" --thread abc12345
```

### Interactive Mode
```bash
# Multi-turn conversation
uv run week_07/03_add_memory/main.py --message "Hello" --interactive

# Then type multiple messages:
# "My name is Bob"
# "I like Python programming"  
# "What do you know about me?"
```

## 🔍 What You'll See

### Memory in Action
```
Thread ID: abc12345

👤 User: Hi, my name is Alice and I love cooking
🤖 Chatbot: Hello Alice! It's nice to meet you. Cooking is wonderful...

💾 Memory: 2 messages in conversation history

# Later, with the same thread:
👤 User: What's my name and what do I like?
🤖 Chatbot: Your name is Alice and you mentioned that you love cooking!

💾 Memory: 4 messages in conversation history
```

## 📈 Key Features Added

- ✅ **Persistent Memory**: Conversations persist across program restarts using SQLite
- ✅ **Thread Management**: Unique thread IDs for different conversations
- ✅ **Context Awareness**: References previous messages naturally
- ✅ **Interactive Mode**: Multi-turn conversations
- ✅ **Memory Tracking**: Shows conversation history size
- ✅ **Cross-Session Persistence**: Memory survives program exits and restarts

## 🆚 vs Previous Lessons

| Feature | Lesson 1 | Lesson 2 | Lesson 3 |
|---------|----------|----------|----------|
| Memory | None | None | Persistent across calls |
| Context | Single message | Single message | Full conversation history |
| Threads | N/A | N/A | Thread-based conversations |
| Use Case | One-shot questions | One-shot with tools | Multi-turn conversations |

## 🔧 Memory Types

### SqliteSaver (Development & Light Production)
```python
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
checkpointer = SqliteSaver(sqlite3.connect("checkpoint.db", check_same_thread=False))
```
- **Pros**: True persistence across restarts, simple setup, file-based storage
- **Cons**: Single-file limitations for high-scale applications

### PostgresSaver (Production)
```python
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver(connection_string="postgresql://...")
```
- **Pros**: Persistent across restarts, scalable
- **Cons**: Requires database setup

## 💡 Best Practices

1. **Thread Management**: Use meaningful thread IDs (user-id, session-id)
2. **Memory Cleanup**: Implement cleanup for old conversations
3. **Error Handling**: Handle checkpointer failures gracefully
4. **Testing**: Use InMemorySaver for development

## 📈 What's Next?

In the next lesson, we'll add human-in-the-loop controls to our chatbot, allowing for human review and intervention in conversations.

## 🔗 Official Documentation

- [LangGraph Add Memory Tutorial](https://langchain-ai.github.io/langgraph/tutorials/get-started/3-add-memory/)
- [LangGraph Persistence Concepts](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [Checkpointer Documentation](https://langchain-ai.github.io/langgraph/how-tos/memory/add-memory/)

---
**Note**: This lesson builds on Lessons 1 & 2. The chatbot now remembers conversations, making it suitable for real chat applications with ongoing user interactions.

## Postgresql database
```
docker run -d \
  --name postgres-langgraph \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=langgraph_db \
  -p 5432:5432 \
  postgres:16
```