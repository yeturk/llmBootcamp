# Lesson 5: Customize State

This lesson teaches custom state management in LangGraph through **3 progressive examples** - from simple to advanced. Learn to extend state beyond basic messages and create adaptive, personalized agents.

## 📚 Three Learning Levels

This lesson includes **3 files** for progressive learning:

1. **`simple_main.py`** - Basic custom state concept (70 lines)
2. **`intermediate_main.py`** - User profile tracking + memory (120 lines)  
3. **`main.py`** - Full complexity with modes, routing, tools (400+ lines)

## 🎯 Learning Objectives

- Start simple: Add custom fields to state
- Build up: Track user information and preferences
- Master: Implement behavioral modes and dynamic routing
- Understand progressive complexity in state management

## 🚀 How to Run Each Level

### Level 1: Simple Custom State
**File**: `simple_main.py` (Start here!)
```bash
# Basic example - adds message counter to state
uv run week_07/05_customize_state/simple_main.py --message "Hello!"
```

**What you'll learn**:
- How to extend `TypedDict` with custom fields
- Basic state updates
- Core concept without complexity

### Level 2: Intermediate - User Profiles  
**File**: `intermediate_main.py` (Next step)
```bash
# Tracks user information + memory
uv run week_07/05_customize_state/intermediate_main.py --message "My name is Alice"

# Continue conversation with same thread
uv run week_07/05_customize_state/intermediate_main.py --message "What's my name?" --thread abc123
```

**What you'll learn**:
- User profile extraction and storage
- Memory persistence across conversations
- Personalized responses

### Level 3: Advanced - Full Complexity
**File**: `main.py` (Final challenge)
```bash
# Multiple behavioral modes
uv run week_07/05_customize_state/main.py --message "Tell me about AI" --mode casual
uv run week_07/05_customize_state/main.py --message "Provide analysis" --mode professional

# Dynamic mode switching
uv run week_07/05_customize_state/main.py --message "Switch to technical mode and explain APIs"
```

**What you'll learn**:
- Complex state schemas with multiple fields
- Dynamic routing based on state
- Behavioral mode switching
- Advanced state management patterns

## 🔧 Progressive Complexity

### Level 1: Simple State Schema
```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_count: int  # Just add a counter
```

### Level 2: User Profile State  
```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_count: int
    user_profile: dict  # Store user information
```

### Level 3: Full Complex State
```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
    mode: Literal["casual", "professional", "technical", "creative"]
    user_profile: dict  # User preferences and information
    session_info: dict  # Session metadata
```

## 💻 Code Structure

### 1. Enhanced State Definition
```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
    mode: Literal["casual", "professional", "technical", "creative"]
    user_profile: dict
    session_info: dict
```

### 2. Mode-Aware System Prompts
```python
def get_system_prompt(state: State) -> str:
    mode = state.get("mode", "casual")
    user_profile = state.get("user_profile", {})
    
    # Customize prompt based on mode and user
    return f"You are in {mode} mode with {personality} traits..."
```

### 3. State Management Nodes
```python
def mode_controller(state: State):
    # Detect mode change requests
    if "switch to professional" in content:
        return {"mode": "professional"}

def profile_updater(state: State):
    # Extract and store user information
    if "my name is" in content:
        return {"user_profile": {"name": extracted_name}}
```

### 4. Enhanced Router
```python
def route_with_state(state: State):
    # Route based on content and current state
    if needs_mode_change(state):
        return "mode_control"
    elif needs_tools(state):
        return "tools"
    return "end"
```

## 🔍 What You'll See at Each Level

### Level 1: Simple Counter
```
SIMPLE CUSTOM STATE EXAMPLE
=====================================
Shows how to add custom fields to state

👤 User: Hello!
📊 Message count: 1
🤖 Chatbot: Hello! How can I help you today?
📊 Final message count: 1
```

### Level 2: User Profile Learning
```
INTERMEDIATE CUSTOM STATE EXAMPLE
=============================================
Shows custom state + user profile tracking + memory
Thread ID: abc123ef

👤 User: My name is Alice
📝 Learned your name: Alice
📊 Message #1
👤 Profile: {'name': 'Alice'}

🤖 Chatbot: Nice to meet you, Alice! How can I help you today?

📊 Final state:
   Messages: 1
   Profile: {'name': 'Alice'}
```

### Level 3: Full Behavioral Modes
```
🎭 Mode: casual
👤 User: Tell me about Python
🤖 Chatbot (casual mode): Hey there! 🐍 Python is this awesome programming language...

🎯 FINAL STATE
Mode: casual
User Profile: {"name": "Alice", "interests": ["Python"]}
Session Info: {"thread_id": "abc123"}
```

## 🎓 Teaching Strategy

### Why 3 Files?

**Problem**: The original `main.py` was too complex (400+ lines) for students to understand the core concept.

**Solution**: Progressive learning approach:

1. **`simple_main.py`**: Focus on ONE concept - adding custom fields to state
2. **`intermediate_main.py`**: Build up - add user profiles and memory  
3. **`main.py`**: Full complexity - multiple modes, routing, tools

### Recommended Teaching Flow

1. **Start with `simple_main.py`** - Students understand basic custom state
2. **Move to `intermediate_main.py`** - Add complexity gradually
3. **Finish with `main.py`** - Show full production patterns

This prevents overwhelming students while building confidence step by step.

## 📊 Comparison of the 3 Levels

| Feature | Simple | Intermediate | Advanced |
|---------|--------|-------------|----------|
| **Lines of Code** | 70 | 120 | 400+ |
| **Custom Fields** | 1 (counter) | 2 (counter + profile) | 4 (full state) |
| **Memory** | ❌ | ✅ Persistent | ✅ Persistent |
| **User Learning** | ❌ | ✅ Basic | ✅ Advanced |
| **Behavioral Modes** | ❌ | ❌ | ✅ 4 modes |
| **Dynamic Routing** | ❌ | ❌ | ✅ Complex |
| **Tools Support** | ❌ | ❌ | ✅ Optional |
| **Complexity** | Beginner | Intermediate | Advanced |

## 📈 Key Concepts Learned

### Level 1: Foundation
- ✅ **Custom State Fields**: Extend beyond messages
- ✅ **State Updates**: Modify state in nodes
- ✅ **TypedDict Usage**: Proper type definitions

### Level 2: Building Up  
- ✅ **User Profile Tracking**: Extract and store user info
- ✅ **Memory Persistence**: State across conversations
- ✅ **Personalization**: Adapt responses to user

### Level 3: Full Power
- ✅ **Multiple Personalities**: 4 distinct behavioral modes
- ✅ **Dynamic Mode Switching**: Change personality mid-conversation
- ✅ **Complex Routing**: State-based decision making
- ✅ **Production Patterns**: Real-world agent architecture

## 📈 What's Next?

This completes the official LangGraph learning path! You now have:
1. ✅ Basic chatbot foundation
2. ✅ Tool integration capabilities  
3. ✅ Persistent memory
4. ✅ Human oversight controls
5. ✅ Custom behavioral states

Next steps: Build production applications combining all these patterns!

## 🔗 Official Documentation

- [LangGraph Customize State Tutorial](https://langchain-ai.github.io/langgraph/tutorials/get-started/5-customize-state/)
- [LangGraph State Management](https://langchain-ai.github.io/langgraph/concepts/low_level/#state)
- [Advanced State Patterns](https://langchain-ai.github.io/langgraph/how-tos/state-management/)

---
**Note**: This lesson demonstrates the full power of LangGraph's state management. You can now build sophisticated, adaptive agents that learn and adjust their behavior based on user preferences and context.