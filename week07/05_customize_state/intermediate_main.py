import argparse
from typing import Annotated
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
import uuid
import sqlite3

load_dotenv()

# Intermediate custom state - adds user profile tracking
class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_count: int
    user_profile: dict  # Store user information

# Initialize LLM
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai", api_key=api_key)

def extract_user_info(state: State):
    """Extract user information from the last message."""
    last_message = state["messages"][-1]
    current_profile = state.get("user_profile", {})
    
    if hasattr(last_message, 'content'):
        content = last_message.content.lower()
        
        # Extract name
        if "my name is" in content:
            parts = content.split("my name is")
            if len(parts) > 1:
                name = parts[1].split()[0].strip(".,!?").title()
                current_profile["name"] = name
                print(f"📝 Learned your name: {name}")
        
        # Extract interests
        if "i like" in content or "i love" in content:
            if "python" in content:
                interests = current_profile.get("interests", [])
                if "Python" not in interests:
                    interests.append("Python")
                    current_profile["interests"] = interests
                    print(f"📝 Added interest: Python")
    
    return current_profile

def chatbot_node(state: State):
    """Chatbot that tracks messages and user profile."""
    # Update user profile
    user_profile = extract_user_info(state)
    
    # Get current count
    current_count = state.get("message_count", 0)
    
    # Create personalized system message
    system_content = "You are a helpful assistant."
    if user_profile.get("name"):
        system_content += f" The user's name is {user_profile['name']}."
    if user_profile.get("interests"):
        interests = ", ".join(user_profile['interests'])
        system_content += f" They are interested in: {interests}."
    
    # Prepare messages with system prompt
    messages = [SystemMessage(content=system_content)] + state["messages"]
    
    # Get response from LLM
    response = llm.invoke(messages)
    
    print(f"📊 Message #{current_count + 1}")
    if user_profile:
        print(f"👤 Profile: {user_profile}")
    
    return {
        "messages": [response],
        "message_count": current_count + 1,
        "user_profile": user_profile
    }

def build_graph():
    """Build graph with memory for persistent state."""
    graph_builder = StateGraph(State)
    
    # Add single node
    graph_builder.add_node("chatbot", chatbot_node)
    
    # Simple flow
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    
    # Add memory for persistence across sessions
    checkpointer = SqliteSaver(sqlite3.connect("checkpoint.db", check_same_thread=False))
    return graph_builder.compile(checkpointer=checkpointer)

def main():
    parser = argparse.ArgumentParser(description="Intermediate Custom State Example")
    parser.add_argument("--message", type=str, required=True, help="Message to send")
    parser.add_argument("--thread", type=str, default=None, help="Thread ID for memory")
    
    args = parser.parse_args()
    
    # Generate thread ID if not provided
    thread_id = args.thread or str(uuid.uuid4())[:8]
    
    print("="*60)
    print("INTERMEDIATE CUSTOM STATE EXAMPLE")
    print("="*60)
    print("Shows custom state + user profile tracking + memory")
    print(f"Thread ID: {thread_id}")
    
    # Build graph
    graph = build_graph()
    
    # Create input - only pass the new message, let memory handle the rest
    inputs = {
        "messages": [HumanMessage(content=args.message)]
    }
    
    # Configure with thread_id for memory
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"\n👤 User: {args.message}")
    
    # Execute
    result = graph.invoke(inputs, config)
    
    # Show results
    final_message = result["messages"][-1].content
    print(f"\n🤖 Chatbot: {final_message}")
    print(f"\n📊 Final state:")
    print(f"   Messages: {result['message_count']}")
    print(f"   Profile: {result.get('user_profile', {})}")
    
    print(f"\n💡 Try: 'My name is Alice' or 'I love Python'")
    print(f"   Then use --thread {thread_id} to continue conversation")

if __name__ == "__main__":
    main()