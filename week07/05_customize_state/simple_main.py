import argparse
import logging
from typing import Annotated
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os
import uuid

load_dotenv()

# Simple custom state - just adds a counter
class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_count: int  # Track how many messages we've processed

# Initialize LLM
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai", api_key=api_key)

def chatbot_node(state: State):
    """Simple chatbot that counts messages."""
    # Get current count
    current_count = state.get("message_count", 0)
    
    # Add message count info to the conversation
    count_info = f"(This is message #{current_count + 1} in our conversation)"
    
    # Get response from LLM
    response = llm.invoke(state["messages"])
    
    print(f"📊 Message count: {current_count + 1}")
    
    return {
        "messages": [response],
        "message_count": current_count + 1
    }

def build_graph():
    """Build a simple graph with custom state."""
    graph_builder = StateGraph(State)
    
    # Add single node
    graph_builder.add_node("chatbot", chatbot_node)
    
    # Simple flow: START -> chatbot -> END
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    
    return graph_builder.compile()

def main():
    parser = argparse.ArgumentParser(description="Simple Custom State Example")
    parser.add_argument("--message", type=str, required=True, help="Message to send")
    
    args = parser.parse_args()
    
    print("="*50)
    print("SIMPLE CUSTOM STATE EXAMPLE")
    print("="*50)
    print("Shows how to add custom fields to state")
    
    # Build graph
    graph = build_graph()
    
    # Create input with custom state
    inputs = {
        "messages": [HumanMessage(content=args.message)],
        "message_count": 0  # Initialize our custom field
    }
    
    print(f"\n👤 User: {args.message}")
    
    # Execute
    result = graph.invoke(inputs)
    
    # Show results
    final_message = result["messages"][-1].content
    print(f"🤖 Chatbot: {final_message}")
    print(f"📊 Final message count: {result['message_count']}")

if __name__ == "__main__":
    main()