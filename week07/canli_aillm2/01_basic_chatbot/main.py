import argparse
import logging
from typing import Annotated
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the state of our chatbot (Official LangGraph Pattern)
class State(TypedDict):
    """
    Represents the state of our chatbot.
    
    The `messages` key contains the conversation history.
    The `add_messages` annotation tells LangGraph to append new messages
    to the list instead of overwriting the entire list.
    """
    messages: Annotated[list, add_messages]

# Initialize LLM
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai", api_key=api_key)

def chatbot(state: State):
    """
    The main chatbot node function.
    
    This function:
    1. Takes the current state as input
    2. Invokes the LLM with the conversation messages
    3. Returns the updated state with the LLM's response
    
    Args:
        state: Current conversation state containing messages
        
    Returns:
        dict: Updated state with new message from LLM
    """
    # Get the LLM's response to the conversation
    response = llm.invoke(state["messages"])
    
    # Return the response in the format expected by our state
    return {"messages": [response]}

def build_chatbot_graph():
    """
    Creates the basic chatbot graph with official LangGraph pattern.
    
    Graph structure:
    START -> chatbot -> END
    
    Returns:
        Compiled LangGraph that can process conversations
    """
    # Create a StateGraph with our State schema
    graph_builder = StateGraph(State)

    # Add the chatbot node
    graph_builder.add_node("chatbot", chatbot)

    # Start'ı chatbota bağla
    graph_builder.add_edge(START, "chatbot")

    # Chatbotu END'e bağla
    graph_builder.add_edge("chatbot", END)

    # Compile the graph
    return graph_builder.compile()

def main():
    parser = argparse.ArgumentParser(description="Basic Chatbot - Official LangGraph Tutorial")
    parser.add_argument("--message", type=str, required=True,
                       help="Message to send to the chatbot")
    
    args = parser.parse_args()

    print("\n" + "="*80)
    print("BASIC CHATBOT - OFFICIAL LANGGRAPH TUTORIAL")
    print("="*80)
    print("Building a basic chatbot using LangGraph StateGraph...")
    print("="*80)

    try:
        # Build the chatbot graph
        graph = build_chatbot_graph()

         # Create the initial state with user message
        from langchain_core.messages import HumanMessage
        initial_state = {
            "messages": [HumanMessage(content=args.message)]
        }

        print(f"\n User: {args.message}")
        print("-" * 50)
        
        # Run the chatbot
        logger.info("Processing message through chatbot graph...")
        
        # Invoke the graph with our initial state
        result = graph.invoke(initial_state)
         # Extract the chatbot's response (last message in the conversation)
        chatbot_response = result["messages"][-1].content

        # Iterate through all messages in the conversation history
        for i, msg in enumerate(result["messages"], 1):
            # Determine the role based on message type (human/user vs assistant/chatbot)
            if hasattr(msg, 'type'):
                role = "User" if msg.type == "human" else "Chatbot"
            else:
                role = "Chatbot"
            # Extract message content safely and truncate to first 100 characters for display
            content = getattr(msg, 'content', str(msg))
            print(f"{i}. {role}: {content[:100]}...")
        
        logger.info("Chatbot session completed successfully")
    except Exception as e:
        logger.error(f"Error in chatbot: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
