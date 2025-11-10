import argparse
import logging
from typing import Annotated
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os
import uuid
import sqlite3

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the state
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize LLM
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai", api_key=api_key)

# Initialize tools
tools = []

# Add Tavily search tool if available
tavily_api_key = os.getenv("TAVILY_API_KEY")
if tavily_api_key:
    search_tool = TavilySearch(
        max_results=2,
        description="Search the web for current information, news, and real-time data"
    )
    tools.append(search_tool)
    logger.info("✅ Tavily search tool added")
else:
    logger.warning("⚠️ TAVILY_API_KEY not found - search functionality disabled")

# Bind tools to the model
llm_with_tools = llm.bind_tools(tools)

def chatbot_node(state: State):
    """Main chatbot node that responds to user messages."""
    response = llm_with_tools.invoke(state["messages"])
    
    # Log the decision
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_names = [tc['name'] for tc in response.tool_calls]
        logger.info(f"🤖 LLM wants to use tools: {', '.join(tool_names)}")
    else:
        logger.info("🤖 LLM responded directly")
    
    return {"messages": [response]}

def route_tools(state: State):
    """Router function to decide next step."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If LLM decided to use tools, route to tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        logger.info("🔄 Routing to tools...")
        return "tools"
    
    # Otherwise end the conversation
    logger.info("✅ Conversation complete")
    return "end"

def build_graph():
    """Build a simple chatbot graph with memory for time travel."""
    graph_builder = StateGraph(State)
    
    # Add nodes
    graph_builder.add_node("chatbot", chatbot_node)
    
    if tools:
        tool_node = ToolNode(tools)
        graph_builder.add_node("tools", tool_node)
    
    # Define edges
    graph_builder.add_edge(START, "chatbot")
    
    if tools:
        graph_builder.add_conditional_edges(
            "chatbot",
            route_tools,
            {
                "tools": "tools",
                "end": END,
            }
        )
        graph_builder.add_edge("tools", "chatbot")
    else:
        graph_builder.add_edge("chatbot", END)
    
    # Add memory for time travel capability
    checkpointer = SqliteSaver(sqlite3.connect("checkpoint.db", check_same_thread=False))
    
    return graph_builder.compile(checkpointer=checkpointer)

def show_state_history(graph, config):
    """Display the complete state history for time travel exploration."""
    print("\n" + "="*80)
    print("🕐 TIME TRAVEL: STATE HISTORY")
    print("="*80)
    
    states = list(graph.get_state_history(config))
    
    if not states:
        print("❌ No state history found for this thread")
        return states
    
    print(f"Found {len(states)} states in history:")
    print("-" * 50)
    
    for i, state in enumerate(states):
        messages = state.values.get("messages", [])
        print(f"State {i + 1}:")
        print(f"  📊 Messages: {len(messages)}")
        print(f"  🔗 Config: {state.config}")
        
        # Show the last message content if available
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, 'content'):
                content_preview = last_msg.content[:50] + "..." if len(last_msg.content) > 50 else last_msg.content
                msg_type = "👤 Human" if hasattr(last_msg, 'type') and last_msg.type == 'human' else "🤖 AI"
                print(f"  💬 Last: {msg_type}: {content_preview}")
        print()
    
    return states

def demonstrate_time_travel(graph, config):
    """Demonstrate time travel capabilities."""
    print("\n" + "="*80)
    print("🕐 TIME TRAVEL DEMONSTRATION")
    print("="*80)
    
    # Get state history
    states = show_state_history(graph, config)
    
    if len(states) < 2:
        print("⚠️ Need at least 2 states for time travel demonstration")
        return
    
    # Demonstrate loading a previous state
    print("🔄 LOADING PREVIOUS STATE...")
    print("-" * 50)
    
    # Get a state from earlier in the conversation
    previous_state = states[1] if len(states) > 1 else states[0]
    
    print(f"Loading state with {len(previous_state.values['messages'])} messages")
    
    # Get the current state at that point
    loaded_state = graph.get_state(previous_state.config)
    print(f"✅ Successfully loaded state from checkpoint")
    print(f"📊 Loaded state has {len(loaded_state.values['messages'])} messages")
    
    # Show what we could do from this point
    print("\n💡 FROM THIS POINT, YOU COULD:")
    print("   • Continue the conversation with new input")
    print("   • Fork the conversation in a different direction") 
    print("   • Analyze what happened at this specific moment")
    print("   • Debug agent behavior step by step")

def main():
    parser = argparse.ArgumentParser(description="LangGraph Time Travel Tutorial")
    parser.add_argument("--message", type=str, default=None,
                       help="Message to send to the chatbot")
    parser.add_argument("--thread", type=str, default=None,
                       help="Thread ID for conversation memory")
    parser.add_argument("--show-history", action="store_true",
                       help="Show state history for time travel")
    parser.add_argument("--time-travel", action="store_true",
                       help="Demonstrate time travel capabilities")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.show_history and not args.time_travel and not args.message:
        parser.error("--message is required unless using --show-history or --time-travel")
    
    # Generate thread ID if not provided
    thread_id = args.thread or str(uuid.uuid4())[:8]
    
    print("\n" + "="*80)
    print("LANGGRAPH TIME TRAVEL TUTORIAL")
    print("="*80)
    print("Learn to navigate through conversation history and explore alternative paths")
    print(f"Thread ID: {thread_id}")
    print(f"Tools available: {len(tools)} ({'Tavily Search' if tools else 'None'})")
    print("="*80)
    
    try:
        # Build the graph
        graph = build_graph()
        
        # Configure with thread_id for memory
        config = {"configurable": {"thread_id": thread_id}}
        
        # Show history if requested
        if args.show_history:
            show_state_history(graph, config)
            return
        
        # Demonstrate time travel if requested
        if args.time_travel:
            demonstrate_time_travel(graph, config)
            return
        
        # Normal conversation
        print(f"\n👤 User: {args.message}")
        print("-" * 50)
        
        # Process the message
        inputs = {"messages": [HumanMessage(content=args.message)]}
        
        logger.info(f"Processing message in thread: {thread_id}")
        
        # Execute the graph
        result = graph.invoke(inputs, config)
        
        # Extract and display the response
        final_messages = result["messages"]
        
        # Find the AI response
        ai_response = None
        for msg in reversed(final_messages):
            if hasattr(msg, 'content') and hasattr(msg, 'type') and msg.type != 'human':
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    continue  # Skip tool calls, look for content response
                ai_response = msg.content
                break
        
        if ai_response:
            print(f"\n🤖 Chatbot: {ai_response}")
        else:
            print(f"\n🤖 Chatbot processed your message (used tools)")
        
        print("\n" + "="*80)
        print("✅ Conversation complete!")
        print("="*80)
        
        # Show time travel options
        print(f"\n🕐 TIME TRAVEL OPTIONS:")
        print(f"   • --show-history --thread {thread_id}")
        print(f"   • --time-travel --thread {thread_id}")
        print(f"   • Continue: --message 'new message' --thread {thread_id}")
        
        logger.info("Time travel tutorial session completed successfully")
        
    except Exception as e:
        logger.error(f"Error in time travel tutorial: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()