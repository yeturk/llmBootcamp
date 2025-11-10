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
from dotenv import load_dotenv
import os
import uuid

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the state of our chatbot (Same as previous lessons)
class State(TypedDict):
    """
    Represents the state of our chatbot with memory.
    
    The `messages` key contains the conversation history.
    The `add_messages` annotation tells LangGraph to append new messages.
    
    With memory, this state persists across multiple invocations!
    """
    messages: Annotated[list, add_messages]

# Initialize LLM
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai", api_key=api_key)

# Initialize tools (same as Lesson 2)
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

def chatbot(state: State):
    """
    The main chatbot node function with memory support.
    
    This function works the same as in Lesson 2, but now the state
    persists across multiple conversations thanks to the checkpointer.
    
    Args:
        state: Current conversation state containing messages
        
    Returns:
        dict: Updated state with new message from LLM
    """
    # Get the LLM's response to the conversation (with memory context)
    response = llm_with_tools.invoke(state["messages"])
    
    # Log what the LLM decided to do
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_names = [tc['name'] for tc in response.tool_calls]
        logger.info(f"🔧 LLM decided to use tools: {', '.join(tool_names)}")
    else:
        logger.info("💭 LLM responded directly (with memory context)")
    
    # Return the response
    return {"messages": [response]}

def route_tools(state: State):
    """
    Router function to decide next step (same as Lesson 2).
    
    Args:
        state: Current conversation state
        
    Returns:
        str: "tools" if LLM wants to use tools, "end" if done
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the LLM makes a tool call, route to tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        logger.info("🔄 Routing to tools...")
        return "tools"
    
    # Otherwise, end the conversation
    logger.info("✅ Conversation step complete")
    return "end"

# Build the graph with memory following official LangGraph tutorial
def build_chatbot_with_memory_graph():
    """
    Creates a chatbot graph with memory support following official LangGraph pattern.
    
    Key addition: InMemorySaver checkpointer for persistent state.
    
    Graph structure: (same as Lesson 2, but with memory)
    START -> chatbot -> [router] -> tools -> chatbot (with persistent state)
                      \\-> END
    
    Returns:
        Compiled LangGraph with memory capabilities
    """
    # Create a StateGraph with our State schema
    graph_builder = StateGraph(State)
    
    # Add the chatbot node
    graph_builder.add_node("chatbot", chatbot)
    
    # Add the tools node if tools are available
    if tools:
        tool_node = ToolNode(tools)
        graph_builder.add_node("tools", tool_node)
    
    # Define the flow: START -> chatbot
    graph_builder.add_edge(START, "chatbot")
    
    # Add conditional edges from chatbot
    if tools:
        graph_builder.add_conditional_edges(
            "chatbot",
            route_tools,
            {
                "tools": "tools",
                "end": END,
            }
        )
        # Tools always go back to chatbot
        graph_builder.add_edge("tools", "chatbot")
    else:
        # No tools available, go directly to END
        graph_builder.add_edge("chatbot", END)
    
    # MEMORY: Add checkpointer for persistent state (Official LangGraph pattern)
    import sqlite3
    checkpointer = SqliteSaver(sqlite3.connect("checkpoint.db", check_same_thread=False))
    
    # Compile the graph WITH the checkpointer
    return graph_builder.compile(checkpointer=checkpointer)

def run_conversation(graph, thread_id: str, message: str):
    """
    Run a single conversation turn with memory.
    
    Args:
        graph: Compiled LangGraph with memory
        thread_id: Unique thread identifier for this conversation
        message: User message to process
        
    Returns:
        Final response from the chatbot
    """
    from langchain_core.messages import HumanMessage
    
    # Create input with user message
    inputs = {"messages": [HumanMessage(content=message)]}
    
    # Configure with thread_id for memory
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"\n👤 User: {message}")
    print("-" * 50)
    
    # Process the message with memory
    logger.info(f"Processing message in thread: {thread_id}")
    
    # Get final result with memory - use invoke instead of stream for simpler debugging
    final_result = graph.invoke(inputs, config)
    
    # Find the final AI response
    ai_messages = [msg for msg in final_result["messages"] if hasattr(msg, 'content') and msg.content]
    if ai_messages:
        final_response = ai_messages[-1].content
    else:
        final_response = "I apologize, but I didn't generate a proper response."
    
    print(f"\n🤖 Chatbot: {final_response}")
    
    return final_response, len(final_result["messages"])

def main():
    parser = argparse.ArgumentParser(description="Chatbot with Memory - Official LangGraph Tutorial")
    parser.add_argument("--message", type=str, required=True,
                       help="Message to send to the chatbot")
    parser.add_argument("--thread", type=str, default=None,
                       help="Thread ID for conversation memory (auto-generated if not provided)")
    parser.add_argument("--interactive", action="store_true",
                       help="Enable interactive conversation mode")
    
    args = parser.parse_args()
    
    # Generate thread ID if not provided
    thread_id = args.thread or str(uuid.uuid4())[:8]
    
    print("\n" + "="*80)
    print("CHATBOT WITH MEMORY - OFFICIAL LANGGRAPH TUTORIAL")
    print("="*80)
    print("Building a chatbot that remembers previous conversations...")
    print(f"Tools available: {len(tools)} ({'Tavily Search' if tools else 'None'})")
    print(f"Thread ID: {thread_id}")
    print("="*80)
    
    try:
        # Build the chatbot graph with memory
        graph = build_chatbot_with_memory_graph()
        
        if args.interactive:
            # Interactive mode
            print("\n🔄 Interactive Mode - Type 'quit' to exit")
            print("=" * 50)
            
            while True:
                user_input = input("\n👤 You: ").strip()
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye!")
                    break
                
                if user_input:
                    response, total_msgs = run_conversation(graph, thread_id, user_input)
                    print(f"\n💾 Memory: {total_msgs} messages in conversation history")
        else:
            # Single message mode
            response, total_msgs = run_conversation(graph, thread_id, args.message)
            
            print("\n" + "="*80)
            print("✅ Conversation with memory complete!")
            print("="*80)
            
            # Show memory info
            print(f"\n📊 Memory Summary:")
            print(f"   Thread ID: {thread_id}")
            print(f"   Total messages in memory: {total_msgs}")
            print(f"   Tools available: {len(tools)}")
            print(f"\n💡 To continue this conversation, use: --thread {thread_id}")
        
        logger.info("Chatbot with memory session completed successfully")
        
    except Exception as e:
        logger.error(f"Error in chatbot with memory: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()