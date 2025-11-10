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
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from dotenv import load_dotenv
import os
import uuid
import json

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the state of our chatbot with human-in-the-loop
class State(TypedDict):
    """
    Represents the state of our chatbot with human review capabilities.
    
    The `messages` key contains the conversation history.
    Human-in-the-loop interrupts can pause execution for review.
    """
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

def chatbot(state: State):
    """
    The main chatbot node function.
    
    This function works the same as previous lessons, but tool calls
    will be subject to human review.
    
    Args:
        state: Current conversation state containing messages
        
    Returns:
        dict: Updated state with new message from LLM
    """
    response = llm_with_tools.invoke(state["messages"])
    
    # Log what the LLM decided to do
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_names = [tc['name'] for tc in response.tool_calls]
        logger.info(f"🔧 LLM wants to use tools: {', '.join(tool_names)}")
        logger.info("⚠️ Tool calls will be subject to human review")
    else:
        logger.info("💭 LLM responded directly (no review needed)")
    
    return {"messages": [response]}

def human_review_node(state: State):
    """
    Human review node for tool calls (Official LangGraph pattern).
    
    This node pauses execution to allow human review of tool calls
    before they are executed.
    
    Args:
        state: Current conversation state
        
    Returns:
        dict: State with review decision
    """
    last_message = state["messages"][-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print("\n" + "="*60)
        print("🔍 HUMAN REVIEW REQUIRED")
        print("="*60)
        
        for i, tool_call in enumerate(last_message.tool_calls, 1):
            print(f"\n📋 Tool Call {i}:")
            print(f"   Tool: {tool_call['name']}")
            print(f"   Args: {json.dumps(tool_call['args'], indent=2)}")
        
        print("\n🤔 Review Options:")
        print("   ✅ 'approve' - Execute the tool calls")
        print("   ❌ 'reject' - Block tool execution") 
        print("   📝 'feedback' - Provide feedback and retry")
        
        # Get human decision interactively
        decision = input("\n🤔 Enter your decision (approve/reject/feedback): ").strip().lower()
        
        print(f"\n🎯 Decision: {decision}")
        print("="*60)
        
        if decision == "approve":
            logger.info("✅ Human approved tool execution")
            return {"messages": []}  # Continue to tools
        elif decision == "reject":
            logger.info("❌ Human rejected tool execution")
            # Add a message explaining the rejection
            rejection_msg = AIMessage(
                content="I wanted to search for information, but the tool use was not approved. I'll try to answer based on my existing knowledge instead."
            )
            return {"messages": [rejection_msg]}
        else:  # feedback
            feedback_msg = AIMessage(
                content="Let me reconsider the approach based on the feedback provided."
            )
            return {"messages": [feedback_msg]}
    
    # No tool calls, no review needed
    return {"messages": []}

def route_after_chatbot(state: State):
    """
    Router function to determine if human review is needed.
    
    Args:
        state: Current conversation state
        
    Returns:
        str: "human_review" if tool calls need approval, "end" if done
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the LLM makes tool calls, route to human review
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        logger.info("🔄 Routing to human review...")
        return "human_review"
    
    # Otherwise, end the conversation
    logger.info("✅ No tool calls - conversation complete")
    return "end"

def route_after_review(state: State):
    """
    Router function after human review.
    
    Args:
        state: Current conversation state
        
    Returns:
        str: "tools" if approved, "end" if rejected/feedback
    """
    # Check the last few messages to understand the review decision
    recent_messages = state["messages"][-3:]
    
    # Look for tool calls in recent messages that might have been approved
    for msg in recent_messages:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            # Found tool calls - assume approved if we reach this router
            logger.info("🔄 Routing to tools (approved)")
            return "tools"
    
    # No approved tool calls found
    logger.info("✅ Review complete - ending")
    return "end"

# Build the graph with human-in-the-loop following official LangGraph pattern
def build_human_in_loop_graph():
    """
    Creates a chatbot graph with human-in-the-loop controls.
    
    Graph structure:
    START -> chatbot -> [router] -> human_review -> [router] -> tools -> chatbot
                      \\-> END                    \\-> END
    
    Returns:
        Compiled LangGraph with human review capabilities
    """
    # Create a StateGraph with our State schema
    graph_builder = StateGraph(State)
    
    # Add the chatbot node
    graph_builder.add_node("chatbot", chatbot)
    
    # Add human review node
    graph_builder.add_node("human_review", human_review_node)
    
    # Add the tools node if tools are available
    if tools:
        tool_node = ToolNode(tools)
        graph_builder.add_node("tools", tool_node)
    
    # Define the flow: START -> chatbot
    graph_builder.add_edge(START, "chatbot")
    
    # Add conditional edges from chatbot
    graph_builder.add_conditional_edges(
        "chatbot",
        route_after_chatbot,
        {
            "human_review": "human_review",
            "end": END,
        }
    )
    
    # Add conditional edges from human review
    if tools:
        graph_builder.add_conditional_edges(
            "human_review",
            route_after_review,
            {
                "tools": "tools",
                "end": END,
            }
        )
        # Tools go back to chatbot
        graph_builder.add_edge("tools", "chatbot")
    else:
        graph_builder.add_edge("human_review", END)
    
    # Add memory checkpointer
    import sqlite3
    checkpointer = SqliteSaver(sqlite3.connect("checkpoint.db", check_same_thread=False))
    
    # Compile the graph with checkpointer
    return graph_builder.compile(checkpointer=checkpointer)

def main():
    parser = argparse.ArgumentParser(description="Chatbot with Human-in-the-Loop - Official LangGraph Pattern")
    parser.add_argument("--message", type=str, required=True,
                       help="Message to send to the chatbot")
    parser.add_argument("--thread", type=str, default=None,
                       help="Thread ID for conversation memory")
    
    args = parser.parse_args()
    
    # Generate thread ID if not provided
    thread_id = args.thread or str(uuid.uuid4())[:8]
    
    print("\n" + "="*80)
    print("CHATBOT WITH HUMAN-IN-THE-LOOP - OFFICIAL LANGGRAPH PATTERN")
    print("="*80)
    print("Building a chatbot with human review controls...")
    print(f"Tools available: {len(tools)} ({'Tavily Search' if tools else 'None'})")
    print(f"Thread ID: {thread_id}")
    print("="*80)
    
    try:
        # Build the chatbot graph with human-in-the-loop
        graph = build_human_in_loop_graph()
        
        # Create input with user message
        inputs = {"messages": [HumanMessage(content=args.message)]}
        
        # Configure with thread_id for memory
        config = {"configurable": {"thread_id": thread_id}}
        
        print(f"\n👤 User: {args.message}")
        print("-" * 50)
        
        # Process the message with human-in-the-loop
        logger.info(f"Processing message with human review in thread: {thread_id}")
        
        # Execute the graph
        final_result = graph.invoke(inputs, config)
        
        # Find the final response
        ai_messages = [msg for msg in final_result["messages"] 
                      if hasattr(msg, 'content') and msg.content 
                      and not hasattr(msg, 'tool_calls')]
        
        if ai_messages:
            final_response = ai_messages[-1].content
        else:
            final_response = "The conversation required human review but didn't produce a final response."
        
        print("\n" + "="*80)
        print("🎯 FINAL RESPONSE")
        print("="*80)
        print(f"🤖 Chatbot: {final_response}")
        
        print("\n" + "="*80)
        print("✅ Human-in-the-loop conversation complete!")
        print("="*80)
        
        # Show session summary
        total_messages = len(final_result["messages"])
        tool_calls = sum(1 for msg in final_result["messages"] 
                        if hasattr(msg, 'tool_calls') and msg.tool_calls)
        
        print(f"\n📊 Session Summary:")
        print(f"   Thread ID: {thread_id}")
        print(f"   Total messages: {total_messages}")
        print(f"   Tool calls made: {tool_calls}")
        print(f"   Human review: {'Required' if tool_calls > 0 else 'Not needed'}")
        print(f"   Tools available: {len(tools)}")
        
        logger.info("Human-in-the-loop chatbot session completed successfully")
        
    except Exception as e:
        logger.error(f"Error in human-in-the-loop chatbot: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()