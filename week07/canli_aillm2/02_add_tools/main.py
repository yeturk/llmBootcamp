import argparse
import logging
from typing import Annotated
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import os

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the state of our chatbot (Same as Lesson 1)
class State(TypedDict):
    """
    Represents the state of our chatbot with tools.
    
    The `messages` key contains the conversation history.
    The `add_messages` annotation tells LangGraph to append new messages.
    """
    messages: Annotated[list, add_messages]

# Initialize LLM
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai", api_key=api_key)

tools = []

# Add Tavily search tool if available
tavily_api_key = os.getenv("TAVILY_API_KEY")
if tavily_api_key:
    search_tool = TavilySearch(
        max_results=2,
        description="Search the web for current information, news, and real-time data"
    )
    tools.append(search_tool)
    logger.info("Tavily search tool added")
else:
    logger.warning("TAVILY_API_KEY not found - search functionality disabled")

# Bind tools to the model (Official LangGraph pattern)
llm_with_tools = llm.bind_tools(tools)

def route_tools(state: State):
    """
    Router function to decide next step (Official LangGraph pattern).
    
    This function examines the last message to determine if the LLM
    wants to use tools or if we should end the conversation.
    
    Args:
        state: Current conversation state
        
    Returns:
        str: "tools" if LLM wants to use tools, "end" if done
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the LLM makes a tool call, route to tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        logger.info("Routing to tools...")
        return "tools"
    
    # Otherwise, end the conversation
    logger.info("Conversation complete")
    return "end"


def chatbot(state: State):
    """
    The main chatbot node function with tool support.
    
    This function:
    1. Takes the current state as input
    2. Invokes the LLM with tools bound to it
    3. Returns the updated state with the LLM's response
    
    The LLM can now decide to use tools or respond directly.
    
    Args:
        state: Current conversation state containing messages
        
    Returns:
        dict: Updated state with new message from LLM
    """
    # Get the LLM's response to the conversation (with tool access)
    response = llm_with_tools.invoke(state["messages"])
    
    # Return the updated state with the new message
    return {"messages": [response]}


def build_chatbot_with_tools_graph():
    """
    Creates a chatbot graph with tool support following official LangGraph pattern.
    
    Graph structure:
    START -> chatbot -> [router] -> tools -> chatbot (loop until no tools needed)
                      \\-> END (if no tools needed)
    
    Returns:
        Compiled LangGraph that can process conversations with tools
    """

    # Create a StateGraph with our State schema
    graph_builder = StateGraph(State)

    # Add the chatbot node
    graph_builder.add_node("chatbot", chatbot)

    # Add the tools node (official prebuilt ToolNode)
    if tools:
        tool_node = ToolNode(tools)
        graph_builder.add_node("tools", tool_node)
    
    # Define the flow: START -> chatbot
    graph_builder.add_edge(START, "chatbot")

    # Add conditional edges from chatbot (official pattern)
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

    # Compile the graph
    return graph_builder.compile()

def main():
    parser = argparse.ArgumentParser(description="Chatbot with Tools - Official LangGraph Tutorial")
    parser.add_argument("--message", type=str, required=True,
                       help="Message to send to the chatbot")
    
    args = parser.parse_args()

    try:
        # Build the chatbot graph with tools
        graph = build_chatbot_with_tools_graph()
        # Create the initial state with user message
        from langchain_core.messages import HumanMessage
        initial_state = {
            "messages": [HumanMessage(content=args.message)]
        }

        # # Get final result
        final_result = graph.invoke(initial_state)
        #final_response = final_result["messages"][-1].content
        
        # Show messages
        for msg in final_result["messages"]:
            print(msg)



    except Exception as e:
        logger.error(f"Error in chatbot with tools: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()