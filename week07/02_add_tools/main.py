import argparse
import logging
from typing import Annotated
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode     # Araç (tool) çalıştırmayı sağlayan hazır bir node sınıfı
from langchain_tavily import TavilySearch   # Web araması yapan Tavily API entegrasyonu
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

# Initialize tools following official LangGraph tutorial
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
# bind_tools() → LLM’e, hangi araçları kullanabileceğini bildirir.
# LLM: “Ben sadece cevap üretebilirim ya da Tavily Search kullanarak güncel bilgi toplayabilirim.”


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
    
    # Log what the LLM decided to do
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_names = [tc['name'] for tc in response.tool_calls]
        logger.info(f"LLM decided to use tools: {', '.join(tool_names)}")
    else:
        logger.info("LLM responded directly (no tools needed)")
    
    # Return the response in the format expected by our state
    return {"messages": [response]}

def route_tools(state: State):  # Son mesaja bakar, nereye yönleneceğine karar verir
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

# Build the graph with tools following official LangGraph tutorial
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
    # Eğer .env içinde TAVILY_API_KEY varsa, tools listesi doludur.
    # ToolNode LangGraph’in hazır (prebuilt) bir node’udur; 
    # tool çalıştırma işini otomatik yönetir.
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
    # compile() LangGraph’in “planlama” aşamasıdır.
    # Node’lar, kenarlar ve koşullar optimize edilir.
    # Sonuç: çalıştırılabilir bir graph nesnesi (CompiledGraph).

def main():
    parser = argparse.ArgumentParser(description="Chatbot with Tools - Official LangGraph Tutorial")
    parser.add_argument("--message", type=str, required=True,
                       help="Message to send to the chatbot")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("CHATBOT WITH TOOLS - OFFICIAL LANGGRAPH TUTORIAL")
    print("="*80)
    print("Building a chatbot that can search the web when needed...")
    print(f"Tools available: {len(tools)} ({'Tavily Search' if tools else 'None'})")
    print("="*80)
    
    try:
        # Build the chatbot graph with tools
        graph = build_chatbot_with_tools_graph()
        
        # Create the initial state with user message
        from langchain_core.messages import HumanMessage
        initial_state = {
            "messages": [HumanMessage(content=args.message)]
        }
        
        print(f"\n👤 User: {args.message}")
        print("-" * 50)
        
        # Run the chatbot with tools
        logger.info("Processing message through chatbot with tools graph...")
        
        # Stream through the graph to see the tool usage process
        print("\nProcessing Steps:")
        print("-" * 30)
        
        for step in graph.stream(initial_state):
            # 🔹 graph.stream() nedir? LangGraph’te iki çalışma modu vardır:
              # 1. .invoke() → tamamını çalıştırır, sadece sonucu döner.
              # 2. .stream() → akışı adım adım (node bazlı) takip etmene izin verir.
            for key, value in step.items():
                if key == "chatbot":
                    msg = value["messages"][-1]
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        print(f"Chatbot: Planning to use {len(msg.tool_calls)} tool(s)")
                    else:
                        print(f"Chatbot: {msg.content[:100]}...")
                
                elif key == "tools":
                    print(f"🛠️ Tools: Executing search...")
                    for tool_msg in value["messages"]:
                        print(f"Result: {tool_msg.content[:100]}...")
        
        # Get final result
        final_result   = graph.invoke(initial_state)
        final_response = final_result["messages"][-1].content
        
        print("\n" + "="*80)
        print("FINAL RESPONSE")
        print("="*80)
        print(f"Chatbot: {final_response}")
        
        print("\n" + "="*80)
        print("Conversation with tools complete!")
        print("="*80)
        
        # Show conversation summary
        total_messages = len(final_result["messages"])
        tool_calls  = sum(1 for msg in final_result["messages"] 
                        if hasattr(msg, 'tool_calls') and msg.tool_calls)
        
        print(f"\nSession Summary:")
        print(f"   Total messages: {total_messages}")
        print(f"   Tool calls made: {tool_calls}")
        print(f"   Tools available: {len(tools)}")
        
        logger.info("Chatbot with tools session completed successfully")
        
    except Exception as e:
        logger.error(f"Error in chatbot with tools: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()