import argparse
import logging
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os
import uuid
import sqlite3

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the enhanced state with custom behavioral modes
class State(TypedDict):
    """
    Enhanced state with custom behavioral patterns.
    
    This state includes:
    - messages: Conversation history
    - mode: Agent behavior mode (casual, professional, technical, creative)
    - user_profile: Information about the user
    - session_info: Session-specific data
    """
    messages: Annotated[list, add_messages]
    mode: Literal["casual", "professional", "technical", "creative"]
    user_profile: dict  # User preferences and information
    session_info: dict  # Session metadata

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

def get_system_prompt(state: State) -> str:
    """
    Generate system prompt based on current state and mode.
    
    Args:
        state: Current state with mode and user profile
        
    Returns:
        str: Customized system prompt
    """
    mode = state.get("mode", "casual")
    user_profile = state.get("user_profile", {})
    
    # Base personality traits for each mode
    mode_personalities = {
        "casual": {
            "tone": "friendly and relaxed",
            "style": "conversational with emojis",
            "approach": "approachable and informal"
        },
        "professional": {
            "tone": "formal and respectful",
            "style": "business-appropriate language",
            "approach": "structured and efficient"
        },
        "technical": {
            "tone": "precise and detailed",
            "style": "technical terminology when appropriate",
            "approach": "thorough explanations with examples"
        },
        "creative": {
            "tone": "imaginative and inspiring",
            "style": "expressive and colorful language",
            "approach": "thinking outside the box"
        }
    }
    
    personality = mode_personalities.get(mode, mode_personalities["casual"])
    
    # Construct system prompt
    system_prompt = f"""You are an AI assistant operating in {mode} mode.

Personality traits:
- Tone: {personality['tone']}
- Style: {personality['style']}
- Approach: {personality['approach']}

"""
    
    # Add user-specific customization
    if user_profile.get("name"):
        system_prompt += f"The user's name is {user_profile['name']}. "
    
    if user_profile.get("expertise_level"):
        system_prompt += f"Adjust your explanations for someone with {user_profile['expertise_level']} expertise. "
    
    if user_profile.get("interests"):
        interests = ", ".join(user_profile['interests'])
        system_prompt += f"The user is interested in: {interests}. "
    
    system_prompt += f"\nAlways maintain the {mode} personality while being helpful and accurate."
    
    return system_prompt

def chatbot_with_mode(state: State):
    """
    Enhanced chatbot node that adapts behavior based on state.
    
    Args:
        state: Current state with mode and customizations
        
    Returns:
        dict: Updated state with LLM response
    """
    # Get mode-specific system prompt
    system_prompt = get_system_prompt(state)
    
    # Prepare messages with system prompt
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    # Get LLM response
    response = llm_with_tools.invoke(messages)
    
    # Log mode and decision
    mode = state.get("mode", "casual")
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_names = [tc['name'] for tc in response.tool_calls]
        logger.info(f"🎭 {mode.title()} mode - LLM wants to use tools: {', '.join(tool_names)}")
    else:
        logger.info(f"🎭 {mode.title()} mode - LLM responded directly")
    
    return {"messages": [response]}

def mode_controller(state: State):
    """
    Node that can change the agent's behavioral mode based on user requests.
    
    Args:
        state: Current state
        
    Returns:
        dict: Updated state with potentially new mode
    """
    last_message = state["messages"][-1]
    
    if hasattr(last_message, 'content'):
        content = last_message.content.lower()
        
        # Check for mode change requests
        if "switch to" in content or "change mode" in content:
            if "professional" in content:
                logger.info("🔄 Switching to professional mode")
                return {"mode": "professional"}
            elif "technical" in content:
                logger.info("🔄 Switching to technical mode")
                return {"mode": "technical"}
            elif "creative" in content:
                logger.info("🔄 Switching to creative mode")
                return {"mode": "creative"}
            elif "casual" in content:
                logger.info("🔄 Switching to casual mode")
                return {"mode": "casual"}
    
    # No mode change needed
    return {}

def profile_updater(state: State):
    """
    Node that updates user profile based on conversation.
    
    Args:
        state: Current state
        
    Returns:
        dict: Updated state with enhanced user profile
    """
    messages = state["messages"]
    current_profile = state.get("user_profile", {})
    
    # Look for human messages that contain profile information
    for message in messages:
        if hasattr(message, 'content') and hasattr(message, 'type'):
            # Only check human messages, not AI responses
            if message.type == 'human':
                content = message.content.lower()
                
                # Extract user information
                if "my name is" in content:
                    # Simple name extraction
                    parts = content.split("my name is")
                    if len(parts) > 1:
                        name = parts[1].split()[0].strip(".,!?").title()
                        current_profile["name"] = name
                        logger.info(f"📝 Updated user profile - Name: {name}")
                
                # Extract interests
                if "i like" in content or "i love" in content:
                    # This is a simplified interest extraction
                    if "python" in content:
                        interests = current_profile.get("interests", [])
                        if "Python programming" not in interests:
                            interests.append("Python programming")
                            current_profile["interests"] = interests
                            logger.info("📝 Updated user profile - Added interest: Python programming")
    
    return {"user_profile": current_profile}

def route_with_state(state: State):
    """
    Enhanced router that considers state and mode.
    
    Args:
        state: Current state
        
    Returns:
        str: Next node to execute
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check if we need to use tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        logger.info("🔄 Routing to tools...")
        return "tools"
    
    # Check if user wants to change mode or update profile
    # Look at the human message (second to last), not the AI response (last)
    if len(messages) >= 2:
        human_message = messages[-2]  # The user's message before the AI response
        if hasattr(human_message, 'content'):
            content = human_message.content.lower()
            
            if any(phrase in content for phrase in ["switch to", "change mode", "my name is", "i like", "i love"]):
                logger.info("🔄 Routing to state management...")
                return "mode_control"
    
    # End conversation
    logger.info("✅ Conversation complete")
    return "end"

# Build the graph with custom state following official LangGraph pattern
def build_custom_state_graph():
    """
    Creates a chatbot graph with custom state management.
    
    Graph structure:
    START -> chatbot -> [router] -> tools -> chatbot
                      \\-> mode_control -> profile_update -> chatbot
                      \\-> END
    
    Returns:
        Compiled LangGraph with custom state capabilities
    """
    # Create a StateGraph with our enhanced State schema
    graph_builder = StateGraph(State)
    
    # Add nodes
    graph_builder.add_node("chatbot", chatbot_with_mode)
    graph_builder.add_node("mode_control", mode_controller)
    graph_builder.add_node("profile_update", profile_updater)
    
    if tools:
        tool_node = ToolNode(tools)
        graph_builder.add_node("tools", tool_node)
    
    # Define flow: START -> chatbot
    graph_builder.add_edge(START, "chatbot")
    
    # Add conditional edges from chatbot
    if tools:
        graph_builder.add_conditional_edges(
            "chatbot",
            route_with_state,
            {
                "tools": "tools",
                "mode_control": "mode_control",
                "end": END,
            }
        )
        # Tools go back to chatbot
        graph_builder.add_edge("tools", "chatbot")
    else:
        graph_builder.add_conditional_edges(
            "chatbot",
            route_with_state,
            {
                "mode_control": "mode_control", 
                "end": END,
            }
        )
    
    # State management flow
    graph_builder.add_edge("mode_control", "profile_update")
    graph_builder.add_edge("profile_update", "chatbot")
    
    # Add memory checkpointer for persistence across sessions
    checkpointer = SqliteSaver(sqlite3.connect("checkpoint.db", check_same_thread=False))
    
    # Compile with checkpointer
    return graph_builder.compile(checkpointer=checkpointer)

def main():
    parser = argparse.ArgumentParser(description="Chatbot with Custom State - Official LangGraph Pattern")
    parser.add_argument("--message", type=str, required=True,
                       help="Message to send to the chatbot")
    parser.add_argument("--mode", type=str, default="casual",
                       choices=["casual", "professional", "technical", "creative"],
                       help="Initial agent behavior mode")
    parser.add_argument("--thread", type=str, default=None,
                       help="Thread ID for conversation memory")
    
    args = parser.parse_args()
    
    # Generate thread ID if not provided
    thread_id = args.thread or str(uuid.uuid4())[:8]
    
    print("\n" + "="*80)
    print("CHATBOT WITH CUSTOM STATE - OFFICIAL LANGGRAPH PATTERN")
    print("="*80)
    print("Building a chatbot with customizable behavioral modes...")
    print(f"Initial mode: {args.mode}")
    print(f"Tools available: {len(tools)} ({'Tavily Search' if tools else 'None'})")
    print(f"Thread ID: {thread_id}")
    print("="*80)
    
    try:
        # Build the chatbot graph with custom state
        graph = build_custom_state_graph()
        
        # Create input - only pass new message and mode, let memory handle the rest
        inputs = {
            "messages": [HumanMessage(content=args.message)],
            "mode": args.mode
        }
        
        # Configure with thread_id for memory
        config = {"configurable": {"thread_id": thread_id}}
        
        print(f"\n👤 User: {args.message}")
        print(f"🎭 Mode: {args.mode}")
        print("-" * 50)
        
        # Process the message with custom state
        logger.info(f"Processing message with custom state in thread: {thread_id}")
        
        # Execute the graph
        final_result = graph.invoke(inputs, config)
        
        # Extract final response
        ai_messages = [msg for msg in final_result["messages"] 
                      if hasattr(msg, 'content') and msg.content 
                      and not hasattr(msg, 'tool_calls')]
        
        if ai_messages:
            final_response = ai_messages[-1].content
        else:
            final_response = "The conversation completed but didn't produce a final response."
        
        print(f"\n🤖 Chatbot ({final_result['mode']} mode): {final_response}")
        
        print("\n" + "="*80)
        print("🎯 FINAL STATE")
        print("="*80)
        print(f"Mode: {final_result['mode']}")
        print(f"User Profile: {final_result.get('user_profile', {})}")
        print(f"Session Info: {final_result.get('session_info', {})}")
        
        print("\n" + "="*80)
        print("✅ Custom state conversation complete!")
        print("="*80)
        
        # Show capabilities
        print(f"\n💡 Try these commands:")
        print(f"   • 'Switch to professional mode'")
        print(f"   • 'Change mode to technical'")
        print(f"   • 'My name is [Your Name]'")
        print(f"   • 'I love [your interests]'")
        
        logger.info("Custom state chatbot session completed successfully")
        
    except Exception as e:
        logger.error(f"Error in custom state chatbot: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()