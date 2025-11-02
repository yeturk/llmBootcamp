import os
from dotenv import load_dotenv
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain import hub

# --------------------------------------------------
# 1️⃣ LOGGER CONFIGURATION
# --------------------------------------------------
# We use Python's logging module to track runtime activity.
# This helps you debug, see what happens during execution,
# and log information both to console and to a file.
logger = logging.getLogger(__name__)  # Logger name = module name (e.g., "05_tools.main")
logger.setLevel(logging.INFO)         # Set log level: INFO means general progress info

# Add handlers only if not already added (avoids duplicates)
if not logger.handlers:
    ch = logging.StreamHandler()             # Output logs to the console
    fh = logging.FileHandler("05_tools.log") # Save logs to a file
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)


# --------------------------------------------------
# 2️⃣ TOOL DEFINITION
# --------------------------------------------------
# A tool is just a normal Python function decorated with @tool.
# The @tool decorator automatically creates a schema describing:
# - name
# - description (from docstring)
# - arguments (from function parameters)
#
# This metadata allows an LLM to understand how to call this function.
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


# --------------------------------------------------
# 3️⃣ MAIN PROGRAM
# --------------------------------------------------
def main():
    # -------------------------
    # Load API key from .env file
    # -------------------------
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not found")
        return  # Exit if key is missing
    logger.info("Environment loaded successfully")

    # -------------------------
    # Initialize the LLM (Gemini model)
    # -------------------------
    # This creates the chat model (the brain).
    # temperature=0 means deterministic outputs (no randomness).
    llm = ChatGoogleGenerativeAI(
        api_key=api_key,
        model="gemini-2.5-flash",
        temperature=0
    )

    # -------------------------
    # List of tools the agent can use
    # -------------------------
    # Each tool is a "skill" or "capability" that the agent can call.
    tools = [multiply]

    # -------------------------
    # Load a pre-built system prompt (ReAct-style)
    # -------------------------
    # The "hub" provides standard LangChain prompts.
    # "openai-functions-agent" is a ReAct-style agent template that supports function calling.
    prompt = hub.pull("hwchase17/openai-functions-agent")

    # -------------------------
    # Create the agent
    # -------------------------
    # The agent knows:
    # - which tools are available
    # - how to use them
    # - and how to reason about user queries
    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    # -------------------------
    # Create the executor
    # -------------------------
    # AgentExecutor actually runs the agent:
    # it manages the reasoning loop (deciding → calling tool → responding)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # -------------------------
    # Invoke the agent
    # -------------------------
    # The agent will read your input, decide if it should use a tool,
    # call it automatically, and then give you the final result.
    result = agent_executor.invoke({"input": "What is 8 times 7?"})
    print("Agent result:", result)


# --------------------------------------------------
# 4️⃣ ENTRY POINT
# --------------------------------------------------
# This ensures main() runs only when executed directly,
# not when imported as a module.
if __name__ == "__main__":
    main()
