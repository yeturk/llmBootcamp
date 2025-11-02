import os
from dotenv import load_dotenv
import logging
from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

logger = logging.getLogger(__name__)  # 👈 Uses the module name
logger.setLevel(logging.INFO)

if not logger.handlers:
    ch = logging.StreamHandler()
    fh = logging.FileHandler("06_tool_calling.log")

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

# Example usage
logger.info("This log message includes the module name.")

@tool
def multiply(a: int, b: int) -> int:
   """Multiply two numbers."""
   return a * b

@tool
def add(a: int, b: int) -> int:
   """Add two numbers."""
   return a + b

def main():
    load_dotenv()
    logger.info("Environment variables loaded from .env file (if present).")
    api_key = os.getenv("GOOGLE_API_KEY")
    # ... (API key check) ...
    if api_key:
        # Print last 4 character of key
        logger.info(f"GOOGLE_API_KEY: ********{api_key[:4]}")
    else:
        logger.error("Error: GOOGLE_API_KEY not found in environment variables.")
        return # Exit if no API key

    logger.info("Creating a ChatGoogleGenerativeAI instance with 'gemini-2.5-flash' model.")
    llm = ChatGoogleGenerativeAI(
        api_key=api_key,
        model="gemini-2.5-flash", # Using the specified Gemini model
        temperature=0.7 # Controls randomness: 0.0 (deterministic) to 1.0 (creative)
    )
    logger.info("Chat model successfully created!")

    tool_list = [multiply, add]

    llm_with_tools = llm.bind_tools(tools=tool_list)

    logger.info("========================= Asking general question =========================")
    logger.info(llm_with_tools.invoke("Where is the capital of France?"))

    logger.info("========================= Asking tool calling question =========================")
    logger.info(llm_with_tools.invoke("what is multiplied 5 and 4?"))

if __name__=='__main__':
    main()