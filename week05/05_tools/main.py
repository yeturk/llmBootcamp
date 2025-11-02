import os
from dotenv import load_dotenv
import logging
from langchain.agents import initialize_agent, AgentType 
# Eğer son sürümü (örneğin langchain==0.2.x veya langchain>=1.x) 
# kullanmak istiyorsan, artık "initialize_agent" yok.
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

logger = logging.getLogger(__name__)  # 👈 Uses the module name
logger.setLevel(logging.INFO)

if not logger.handlers:
    ch = logging.StreamHandler()
    fh = logging.FileHandler("05_tools.log")

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

    # invoke tool directly
    logger.info(f"Using tool directly {multiply.invoke({"a": 2, "b": 3})} - name: {multiply.name} - Description: {multiply.description} - Args: {multiply.args}")

if __name__=='__main__':
    main()