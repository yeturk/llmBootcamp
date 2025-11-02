import argparse     # Komut satırından parametre almak için (etc. --system ve --question)
import logging
import os
from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)  # 👈 Uses the module name
logger.setLevel(logging.INFO)

if not logger.handlers:
    ch = logging.StreamHandler()
    fh = logging.FileHandler("01_creating_a_plain_vanilla_bot.log")

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

# Example usage
logger.info("This log message includes the module name.")
api_key = os.getenv("GOOGLE_API_KEY")

chat_model = init_chat_model("gemini-2.5-flash", model_provider="google_genai",  api_key=api_key)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Chat with gemini-2.5-flash model using LangChain.")
    parser.add_argument("--system", "-s", type=str, required=False, help="You are a helpful assistant that help the user to plan an optimized itinerary.")
    parser.add_argument("--question", "-q", type=str, required=True, help="I'm going to Istanbul for 2 days, offer me top 3 places to visit?")
    args = parser.parse_args()

    response = chat_model.invoke([
        SystemMessage(content=args.system),
        HumanMessage(content=args.question)]
        )
    logger.info(f"Response: \n{response.content}")