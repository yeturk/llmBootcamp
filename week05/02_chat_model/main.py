import os
from dotenv import load_dotenv
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

logger = logging.getLogger(__name__)  # 👈 Uses the module name
logger.setLevel(logging.INFO)

if not logger.handlers:
    ch = logging.StreamHandler()
    fh = logging.FileHandler("02_chat_model.log")

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

# Example usage
logger.info("This log message includes the module name.")



def main():
    load_dotenv()
    logger.info("Environment variables loaded from .env file (if present).")
    api_key = os.getenv("GOOGLE_API_KEY")
    # ... (API key check) ...
    if api_key:
        # Print last 6 character of key
        logger.info(f"GOOGLE_API_KEY: {api_key[:6]}********")
    else:
        logger.error("Error: GOOGLE_API_KEY not found in environment variables.")
        return # Exit if no API key
    
    # 1. Initialize the Chat Model
    logger.info("1. Initializing the Chat Model")
    logger.info("Creating a ChatGoogleGenerativeAI instance with 'gemini-2.5-flash' model.")
    # model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    chat_model = ChatGoogleGenerativeAI(
        api_key=api_key,
        model="gemini-2.5-flash", # Using the specified Gemini model
        temperature=0.7 # Controls randomness: 0.0 (deterministic) to 1.0 (creative)
    )
    logger.info("Chat model successfully created!")

    # 2. Sending a Simple Human Message
    logger.info("2. Sending a Simple Human Message")
    logger.info("Creating a HumanMessage and invoking the model.")
    
    # A HumanMessage represents input from the user.
    human_message = HumanMessage(content="Hello! Who are you?")
    
    # The invoke method sends a list of messages to the model.
    # Even for a single message, it expects a list.
    response = chat_model.invoke([human_message])
    
    logger.info(f"User: {human_message.content}")
    logger.info(f"AI: {response.content}")
    logger.info("Observe how the AI responds to a direct question.")
    print()

    # 3. Starting with a System Message (Setting Persona/Context)
    # System messages provide initial instructions or a persona to the AI.
    # They influence the AI's behavior throughout the conversation.
    logger.info("3. Starting with a System Message")
    logger.info("Adding a SystemMessage to guide the AI's behavior.")

    # A SystemMessage sets the context or persona for the AI.
    system_message = SystemMessage(content="You are a helpful AI assistant. You can only speak English, not other languages.")
    # user_message_2 = HumanMessage(content="Can you provide information about the Python programming language?")
    user_message_2 = HumanMessage(content="Are you AI assistant and can you speak English?")
    
    # We pass both the system message and the user message as a list.
    messages_with_system = [system_message, user_message_2]
    response_with_system = chat_model.invoke(messages_with_system)
    
    logger.info(f"System: {system_message.content}")
    logger.info(f"User: {user_message_2.content}")
    logger.info(f"AI: {response_with_system.content}")
    logger.info("Notice how the AI's response aligns with the system message (e.g., speaking English).")

if __name__=='__main__':
    main()