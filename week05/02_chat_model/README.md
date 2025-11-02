# ChatModels
- "LLM" and "Chat Model" interchangeably. 
- Modern LLMs are typically accessed through a chat model interface that takes a list of messages as input and returns a message as output.

The newest generation of chat models offer additional capabilities:

- **Tool calling:** Enables LLMs to interact with external services, APIs, and databases.
- **Structured output:** A technique to make a chat model respond in a structured format, such as JSON that matches a given schema, for example return yes or no.
- **Multimodality:** The ability to work with data other than text; for example, images, audio, and video.

## Features
LangChain provides a consistent interface for working with chat models from different providers.

- Integrations: Can integrate many different models like Anthropic, OpenAI, Ollama, Microsoft Azure, Google Vertex, Amazon Bedrock, Hugging Face, Cohere, Groq.
- Tool calling: Standard interface for binding tools to models
- Structured output: Standard API for structuring outputs.
- Provides support for async programming, efficient batching, a rich streaming API.
- Token usage, rate limiting, caching.

## Interface
- What is an Interface? (In Simple Terms): An interface defines a contract that a class must follow — a set of methods it must implement. You can think of it as a blueprint or template. It doesn’t contain the full logic — it just says:
💬 "Any class that wants to be considered of this type must have these methods, with these names and inputs/outputs."
- LangChain chat models implement the BaseChatModel interface. Because BaseChatModel also implements the Runnable Interface.
- Many of the key methods of chat models operate on messages as input and return messages as output.

## Key methods
The key methods of a chat model are:

- **invoke:** The primary method for interacting with a chat model. It takes a list of messages as input and returns a list of messages as output.
- **stream:** A method that allows you to stream the output of a chat model as it is generated.
- **batch:** A method that allows you to batch multiple requests to a chat model together for more efficient processing.
- **bind_tools:** A method that allows you to bind a tool to a chat model for use in the model's execution context.
- **with_structured_output:** A wrapper around the invoke method for models that natively support structured output.

## Inputs and outputs
Modern LLMs are typically accessed through a chat model interface that takes messages as input and returns messages as output. Messages are typically associated with a role (e.g., "system", "human", "assistant") and one or more content blocks that contain text or potentially multimodal data (e.g., images, audio, video).
LangChain supports two message formats to interact with chat models:

- **LangChain Message Format:** LangChain's own message format, which is used by default and is used internally by LangChain.
- OpenAI's Message Format: OpenAI's message format.

## Standard parameters
- **model:** The name or identifier of the specific AI model you want to use (e.g., "gpt-3.5-turbo" or "gpt-4").
- **temperature**: Controls the randomness of the model's output. A higher value (e.g., 1.0) makes responses more creative, while a lower value (e.g., 0.0) makes them more deterministic and focused.
- timeout
- **max_tokens:** Limits the total number of tokens (words and punctuation) in the response. This controls how long the output can be.

## Tool calling
Chat models can call tools to perform tasks such as fetching data from a database, making API requests, or running custom code. 

## Structured outputs
Chat models can be requested to respond in a particular format (e.g., JSON or matching a particular schema). This feature is extremely useful for information extraction tasks. 

## Multimodality
Large Language Models (LLMs) are not limited to processing text. They can also be used to process other types of data, such as images, audio, and video. 
## Context window
*The maximum size of the input sequence the model can process at one time.* 
If the input exceeds the context window, the model may not be able to process the entire input and could raise an error. In conversational applications, this is especially important because the context window determines how much information the model can "remember" throughout a conversation. Developers often need to manage the input within the context window to maintain a coherent dialogue without exceeding the limit. For more details on handling memory in conversations, refer to the memory.

## Rate-limiting
Many chat model providers impose a limit on the number of requests that can be made in a given time period.

## Caching
Theoretically, caching can help improve performance by reducing the number of requests made to the model provider. In practice, caching chat model responses is a complex problem and should be approached with caution. However, there might be situations where caching chat model responses is beneficial. For example, if you have a chat model that is used to answer frequently asked questions, caching responses can help reduce the load on the model provider, costs, and improve response times.

## Example
- Create main.py  in 02_chat_model directory

### Check Google API Key
```python
import os
from dotenv import load_dotenv
import logging

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
        # Print last 4 character of key
        logger.info(f"GOOGLE_API_KEY: ********{api_key[:4]}")
    else:
        logger.error("Error: GOOGLE_API_KEY not found in environment variables.")
        return # Exit if no API key
    
if __name__=='__main__':
    main()
```
- Run 
```bash
uv run week_05/02_chat_model/main.py
```

### Initialize chat model
```python
from langchain_google_genai import ChatGoogleGenerativeAI

# rest above

 # 1. Initialize the Chat Model
    logger.info("1. Initializing the Chat Model")
    logger.info("Creating a ChatGoogleGenerativeAI instance with 'gemini-2.5-flash' model.")
    chat_model = ChatGoogleGenerativeAI(
        api_key=api_key,
        model="gemini-2.5-flash", # Using the specified Gemini model
        temperature=0.7 # Controls randomness: 0.0 (deterministic) to 1.0 (creative)
    )
    logger.info("Chat model successfully created!")
if __name__=='__main__':
    main()
```
- Run
```bash
uv run week_05/02_chat_model/main.py
```
- Output
```
2025-07-29 10:58:31,231 - __main__ - INFO - This log message includes the module name.
2025-07-29 10:58:31,232 - __main__ - INFO - Environment variables loaded from .env file (if present).
2025-07-29 10:58:31,232 - __main__ - INFO - GOOGLE_API_KEY: ********AIza
2025-07-29 10:58:31,232 - __main__ - INFO - 1. Initializing the Chat Model
2025-07-29 10:58:31,232 - __main__ - INFO - Creating a ChatGoogleGenerativeAI instance with 'gemini-2.5-flash' model.
2025-07-29 10:58:31,255 - __main__ - INFO - Chat model successfully created!
```

### Sending HumaMessage
```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# rest above

 # This demonstrates the most basic interaction: a single user query.
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

if __name__=='__main__':
    main()
```

- Run
```bash
uv run week_05/02_chat_model/main.py
```
- Output
```
2025-07-29 11:06:25,166 - __main__ - INFO - This log message includes the module name.
2025-07-29 11:06:25,167 - __main__ - INFO - Environment variables loaded from .env file (if present).
2025-07-29 11:06:25,167 - __main__ - INFO - GOOGLE_API_KEY: ********AIza
2025-07-29 11:06:25,167 - __main__ - INFO - 1. Initializing the Chat Model
2025-07-29 11:06:25,167 - __main__ - INFO - Creating a ChatGoogleGenerativeAI instance with 'gemini-2.5-flash' model.
2025-07-29 11:06:25,179 - __main__ - INFO - Chat model successfully created!
2025-07-29 11:06:25,179 - __main__ - INFO - 2. Sending a Simple Human Message
2025-07-29 11:06:25,179 - __main__ - INFO - Creating a HumanMessage and invoking the model.
2025-07-29 11:06:28,574 - __main__ - INFO - User: Hello! Who are you?
2025-07-29 11:06:28,574 - __main__ - INFO - AI: Hello!

I am a large language model, an AI. I was trained by Google.

I don't have a name, personal identity, feelings, or consciousness. My purpose is to assist you by providing information, answering questions, generating text, and engaging in helpful conversation based on the vast amount of text data I've been trained on.

Think of me as a helpful digital assistant!
2025-07-29 11:06:28,574 - __main__ - INFO - Observe how the AI responds to a direct question.
```

### Starting with a System Message (Setting Persona/Context)
```python

## rest above

    # 3. Starting with a System Message (Setting Persona/Context)
    # System messages provide initial instructions or a persona to the AI.
    # They influence the AI's behavior throughout the conversation.
    logger.info("3. Starting with a System Message")
    logger.info("Adding a SystemMessage to guide the AI's behavior.")

    # A SystemMessage sets the context or persona for the AI.
    system_message = SystemMessage(content="You are a helpful AI assistant. You speak English.")
    user_message_2 = HumanMessage(content="Can you provide information about the Python programming language?")
    
    # We pass both the system message and the user message as a list.
    messages_with_system = [system_message, user_message_2]
    response_with_system = chat_model.invoke(messages_with_system)
    
    logger.info(f"System: {system_message.content}")
    logger.info(f"User: {user_message_2.content}")
    logger.info(f"AI: {response_with_system.content}")
    logger.info("Notice how the AI's response aligns with the system message (e.g., speaking English).")
if __name__=='__main__':
    main()
```
- Run
```bash
uv run week_05/02_chat_model/main.py
```