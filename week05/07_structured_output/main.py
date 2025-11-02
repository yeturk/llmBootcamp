import os, logging
from dotenv import load_dotenv
from enum import Enum
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

print("07_structured_output/main.py\n")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    ch = logging.StreamHandler()
    fh = logging.FileHandler("06_tool_calling.log")

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)


# 1. Define your simplified classification categories as an Enum
class SimpleSentiment(str, Enum):
    """Basic sentiment classifications."""
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL  = "NEUTRAL"

# 2. Define the simplified structured output schema using Pydantic
class SimpleSentimentAnalysis(BaseModel):
    """Analyzed sentiment of a given text (simplified)."""
    sentiment: SimpleSentiment = Field(
        description="The overall sentiment of the text: POSITIVE, NEGATIVE, or NEUTRAL."
    )
    reason: str = Field(
        description="A brief explanation for the determined sentiment."
    )
    confidence_score: float = Field(
        description="A score between 0.0 and 1.0 indicating confidence in the sentiment classification."
    )

def main():
    load_dotenv()
    logger.info("Environment variables loaded from .env file (if present).")
    api_key = os.getenv("GOOGLE_API_KEY")
    # ... (API key check) ...
    if api_key:
        # logger.info the first and the last 4 character of key
        logger.info(f"GOOGLE_API_KEY: {api_key[:4]}********{api_key[-4::]}")
    else:
        logger.error("Error: GOOGLE_API_KEY not found in environment variables.")
        return # Exit if no API key

    logger.info("Creating a ChatGoogleGenerativeAI instance with 'gemini-2.5-flash' model.")
    llm = ChatGoogleGenerativeAI(
        api_key=api_key,
        model="gemini-2.5-flash",   # Using the specified Gemini model
        temperature=0.6             # Controls randomness: 0.0 (deterministic) to 1.0 (creative)
    )
    logger.info("Chat model successfully created!")

    # Bind the structured output schema to the model
    model_with_structured_output = llm.with_structured_output(SimpleSentimentAnalysis, include_raw=False)

    # --- Test Cases ---
    texts_to_analyze = [
        "I absolutely love this product! It's fantastic.",          # Positive
        "The service was terrible and the food was cold.",          # Negative
        "The sky is blue today.",                                   # Neutral
        "What a wonderful surprise, I'm so happy!",                 # Positive
        "I'm very disappointed with the outcome.",                  # Negative
        "The book is exactly 300 pages long.",                      # Neutral
        "This news has made my day so much better!",                # Positive
        "I'm truly upset by the constant delays.",                  # Negative
    ]

    logger.info("--- Simplified Sentiment Analysis Examples ---")

    for i, text in enumerate(texts_to_analyze):
        logger.info(f"\n--- Analysis {i+1} ---")
        logger.info(f"Input Text: \"{text}\"")

        # 4. Ask a question and get the structured output
        # The prompt instructions are now part of the message sent directly to the model.
        # We send a list of messages, as expected by chat models.
        messages = [
            ("system",
            "You are an expert sentiment analysis AI. Classify the sentiment of the following text "
            "as POSITIVE, NEGATIVE, or NEUTRAL. "
            "Provide a brief reason for your classification and a confidence score between 0.0 and 1.0."
            ),
            ("human", f"Analyze the sentiment of this text: {text}"),
        ]

        try:
            # Invoke the model directly with the messages
            response: SimpleSentimentAnalysis = model_with_structured_output.invoke(messages)

            # 5. Return structured output
            logger.info("\nStructured Output:")
            logger.info(f"  Sentiment: {response.sentiment.value}")
            logger.info(f"  Reason: {response.reason}")
            logger.info(f"  Confidence: {response.confidence_score:.2f}")
            logger.info("\nFull Pydantic object (JSON):")
            logger.info(response.model_dump_json(indent=2))

        except Exception as e:
            logger.error(f"An error occurred during analysis: {e}")
            logger.error("Please ensure your GOOGLE_API_KEY is correctly set and you have access to gemini-flash-2.5.")

if __name__=='__main__':
    main()