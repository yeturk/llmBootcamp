import os, json, uuid
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Optional, Literal

print("************* WEEK05 - HW5 *************\n")

class Entities(BaseModel):
    amount:         Optional[float] = None
    invoice_period: Optional[str]   = None
    ticket_id:      Optional[str]   = None
    device:         Optional[str]   = None
    address_move:   Optional[bool]  = None


class TicketExtraction(BaseModel):
    issue_type:         Literal["billing", "technical", "account", "general"]
    urgency:            Literal["low", "medium", "high"]
    channel:            Literal["phone", "email", "chat", "unknown"]
    entities:           Entities
    summary:            str
    status_suggestion:  Literal["open", "in_progress", "resolved"]


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# ... (API key check) ...
if api_key:
    # logger.info the first and the last 4 character of key
    print(f"GOOGLE_API_KEY: {api_key[:4]}********{api_key[-4::]}")
else:
    print("Error: GOOGLE_API_KEY not found in environment variables.")


print("\nCreating Gemini 2.5 Flash client...\n")

llm = ChatGoogleGenerativeAI(
    api_key=api_key,
    model="gemini-2.5-flash",
    temperature=0.4
)

# Modeli Pydantic ile bağla
structured_llm = llm.with_structured_output(TicketExtraction, include_raw=False)
print("Model successfully connected with structured output!\n")

# CSV'yi oku
df = pd.read_csv("support_tickets_minimal.csv")

print(f"Loaded {len(df)} tickets.")
# print(df.head())
# print(df.info())


# Logs klasörü oluştur
os.makedirs("logs", exist_ok=True)
log_path = "logs/outputs.jsonl"
run_id = str(uuid.uuid4())

for idx, row in df.iterrows():
    ticket_text = row["ticket_text"] if "ticket_text" in row else row.iloc[0]
    print(f"\n--- Processing Ticket #{idx} ---")
    print(f"Ticket text:\n{ticket_text}\n")

    try:
        response: TicketExtraction = structured_llm.invoke([
            ("system", "You are an assistant that extracts structured data from customer support tickets."),
            ("human", f"Extract structured data according to the provided schema from this ticket:\n{ticket_text}")
        ])

        # Terminale pretty print JSON
        print(json.dumps(response.model_dump(), indent=2))

        # Log dosyasına kaydet (append)
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "run_id": run_id,
                "source_id": idx,
                "data": response.model_dump()
            }) + "\n")

    except Exception as e:
        print(f"Error processing ticket {idx}: {e}")
