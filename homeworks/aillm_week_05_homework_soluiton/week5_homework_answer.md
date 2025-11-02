# Week 5 Reference Answer — Logging Only, Pydantic Schema

> Sample solution for Python 3.12, `langchain==0.3.26`, Gemini 2.5 Flash. No database; results are logged to stdout and JSONL.
## Inıt uv project
```
uv init -p 3.12 --name week05_answer --description "AI LLM Bootcamp Week05 answer with uv" week05_answer

cd week05_answer
```
## Create package and modules
```
mkdir app
mkdir logs
touch app/__init__.py
touch app/models.py
touch app/llm_chain.py
touch app/main.py
echo "GOOGLE_API_KEY=your-api-key-here" >> .env
touch requirements.txt
```
## File Tree
```
.
├── app
│   ├── __init__.py
│   ├── models.py
│   ├── llm_chain.py
│   └── main.py
├── logs
│   └── (created at runtime) outputs.jsonl
├── .env.example
└── requirements.txt
```

### requirements.txt
```
langchain==0.3.26
langchain-google-genai
pydantic==2.*
python-dotenv
pandas
```

### .env.example
```
GOOGLE_API_KEY=your_key_here
```
## Install requirements
```
uv add -r requirements.txt
```

## Activate virtual environment
```
train@lenovo128 week05_answer]$ source .venv/bin/activate
(week05-answer) [train@lenovo128 week05_answer]$ 
```

## `app/models.py`
```python
from typing import Optional, Literal
from pydantic import BaseModel, Field

class Entities(BaseModel):
    amount: Optional[float] = Field(default=None, description="Numeric amount, e.g., 49.99")
    invoice_period: Optional[str] = None
    ticket_id: Optional[str] = None
    device: Optional[str] = None
    address_move: Optional[bool] = None

class TicketExtraction(BaseModel):
    issue_type: Literal["billing","technical","account","general"]
    urgency: Literal["low","medium","high"]
    channel: Literal["phone","email","chat","unknown"]
    entities: Entities
    summary: str
    status_suggestion: Literal["open","in_progress","resolved"]
```

## `app/llm_chain.py`
```python
import os
import re
from typing import Any

from dotenv import load_dotenv
load_dotenv()  # ensure GOOGLE_API_KEY is present before LLM init

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from app.models import TicketExtraction

SYSTEM = (
    "You are a strict information extractor. "
    "Return JSON with EXACTLY these keys: "
    "issue_type, urgency, channel, entities, summary, status_suggestion. "
    "Allowed enums: "
    "issue_type: {{billing, technical, account, general}}; "
    "urgency: {{low, medium, high}}; "
    "channel: {{phone, email, chat, unknown}}; "
    "status_suggestion: {{open, in_progress, resolved}}. "
    "entities must contain: amount (number|null), invoice_period (string|null), "
    "ticket_id (string|null), device (string|null), address_move (boolean|null). "
    "If something is unknown, use null for nested fields or pick the closest enum at the top level. "
    "Do NOT add extra fields. Return JSON only."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "Ticket text:\n{ticket_text}\n\nReturn JSON only.")
])

# Don't hardcode the key; rely on env (GOOGLE_API_KEY) which is already loaded above.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Enforce Pydantic-validated structured output
chain: Runnable = prompt | llm.with_structured_output(TicketExtraction)

def _normalize_amount_like(value: Any):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"[-+]?\d+(?:\.\d+)?", value.replace(",", ""))
        return float(m.group(0)) if m else None
    return None

def extract_ticket(ticket_text: str) -> TicketExtraction:
    result: TicketExtraction = chain.invoke({"ticket_text": ticket_text})
    result.entities.amount = _normalize_amount_like(result.entities.amount)
    return result
```

## `app/main.py`
```python
import os, sys, json, time, logging
from uuid import uuid4
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from app.llm_chain import extract_ticket

load_dotenv()  # load GOOGLE_API_KEY

# --- configure logger ---
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True, parents=True)

logger = logging.getLogger("week05")
logger.setLevel(logging.INFO)

# console handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)

# file handler
fh = logging.FileHandler(logs_dir / "run.log", encoding="utf-8")
fh.setLevel(logging.INFO)

# formatter
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
fh.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(fh)
# -------------------------

def main():
    if len(sys.argv) < 2:
        logger.error("Usage: python -m app.main /path/to/support_tickets_minimal.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)

    if not {"user_id", "sikayet"}.issubset(df.columns):
        logger.error(f"Expected columns user_id and sikayet. Found: {list(df.columns)}")
        sys.exit(1)

    out_path = logs_dir / "outputs.jsonl"
    run_id = str(uuid4())
    max_retries = 2

    with open(out_path, "a", encoding="utf-8") as fout:
        for _, row in df.iterrows():
            source_id = str(row["user_id"])
            ticket_text = str(row["sikayet"])

            attempts = 0
            while True:
                try:
                    result = extract_ticket(ticket_text)
                    payload = result.model_dump()
                    # log to console/file (pretty JSON one-liner)
                    logger.info("row %s: %s", source_id, json.dumps(payload, ensure_ascii=False))
                    # append JSONL with metadata
                    fout.write(json.dumps({
                        "run_meta": {"run_id": run_id, "source_id": source_id, "ts": time.time()},
                        "data": payload
                    }, ensure_ascii=False) + "\n")
                    break
                except Exception as e:
                    if attempts >= max_retries:
                        logger.error("row %s failed permanently: %s", source_id, e)
                        break
                    attempts += 1
                    logger.warning("row %s failed (attempt %d), retrying: %s", source_id, attempts, e)
                    time.sleep(1.5 ** attempts)

    logger.info("Done. Wrote logs to %s", out_path)

if __name__ == "__main__":
    main()

```

---

## How to Run
```bash

# Env
cp .env.example .env  # then fill GOOGLE_API_KEY

# Execute
uv run python -m app.main support_tickets_minimal.csv
```

**Notes**
- Uses `with_structured_output(TicketExtraction)` to hard-enforce the schema.
- JSONL contains `run_meta` wrapper + `data` (the strict object). The strict fields match the assignment exactly.
- If the CSV has a different text column name, adjust `choose_text_col()` or rename your column.
