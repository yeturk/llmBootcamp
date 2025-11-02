# Week 5 Homework — LangChain Structured Output ➜ Logging (No DB)

**Duration:** ~3 hours  
**Languages/Versions:** Python 3.12, `langchain==0.3.26`  
**LLM:** Google Gemini 2.5 Flash (via `langchain-google-genai`)  
**Rule:** Do **not** expose your API key in code. Use environment variables / `.env` only.  
**Change:** No database. **Log** results to console and to a JSONL file.

---

## Objective
Using the attached CSV (`support_tickets_minimal.csv`), parse each ticket text with an LLM and extract a **strict structured object** (validated by **Pydantic**). Log each validated record to:
1) **stdout** (pretty-printed JSON) and  
2) `logs/outputs.jsonl` (one JSON per line, append-safe).

### Target schema (must match exactly)
```json
{
  "issue_type": "billing | technical | account | general",
  "urgency": "low | medium | high",
  "channel": "phone | email | chat | unknown",
  "entities": {
    "amount": "number | null",
    "invoice_period": "string | null",
    "ticket_id": "string | null",
    "device": "string | null",
    "address_move": "boolean | null"
  },
  "summary": "string",
  "status_suggestion": "open | in_progress | resolved"
}
```

**Rules**
- Use **Pydantic** models for validation (no `TypedDict`).
- No extra keys. Enforce enums. Coerce numeric/boolean fields when possible; otherwise `None`.
- If unknown/not present: use the closest enum (top-level) or `None` (nested fields).

---

## Tasks
1. **Environment**
   - Create & activate a Python 3.12 venv.
  
   - Create `.env`:
    

2. **Pydantic schema + Structured output**
   - Define `Entities` and `TicketExtraction` Pydantic models (see rubric).
   - Build a LangChain pipeline that reads each CSV row, calls Gemini 2.5 Flash, and **returns exactly** the target structure.

3. **Logging (no database)**
   - For each row:
     - Print the validated Pydantic model as formatted JSON to stdout.
     - Append the same JSON to `logs/outputs.jsonl` (create `logs/` if missing).
   - Add a `run_id` (UUID) and `source_id` (CSV index or an `id` column if present) **only in the log line** (do not change the schema fields themselves).

4. **CLI**
   - Command:
     ```bash
     python pplication_file/module /path/to/support_tickets_minimal.csv
     ```

5. **Deliverables**
   - `README.md` with steps to run and a screenshot of terminal output.
   - `logs/outputs.jsonl`

---
