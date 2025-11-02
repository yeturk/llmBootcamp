# Week 5 Homework — LangChain Structured Output ➜ Logging (No DB)

**Student:** Yunus Emre TÜRK

---

## 🧭 Objective
Using `support_tickets_minimal.csv`, each support ticket is parsed by the Gemini model and transformed into a **structured JSON object** validated by **Pydantic**.  
Validated results are logged both:
1. to **stdout** (pretty-printed JSON)  
2. to `logs/outputs.jsonl` (one JSON per line, append-safe).

---

## 🧰 Environment Setup

### 1️⃣ Create virtual environment
```bash
conda create -n hw5_env python=3.12 -y
conda activate hw5_env
```
### 2️⃣ Install required libraries
```bash
pip install langchain==0.3.26
pip install "langchain-google-genai>=2.1.0,<3.0.0"
pip install pydantic==2.12.3 python-dotenv pandas
```
### 3️⃣ Set up .env file
```bash
GOOGLE_API_KEY="AIzaSXXXXXXXXXXXXXXXXXXXXX"
```

### 🧩 File Structure
```bash
hw5/
├── app.py
├── support_tickets_minimal.csv
├── logs/
│   └── outputs.jsonl
└── README.md
```

### ⚙️ How to Run
```bash
python app.py
```