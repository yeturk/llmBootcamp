import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

conn_str = os.getenv("POSTGRES_URL")

try:
    conn = psycopg2.connect(conn_str)
    print("✅ PostgreSQL bağlantısı başarılı!")
    conn.close()
except Exception as e:
    print("❌ Bağlantı başarısız:", e)
