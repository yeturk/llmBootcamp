# playwright_scraper.py
# Author: Yunus Emre
# Description: Fetches fully rendered HTML text from dynamic EU AI Act pages using Playwright.

from playwright.sync_api import sync_playwright
import time

urls = [
    "https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng",
    "https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai",
    "https://commission.europa.eu/news-and-media/news/ai-act-enters-force-2024-08-01_en",
    "https://digital-strategy.ec.europa.eu/en/faqs/general-purpose-ai-models-ai-act-questions-answers"
]

def fetch_with_playwright(url: str) -> str:
    print(f"🌐 Fetching: {url}")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Sayfanın tamamen yüklenmesini bekle
        page.goto(url, wait_until="networkidle")

        # 2-3 sn ek bekleme (bazı JS istekleri geç yükleniyor)
        time.sleep(3)

        # Sayfanın metin içeriğini al (body)
        text = page.inner_text("body")
        browser.close()
        return text

def main():
    all_texts = []
    for url in urls:
        try:
            text = fetch_with_playwright(url)
            print(f"✅ Length: {len(text)} characters\n{'-'*80}")
            all_texts.append({"url": url, "text": text})
        except Exception as e:
            print(f"❌ Failed to fetch {url}: {e}")

    # Kaydet (isteğe bağlı)
    with open("eu_ai_act_texts.txt", "w", encoding="utf-8") as f:
        for item in all_texts:
            f.write(f"\n\n=== {item['url']} ===\n\n")
            f.write(item["text"])

if __name__ == "__main__":
    main()
