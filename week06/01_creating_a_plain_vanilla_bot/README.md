## Add new dependencies to requirements.txt
- with new ones 
```
langchain==0.3.27
langchain-google-genai==2.1.8
langchain-chroma==0.2.5
langchain-community==0.3.27
python-dotenv==1.1.1
pypdf==6.0.0
pysqlite3-binary
langchain-tavily
```

## Install new requirements
- From the root uv project directory where the pyproject.toml is in.
```
uv add -r requirements.txt
```
## 
- main.py
```python
import argparse
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
    parser = argparse.ArgumentParser(description="Chat with OpenAI model using LangChain.")
    parser.add_argument("--system", type=str, required=False, help="You are a helpful assistant that help the user to plan an optimized itinerary.")
    parser.add_argument("--question", type=str, required=True, help="I'm going to Istanbul for 2 days, offer me top 5 places to visit?")
    args = parser.parse_args()

    response = chat_model.invoke([
        SystemMessage(content=args.system),
        HumanMessage(content=args.question)]
        )
    logger.info(f"Response: {response.content}")
```

## Run
```
uv run week_06/01_creating_a_plain_vanilla_bot/main.py \
--system "You are a helpful assistant that help the user to plan an optimized itinerary." \
--question "Selam"
```
- Output
```
2025-09-02 21:53:38,016 - __main__ - INFO - This log message includes the module name.
2025-09-02 21:53:39,579 - __main__ - INFO - Response: Aleykümselam!

Size nasıl yardımcı olabilirim? Seyahat planınızı optimize etmek için buradayım. Lütfen detayları paylaşın.
```

## Second question
```
uv run week_06/01_creating_a_plain_vanilla_bot/main.py \
--system "You are a helpful assistant that help the user to plan an optimized itinerary." \
--question "I'm going to Istanbul for 2 days, offer me top 5 places to visit?"
```
- Output
```
2025-09-02 21:56:57,004 - __main__ - INFO - This log message includes the module name.
2025-09-02 21:57:11,509 - __main__ - INFO - Response: Welcome to Istanbul! Two days is a whirlwind, but you can definitely hit the major highlights. The key is to group attractions geographically.

Here are the top 5 places I recommend, plus a bonus activity, structured into a logical 2-day itinerary:

---

## Top 5 Places to Visit in Istanbul (2 Days)

1.  **Hagia Sophia (Ayasofya):** A monument of immense historical and architectural significance, having served as a church, mosque, and museum, now a mosque again. Its sheer scale and stunning mosaics are breathtaking.
2.  **Blue Mosque (Sultan Ahmed Mosque):** Directly facing Hagia Sophia, this iconic mosque is famous for its six minarets and the exquisite blue tiles that adorn its interior. It's an active place of worship.
3.  **Topkapi Palace Museum:** The opulent primary residence of the Ottoman Sultans for nearly 400 years. Explore its courtyards, treasuries, and the Harem (highly recommended, separate ticket).
4.  **Basilica Cistern (Yerebatan Sarnıcı):** An atmospheric underground marvel, this ancient water reservoir features hundreds of illuminated columns, including the famous Medusa heads.
5.  **Grand Bazaar (Kapalıçarşı) & Spice Bazaar (Mısır Çarşısı):** Two iconic, bustling markets. The Grand Bazaar is a labyrinth of shops selling everything from carpets to jewelry, while the Spice Bazaar offers an aromatic feast of spices, sweets, and dried fruits.

---

## Your Optimized 2-Day Istanbul Itinerary

**Day 1: Sultanahmet Immersion (History & Grandeur)**
*All these sites are within a very short walking distance of each other.*

*   **Morning (9:00 AM - 1:00 PM): Hagia Sophia & Blue Mosque**
    *   Start early at **Hagia Sophia** to beat some of the crowds. Allow 1.5-2 hours to truly appreciate its history and architecture.
    *   Walk directly across the square to the **Blue Mosque**. Check prayer times as it's closed to tourists during prayers. Dress appropriately (shoulders and knees covered, headscarf for women). Allow 1 hour.
*   **Lunch (1:00 PM - 2:00 PM):** Grab a bite at one of the many restaurants in the Sultanahmet area.
*   **Afternoon (2:00 PM - 5:30 PM): Topkapi Palace & Basilica Cistern**
    *   Head to **Topkapi Palace**. This will take a good 2.5-3 hours, especially if you visit the Harem (which I recommend for its beauty and history).
    *   Finish your day with the mystical **Basilica Cistern**. It's a shorter visit (45 minutes - 1 hour) and offers a cool, atmospheric escape.

**Day 2: Bazaars, Bosphorus & Views**

*   **Morning (9:30 AM - 1:00 PM): Grand Bazaar & Spice Bazaar**
    *   Take a tram or taxi to the **Grand Bazaar**. Get lost in its maze-like alleys, soak in the atmosphere, and practice your haggling skills. Allow 2-2.5 hours.
    *   From the Grand Bazaar, it's about a 15-20 minute walk (or short tram ride) to the **Spice Bazaar** near Eminönü. Enjoy the vibrant colors and aromas of spices, Turkish delight, and more. Allow 1 hour.
*   **Lunch (1:00 PM - 2:00 PM):** Enjoy lunch near Eminönü, perhaps a famous fish sandwich from a boat under the Galata Bridge, or a meal at a local restaurant.
*   **Afternoon (2:00 PM - 4:00 PM): Bosphorus Cruise (Highly Recommended Bonus!)**
    *   From Eminönü, you can easily catch a **Bosphorus Cruise**. This is an absolute must-do for a unique perspective of Istanbul, seeing both European and Asian sides, historic palaces, and Ottoman villas from the water. There are various options, from short 1.5-hour tours to longer ones. A 1.5-2 hour public ferry cruise is usually sufficient for a quick trip.
*   **Late Afternoon/Evening (Optional): Galata Tower & Istiklal Avenue**
    *   After your cruise, you could walk across the Galata Bridge (enjoying the fishermen) towards the **Galata Tower**. Go up for panoramic views of the city (can be crowded, consider booking ahead).
    *   From Galata Tower, it's a short walk to **Istiklal Avenue**, a bustling pedestrian street perfect for people-watching, dinner, and soaking in modern Istanbul's energy.

---

**Important Tips for Your Trip:**

*   **Comfortable Shoes:** You'll be doing a lot of walking!
*   **Dress Code:** For mosques, women should carry a headscarf to cover their hair, and both men and women should ensure shoulders and knees are covered.
*   **Opening Hours:** Check the current opening hours and days for each attraction, as they can change. Most are closed one day a week (e.g., Topkapi Palace on Tuesdays).
*   **Istanbulkart:** Purchase an Istanbulkart for easy and cheaper public transport (trams, metro, ferries).
*   **Hydrate & Snack:** Keep water and small snacks with you, especially during hot weather.
*   **Enjoy the Food:** Don't forget to try local delicacies like Turkish delight, baklava, kebabs, and Turkish tea/coffee!

Enjoy your incredible 2 days in Istanbul!
```