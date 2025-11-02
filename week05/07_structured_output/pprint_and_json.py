from pprint import pprint

data = {
    "sentiment": "POSITIVE",
    "reason": "The text uses strong positive words like 'love' and 'fantastic'.",
    "confidence_score": 0.98,
    "extra_info": {
        "language": "en",
        "source": "user_input",
        "tokens": [1, 2, 3, 4]
    }
}

pprint(data)

print()

# Eğer dict içeriğini JSON formatında ve renksiz ama temiz görmek istersen:
import json
print(json.dumps(data, indent=4, ensure_ascii=False))
