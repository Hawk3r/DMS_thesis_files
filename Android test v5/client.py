# gemini_client.py

import requests
import json

class GeminiClient:
    def __init__(self, api_key, model="gemini-1.5-flash"):
        self.api_key = api_key
        self.model = model
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={api_key}"
        self.headers = {
            "Content-Type": "application/json"
        }

    def chat(self, prompt):
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }

        try:
            response = requests.post(self.url, headers=self.headers, data=json.dumps(data))
            response.raise_for_status()
            result = response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except requests.exceptions.RequestException as e:
            return f"Network error: {e}"
        except Exception as e:
            return f"Parsing error: {e}"


client = GeminiClient("AIzaSyDU3X8MnBiFBvART6R9zMeyuL_UCHeLXTI")  # Replace with your actual Gemini API key

response = client.chat("can you act as a driver monitoring system, and if i say \"sleepy\" you automatically warn me of my sleepiness")
response = client.chat("sleepy")
print(response)