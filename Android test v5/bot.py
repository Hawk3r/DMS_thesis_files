# gemini_chatbot.py

import requests
import json

class GeminiChatbot:
    def __init__(self, api_key, model="gemini-1.5-flash"):
        self.api_key = api_key
        self.model = model
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        self.headers = {
            "Content-Type": "application/json"
        }
        self.chat_history = []  # Proper Gemini format: list of dicts with 'role' and 'parts'

    def send_message(self, user_message):
        # Add the user message to history
        self.chat_history.append({
            "role": "user",
            "parts": [{"text": user_message}]
        })

        data = {
            "contents": self.chat_history
        }

        try:
            response = requests.post(self.url, headers=self.headers, data=json.dumps(data))
            response.raise_for_status()

            result = response.json()
            reply = result["candidates"][0]["content"]["parts"][0]["text"]

            # Add model's reply to history
            self.chat_history.append({
                "role": "model",
                "parts": [{"text": reply}]
            })

            return reply

        except requests.exceptions.RequestException as e:
            return f"Network error: {e}"
        except Exception as e:
            return f"Error: {e}"
        


bot = GeminiChatbot("AIzaSyDU3X8MnBiFBvART6R9zMeyuL_UCHeLXTI")
print(bot.send_message("You are the **Driver Monitoring Assistant (DMS)**. "
                    "Your job is to **detect drowsiness, check driver alertness, and ensure road safety**. "
                    "If the driver reports feeling tired, recommend stopping for rest or drinking coffee. "
                    "Ask how long they have been driving and give advice based on their response. "
                    "Keep responses ** very short and direct**."))
print(bot.send_message("quuuuux"))
print(bot.send_message("im distracted"))


