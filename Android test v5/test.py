# main.py
from client import GeminiChatbot

bot = GeminiChatbot("")

print(bot.send_message("Hello!"))
print(bot.send_message("What's the capital of France?"))
print(bot.send_message("And what language do they speak there?"))

