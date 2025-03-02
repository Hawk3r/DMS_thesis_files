import requests

def chat_with_llama():
    print("Chatbot using Llama3:8b via API. Type 'exit' to quit.\n")

    messages = []  # Store conversation history
    url = "http://localhost:11434"  # Ollama API endpoint

    while True:
        try:
            user_input = input("You: ")  # Ensure this works in VS Code
            if user_input.lower() == "exit":
                print("Goodbye!")
                break

            # Add user message to conversation history
            messages.append({"role": "user", "content": user_input})

            # Prepare API request payload
            payload = {
                "model": "llama3:8b",
                "messages": messages,
                "stream": True  # Enables real-time response streaming
            }

            # Send request to Ollama API
            response = requests.post(url, json=payload, stream=True)

            # Check response status
            if response.status_code == 200:
                print("Bot: ", end="", flush=True)
                bot_response = ""

                # Stream response output in real-time
                for line in response.iter_lines():
                    if line:
                        data = line.decode("utf-8").strip()
                        bot_response += data
                        print(data, end="", flush=True)  # Print as it arrives

                print("\n")  # Newline after complete response

                # Add bot response to conversation history
                messages.append({"role": "assistant", "content": bot_response})
            else:
                print("Error:", response.text)

        except requests.exceptions.RequestException as e:
            print("Connection Error:", e)
            break

if __name__ == "__main__":
    chat_with_llama()
