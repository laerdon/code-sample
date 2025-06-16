import requests

# Setup
model = "llama3"  # or any other Ollama-supported model
base_url = "http://localhost:11434/api/chat"
chat_history = []


def chat_with_ollama(user_input):
    chat_history.append({"role": "user", "content": user_input})
    response = requests.post(
        base_url,
        json={
            "model": model,
            "messages": chat_history,
            "stream": False,  # Explicitly set streaming to false
        },
    )

    if response.status_code == 200:
        try:
            data = response.json()
            reply = data["message"]["content"]
            chat_history.append({"role": "assistant", "content": reply})
            return reply
        except (KeyError, ValueError) as e:
            print(f"Error parsing response: {e}")
            print(f"Response content: {response.text}")
            return "Sorry, there was an error processing the response."
    else:
        print(f"Error: Status code {response.status_code}")
        print(f"Response: {response.text}")
        return "Sorry, there was an error connecting to the model."
