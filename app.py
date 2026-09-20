import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

chat = client.chats.create(model="gemini-3.6-flash")

print("Apple is ready! Type 'exit' to stop.\n")

while True:
    user_input = input("you: ")

    if user_input.lower() == "exit":
        print("Apple: Goodbye!")
        break

    response = chat.send_message(user_input)
    print("Apple:", response.text)