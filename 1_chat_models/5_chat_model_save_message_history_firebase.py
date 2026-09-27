from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
load_dotenv(override=True)

PROJECT_ID = "langchain-demo-6119c"
SESSION_ID = "user1_session"
COLLECTION_NAME = "chat_history"

print("Initializing firestore client...")
client = firestore.Client(project=PROJECT_ID)

print("Initializing Firestore Chat Message History...")
chat_history = FirestoreChatMessageHistory(
    session_id = SESSION_ID,
    collection = COLLECTION_NAME,
    client = client
)

print("Chat History initialized successfully.")
print("Current Chat History:", chat_history.messages)

model = ChatOpenAI(model="gpt-6-luna")

print("Starting chatting with the AI. Type 'exit' to quit.")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Append the user's message to the chat history
    chat_history.add_user_message(user_input)

    # Get the AI's response
    response = model.invoke(chat_history.messages)

    # Append the AI's response to the chat history
    chat_history.add_ai_message(response.content)

    print(f"AI: {response.content}")