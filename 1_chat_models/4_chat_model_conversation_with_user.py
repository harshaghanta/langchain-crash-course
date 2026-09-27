from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv(override=True)
model = ChatOpenAI(model="gpt-6-luna")

chat_history = []

system_message = SystemMessage(content="You are a helpful AI assistant")
chat_history.append(system_message)

#Chat loop
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break
    chat_history.append(HumanMessage(content=query))
    response = model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    
    print(f"AI: {response.content}")
    
print("-----------Chat History-----------")
print(chat_history)