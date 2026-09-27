from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


load_dotenv(override=True)
model = ChatOpenAI(model="gpt-6-luna")

messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?")
]

response = model.invoke(messages)
print(f"Content: {response.content}")

messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
    AIMessage(content="81 ÷ 9 = 9"),
    HumanMessage(content="What is the result multiplied by 4?")
]

response = model.invoke(messages)
print(f"Content: {response.content}")
