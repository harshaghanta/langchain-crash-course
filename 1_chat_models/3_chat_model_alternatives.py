from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv(override=True)

model = ChatAnthropic(model="gemini-2.5-flash")

messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
    AIMessage(content="81 ÷ 9 = 9"),
    HumanMessage(content="What is the result multiplied by 4?")
]

response = model.invoke(messages)
print(f"Content: {response.content}")
