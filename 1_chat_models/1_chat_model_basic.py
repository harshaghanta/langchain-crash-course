from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

model = ChatOpenAI(model="gpt-6-luna")

response = model.invoke("What is 81 divided by 9?")
print("Full result:")
print(response)
print("Content only:")
print(response.content)