from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

MODEL_NAME = "gpt-oss-120b-medium"
# MODEL_NAME = "gpt-6-luna"

model = ChatOpenAI(model=MODEL_NAME)

response = model.invoke("What is 81 divided by 9?")
print("Full result:")
print(response)
print("Content only:")
print(response.content)