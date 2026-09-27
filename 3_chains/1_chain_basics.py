from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv(override=True)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human","Tell me {joke_count} jokes.")
    ]
)

model = ChatOpenAI(model="gpt-6-luna")

chain = prompt_template | model 

result = chain.invoke({"topic": "programming", "joke_count": 3})
print(result.content)

