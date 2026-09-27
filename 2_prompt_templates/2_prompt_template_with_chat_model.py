from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv(override=True)


template = "Tell me a joke about {topic}."
prompt_template = ChatPromptTemplate.from_template(template)


model = ChatOpenAI(model="gpt-6-luna")
prompt = prompt_template.invoke({"topic": "programming"})
response = model.invoke(prompt)
print(response.content)