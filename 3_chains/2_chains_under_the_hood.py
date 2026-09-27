from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableSequence, RunnableLambda
from dotenv import load_dotenv


load_dotenv(override=True)

model = ChatOpenAI(model="gpt-6-luna")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human","Tell me {joke_count} jokes.")
    ]
)

format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

response = chain.invoke({"topic": "programming", "joke_count": 3})
print(response)