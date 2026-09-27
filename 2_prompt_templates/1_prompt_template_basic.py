from langchain.prompts import ChatPromptTemplate

# template = "Tell me a joke about {topic}."
# prompt_template = ChatPromptTemplate.from_template(template)

# prompt = prompt_template.invoke({"topic": "programming"})
# print(prompt)

messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes.")
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "programming", "joke_count": 3})
print(prompt)