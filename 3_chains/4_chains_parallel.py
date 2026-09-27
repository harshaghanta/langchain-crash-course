from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableSequence, RunnableLambda, RunnableParallel
from langchain.schema.output_parser import StrOutputParser

load_dotenv(override=True)

model = ChatOpenAI(model="gpt-6-luna")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert product reviewer"),
        ("human","List the main features of the product {product_name}.")
    ]
)

def analyze_pros(features):
    pros_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer"),
            ("human", "Given these features: {features} list the pros of these features.")
        ]
    )
    return pros_template.format_prompt(features=features)

def analyze_cons(features):
    cons_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer"),
            ("human", "Given these features: {features} list the cons of these features.")
        ]
    )
    return cons_template.format_prompt(features=features)

def combine_pros_cons(pros, cons):
    return f"Pros: \n{pros}\n\nCons: \n{cons}"

pros_branch = (
    RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser()
)

cons_branch = (
    RunnableLambda(lambda x: analyze_cons(x)) | model | StrOutputParser()
)



chain = (
    prompt_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"pros": pros_branch, "cons": cons_branch})
    | RunnableLambda(lambda x: combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"]))
)

result = chain.invoke({"product_name": "Macbook Pro"})

print(result)