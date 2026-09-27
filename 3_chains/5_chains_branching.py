from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableBranch, RunnableSequence, RunnableLambda, RunnableParallel
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv

load_dotenv(override=True)

model = ChatOpenAI(model="gpt-6-luna")

SYSTEM_MESSAGE = "You are a helpful assistant"

positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MESSAGE),
        ("human", "Generate a request for more details for this positive feedback: {feedback}.")
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MESSAGE),
        ("human", "Generate a request for more details for this negative feedback: {feedback}")
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MESSAGE),
        ("human", "Generate a request for more details for this neutral feedback: {feedback}.")
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MESSAGE),
        ("human", "Generate a message to escalate this feedback to a human agent: {feedback}")
    ]
)

classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MESSAGE),
        ("human", "Classify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}")
    ]
)

branches = RunnableBranch(
    (
        lambda x: "positive" in x,
        positive_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "negative" in x,
        negative_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "neutral" in x,
        neutral_feedback_template | model | StrOutputParser()
    ),    
    escalate_feedback_template | model | StrOutputParser()    
)

classification_chain = classification_template | model | StrOutputParser()

chain = classification_chain | branches

positiveReview = "The product is excellent. I really enjoyed using it and found it very helpful."
negativeReview = "The product is terrible. It broke after just one use and the quality is very poor."
neutralReview = "The product is okay. It works as expected but nothing special."

review = neutralReview

result = chain.invoke({"feedback": review})

print(result)