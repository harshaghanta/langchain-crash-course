import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

embeddings = OpenAIEmbeddings(model= "openrouter/openai/text-embedding-3-small")

db = Chroma(persist_directory=persistent_directory, embedding_function= embeddings)

query = "Help me understand LangChain?"

retriever = db.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k": 1}
)

relevant_docs = retriever.invoke(query)

print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    
combined_input = (
    "Here are some documents that might help answer the question: "
    + query
    + "\n\nRelevant Documents:\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs ])
    + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
)

model = ChatOpenAI(model = "gpt-oss-120b-medium")

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input)
]
print("\n---------------------------------------------\n")
# print(combined_input)
result = model.invoke(messages)

# print("\n--- Generated Response ---")

print("Content only:")
print(result.content)