import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv(override=True)

TEXT_EMBEDDING_MODEL = "openrouter/openai/text-embedding-3-small"

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

embeddings = OpenAIEmbeddings(model=TEXT_EMBEDDING_MODEL)

db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

query = "Who is Odysseus wife?"

retriever = db.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs = {"score_threshold": 0.4, "k": 3}
)

relevant_documents = retriever.invoke(query)

print("\n ---- Relevant Documents ----")

for i, doc in enumerate(relevant_documents, 1):
    print(f"Document {i}:\n {doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    