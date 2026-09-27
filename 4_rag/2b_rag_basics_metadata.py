import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings

load_dotenv(override=True)

TEXT_EMBEDDING_MODEL = "openrouter/openai/text-embedding-3-small"

curr_dir = os.path.dirname(os.path.abspath(__file__))

persist_directory = os.path.join(curr_dir, "db", "chroma_db_with_metadata")

embedding = OpenAIEmbeddings(model=TEXT_EMBEDDING_MODEL)


db = Chroma(persist_directory=persist_directory, embedding_function= embedding )

query = "How did Juliet die?"

retriever = db.as_retriever(search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.1, "k": 3},)

matching_docs = retriever.invoke(query)

print("\n---Relevant Documents ----")

for i, doc in enumerate(matching_docs, 1):
    print(f"Document {i}: \n {doc.page_content}\n")
    print(f"Source: {doc.metadata['source']}\n")