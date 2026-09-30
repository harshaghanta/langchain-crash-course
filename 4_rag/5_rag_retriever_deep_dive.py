import os

from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv(override=True)

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

embeddings = OpenAIEmbeddings(model= "openrouter/openai/text-embedding-3-small")

db = Chroma(persist_directory=persistent_directory, embedding_function= embeddings)

def query_vector_store(
    store_name, query, embedding_function, search_type, search_kwargs
    ):
    
    persistent_directory = os.path.join(db_dir, store_name)
    if os.path.exists(persistent_directory):
        print(f"\n--- Querying the vector store {store_name} ---")
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_function
        )
        
        retriever = db.as_retriever(
            search_type = search_type,
            search_kwargs = search_kwargs
        )
        print(f"\n----Relevenat documents for Store Name: {store_name}")
        relevant_docs = retriever.invoke(query)
        for i, doc in enumerate(relevant_docs, 1):
            print(f"Document {i}: \n {doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    else:
        print(f"Vector store: {store_name} doesn't exist")
        
query = "How did Juliet die?"

print("\n---- Using Similarity Search ----")
query_vector_store("chroma_db_with_metadata", query, embeddings, "similarity", {"k": 3})

print("\n---- Using Max Marginal Relevance (MMR) ----")
query_vector_store("chroma_db_with_metadata", query, embeddings, "mmr", {"k": 3, "fetch_k": 20, "lambda_mult": 0.5})

print("\n---- Using Similarity Score Threshold ----")
query_vector_store("chroma_db_with_metadata", query, embeddings, "similarity_score_threshold", {"k": 3, "score_threshold": 0.1})
