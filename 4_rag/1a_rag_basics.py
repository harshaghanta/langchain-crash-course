import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

TEXT_EMBEDDING_MODEL = "openrouter/openai/text-embedding-3-small"

load_dotenv(override=True)

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

if not os.path.exists(persistent_directory):
    print(f"The directory '{persistent_directory}' does not exist. Initializing the vector store...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file '{file_path}' does not exist. Please check the path."
        )
    
    loader = TextLoader(file_path)
    documents = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
    chunked_documents = text_splitter.split_documents(documents)
    
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(chunked_documents)}")
    print(f"Sample chunk:\n{chunked_documents[0].page_content[:500]}...\n")
    
    
    
    print("\n--- Creating vector store ---")
    
    
    embeddings = OpenAIEmbeddings(
            model=TEXT_EMBEDDING_MODEL,
        )
    db = Chroma.from_documents(chunked_documents, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating vector store ---")

else:
    print("Vector store already exists. No Initialization needed.")
    