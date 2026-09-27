import os
from langchain.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv


TEXT_EMBEDDING_MODEL = "openrouter/openai/text-embedding-3-small"

load_dotenv(override=True)

current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "books")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

print(f"Books directory: {books_dir}")
print(f"Persistent directory: {persistent_directory}")

if not os.path.exists(persistent_directory):
    print("Persistent directory doesn't exist. Initializing vector store...")


    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"The directory {books_dir} doesn't exist. Please check the path"
        )
        
    book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]
    
    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        loader = TextLoader(file_path= file_path)
        book_docs = loader.load()
        
        for book_doc in book_docs:
            book_doc.metadata = {"source": book_file}
            documents.append(book_doc)
            
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunked_docs = text_splitter.split_documents(documents)
    
    print("\n---- Documents Chunks Information ----")
    print(f"Number of document chunks: {len(chunked_docs)}")
    
    print("---- Creating and Persisting to vector store ----")
    
    embedding = OpenAIEmbeddings(model= TEXT_EMBEDDING_MODEL)
    
    chroma = Chroma.from_documents(chunked_docs, embedding= embedding, persist_directory= persistent_directory)
    print("\n---- Finished creating and persisting vector store ----")
    
else:
    print("Vector store already exists. No need to initialize.")