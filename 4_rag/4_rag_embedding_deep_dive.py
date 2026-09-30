import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings

load_dotenv(override=True)

OPEN_AI_TEXT_EMBEDDING_MODEL = "openrouter/openai/text-embedding-3-small"
# HUGGING_FACE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HUGGING_FACE_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
db_dir = os.path.join(current_dir, "db")

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"The file {file_path} doesn't exist. Please check the path."
    )
    
loader = TextLoader(file_path= file_path)
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

print("\n---Document Chunks Information---")
print(f"Number of document chunks: {len(docs)}")
print(f"Sample Chunk: \n {docs[0].page_content}\n")



def create_vector_store(docs, embeddings, store_name):
    persist_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persist_directory):    
        print(f"---Creating Vector store: {store_name}....")
        Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
        print(f"----Finished creating Vector store: {store_name}....")
    else:
        print(f"---Vector store: {store_name} already exists. No initialization needed.")


print("\n----Using OpenAI embeddings-------")
openai_embeddings = OpenAIEmbeddings(model_name=OPEN_AI_TEXT_EMBEDDING_MODEL)
# create_vector_store(docs, embeddings, "chroma_db_all_minilm_l6_v2")
create_vector_store(docs, openai_embeddings, "chroma_db_openai")

print("\n----Using Hugging Face embeddings-------")
hugging_face_embeddings = HuggingFaceEmbeddings(model_name=HUGGING_FACE_EMBEDDING_MODEL)
# create_vector_store(docs, embeddings, "chroma_db_all_minilm_l6_v2")
create_vector_store(docs, hugging_face_embeddings, "chroma_db_all_mpnet_base_v2")

def query_vector_store(store_name, query, embedding_function):
    persistent_directory = os.path.join(db_dir, store_name)
    if os.path.exists(persistent_directory):
        print(f"\n--- Querying the vector store {store_name} ---")
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_function
        )
        
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.1, "k": 3}
        )
        print(f"\n----Relevenat documents for Store Name: {store_name}")
        relevant_docs = retriever.invoke(query)
        for i, doc in enumerate(relevant_docs, 1):
            print(f"Document {i}: \n {doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    else:
        print(f"Vector store: {store_name} doesn't exist")
        
query = "Who is Odysses's wife?"

# query_vector_store("chroma_db_openai", query, openai_embeddings)

query_vector_store("chroma_db_all_mpnet_base_v2", query, hugging_face_embeddings)
    


