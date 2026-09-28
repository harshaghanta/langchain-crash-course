import os

from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    # SentenceTransformersTokenTextSplitter,
    TextSplitter,
    TokenTextSplitter,
)
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

TEXT_EMBEDDING_MODEL = "openrouter/openai/text-embedding-3-small"

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "romeo_and_juliet.txt")
db_dir = os.path.join(current_dir, "db")

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"The file {file_path} does not exist. Please check the path."
    )
    
loader = TextLoader(file_path= file_path)
documents = loader.load()

embeddings = OpenAIEmbeddings(model= TEXT_EMBEDDING_MODEL)


def create_vector_store(docs, store_name):
    persistent_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_directory):
        print(f"\n---- Creating vector store {store_name} ----")
        
        Chroma.from_documents(docs, embeddings, persist_directory= persistent_directory)
        print(f"\n---- Finished Creating vector store {store_name} ----")
    else:
        print(f"Vector store {store_name} already exists. No need to initialize.")

    
    print(f"{docs} {store_name}" )


print("\n-----Using Character-based Splitting-----")
char_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
char_docs = char_splitter.split_documents(documents)
create_vector_store(char_docs, "chroma_db_char")


# print("\n-----Using Sentence -based Splitting-----")
# sent_splitter = SentenceTransformersTokenTextSplitter(chunk_size=1000)
# sent_docs = sent_splitter.split_documents(documents)
# create_vector_store(sent_docs, "chroma_db_sent")


print("\n-----Using Token-based Splitting-----")
token_splitter = TokenTextSplitter(chunk_overlap=0, chunk_size=512)
token_docs = token_splitter.split_documents(documents)
create_vector_store(token_docs, "chroma_db_token")


print("\n-----Using Recursive Character-based Splitting-----")
rec_char_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
rec_char_docs = rec_char_splitter.split_documents(documents)
create_vector_store(rec_char_docs, "chroma_db_rec_char")


print("\n------ Using custom splitting -------")

class CustomTextSplitter(TextSplitter):
    def split_text(self, text):
        return [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    
custom_splitter = CustomTextSplitter()
custom_docs = custom_splitter.split_documents(documents)
create_vector_store(custom_docs, "chroma_db_custom")

def query_vector_store(store_name, query):
    persist_directory = os.path.join(db_dir, store_name)
    
    if os.path.exists(persist_directory):
        
        db = Chroma(embedding_function= embeddings, persist_directory=persist_directory)
        retriever = db.as_retriever(
            search_type ="similarity_score_threshold",
            search_kwargs =  {"k": 1, "score_threshold": 0.1}
        )
        
        relevant_docs = retriever.invoke(query)
        print(f"\n-----Relevant documents for store name: {store_name}-----")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"Document {i}: \n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    else:
        print(f"Vector store {store_name} doesn't exist")

query = "How did Juliet die?"

query_vector_store("chroma_db_char", query)
query_vector_store("chroma_db_token", query)
query_vector_store("chroma_db_rec_char", query)
query_vector_store("chroma_db_custom", query)

