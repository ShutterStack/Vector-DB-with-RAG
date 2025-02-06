from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader("data.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap = 50   # character size to parse
)

texts = text_splitter.split_documents(documents) #split the documents into chunks

#load the embidding model

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

print("Embedding model loaded")

#load the vector store
url = "http://localhost:6333"
collection_name = "gpt_db"
qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url =url,
    prefer_grpc = False,
    collection_name = collection_name
)

print("Vector store loaded")




