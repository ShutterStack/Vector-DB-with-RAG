from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

print("Embedding model loaded")

#load the vector store
url = "http://localhost:6333"
collection_name = "gpt_db"

client = QdrantClient(url=url,prefer_grpc= False)

print(client)

db = Qdrant(
    client=client,
    collection_name=collection_name,
    embeddings=embeddings
)

print(db)

query = "What are the limitations of RAG?"

docs = db.similarity_search_with_score(query=query, k=5)

for i in docs:
    doc, score = i
    print({"score":score,"content":doc.page_content,"metadata":doc.metadata})