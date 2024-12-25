import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

with open('bank_faqs.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

faqs = data['bank']['accounts']
documents = [f"Q: {faq[0]}\nA: {faq[1]}" for faq in faqs]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
documents = text_splitter.create_documents(documents)

embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url='http://localhost:11434')

db = FAISS.from_documents(documents, embeddings)

db.save_local("faq_index")
