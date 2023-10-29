import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

from langchain.document_loaders import PyPDFLoader

# Load PDF
loaders = [
    # Duplicate documents on purpose - messy data
    PyPDFLoader("docs/testdoc/MachineLearning.pdf"),
    PyPDFLoader("docs/testdoc/MachineLearning.pdf"),
    PyPDFLoader("docs/testdoc/MachineLearning2.pdf"),
    PyPDFLoader("docs/testdoc/MachineLearning3.pdf")
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)

splits = text_splitter.split_documents(docs)
# print(len(splits))

from langchain.embeddings.openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()
from langchain.vectorstores import Chroma
persist_directory = 'docs/chroma/'

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

print(vectordb._collection.count())