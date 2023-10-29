import os
import openai
import sys
sys.path.append('../..')

import panel as pn  # GUI
pn.extension()

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

import datetime
current_date = datetime.datetime.now().date()
if current_date < datetime.date(2023, 9, 2):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"
# print(llm_name)

# from langchain.vectorstores import Chroma
# from langchain.embeddings.openai import OpenAIEmbeddings
# persist_directory = 'docs/chroma/'
# embedding = OpenAIEmbeddings()
# vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# question = "What are major topics for this class?"
# docs = vectordb.similarity_search(question,k=3)
# len(docs)

# from langchain.chat_models import ChatOpenAI
# llm = ChatOpenAI(model_name=llm_name, temperature=0)
# llm.predict("Hello world!")

# from langchain.prompts import PromptTemplate
# template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
# {context}
# Question: {question}
# Helpful Answer:"""
# QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

# # Run chain
# from langchain.chains import RetrievalQA
# question = "Is probability a class topic?"
# qa_chain = RetrievalQA.from_chain_type(llm,
#                                        retriever=vectordb.as_retriever(),
#                                        return_source_documents=True,
#                                        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})


# result = qa_chain({"query": question})
# result["result"]

# from langchain.memory import ConversationBufferMemory
# memory = ConversationBufferMemory(
#     memory_key="chat_history",
#     return_messages=True
# )
# result['answer']

# from langchain.chains import ConversationalRetrievalChain
# retriever=vectordb.as_retriever()
# qa = ConversationalRetrievalChain.from_llm(
#     llm,
#     retriever=retriever,
#     memory=memory
# )

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader

def load_db(file, chain_type, k):
    # load documents
    loader = PyPDFLoader(file)
    documents = loader.load()
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    # define embedding
    embeddings = OpenAIEmbeddings()
    # create vector database from data
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    # define retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # create a chatbot chain. Memory is managed externally.
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0), 
        chain_type=chain_type, 
        retriever=retriever, 
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa 

# import panel as pn
# import param

# class cbfs(param.Parameterized):
#     chat_history = param.List([])
#     answer = param.String("")
#     db_query  = param.String("")
#     db_response = param.List([])
    
#     def __init__(self,  **params):
#         super(cbfs, self).__init__( **params)
#         self.panels = []
#         self.loaded_file = "docs/testdoc/MachineLearning.pdf"
#         self.qa = load_db(self.loaded_file,"stuff", 4)
    
#     def call_load_db(self, count):
#         if count == 0 or file_input.value is None:  # init or no file specified :
#             return pn.pane.Markdown(f"Loaded File: {self.loaded_file}")
#         else:
#             file_input.save("temp.pdf")  # local copy
#             self.loaded_file = file_input.filename
#             button_load.button_style="outline"
#             self.qa = load_db("temp.pdf", "stuff", 4)
#             button_load.button_style="solid"
#         self.clr_history()
#         return pn.pane.Markdown(f"Loaded File: {self.loaded_file}")

#     def convchain(self, query):
#         if not query:
#             return pn.WidgetBox(pn.Row('User:', pn.pane.Markdown("", width=600)), scroll=True)
#         result = self.qa({"question": query, "chat_history": self.chat_history})
#         self.chat_history.extend([(query, result["answer"])])
#         self.db_query = result["generated_question"]
#         self.db_response = result["source_documents"]
#         self.answer = result['answer'] 
#         self.panels.extend([
#             pn.Row('User:', pn.pane.Markdown(query, width=600)),
#             pn.Row('ChatBot:', pn.pane.Markdown(self.answer, width=600, style={'background-color': '#F6F6F6'}))
#         ])
#         inp.value = ''  #clears loading indicator when cleared
#         return pn.WidgetBox(*self.panels,scroll=True)

#     @param.depends('db_query ', )
#     def get_lquest(self):
#         if not self.db_query :
#             return pn.Column(
#                 pn.Row(pn.pane.Markdown(f"Last question to DB:", styles={'background-color': '#F6F6F6'})),
#                 pn.Row(pn.pane.Str("no DB accesses so far"))
#             )
#         return pn.Column(
#             pn.Row(pn.pane.Markdown(f"DB query:", styles={'background-color': '#F6F6F6'})),
#             pn.pane.Str(self.db_query )
#         )

#     @param.depends('db_response', )
#     def get_sources(self):
#         if not self.db_response:
#             return 
#         rlist=[pn.Row(pn.pane.Markdown(f"Result of DB lookup:", styles={'background-color': '#F6F6F6'}))]
#         for doc in self.db_response:
#             rlist.append(pn.Row(pn.pane.Str(doc)))
#         return pn.WidgetBox(*rlist, width=600, scroll=True)

#     @param.depends('convchain', 'clr_history') 
#     def get_chats(self):
#         if not self.chat_history:
#             return pn.WidgetBox(pn.Row(pn.pane.Str("No History Yet")), width=600, scroll=True)
#         rlist=[pn.Row(pn.pane.Markdown(f"Current Chat History variable", styles={'background-color': '#F6F6F6'}))]
#         for exchange in self.chat_history:
#             rlist.append(pn.Row(pn.pane.Str(exchange)))
#         return pn.WidgetBox(*rlist, width=600, scroll=True)

#     def clr_history(self,count=0):
#         self.chat_history = []
#         return 


# cb = cbfs()

# file_input = pn.widgets.FileInput(accept='.pdf')
# button_load = pn.widgets.Button(name="Load DB", button_type='primary')
# button_clearhistory = pn.widgets.Button(name="Clear History", button_type='warning')
# button_clearhistory.on_click(cb.clr_history)
# inp = pn.widgets.TextInput( placeholder='Enter text here…')

# bound_button_load = pn.bind(cb.call_load_db, button_load.param.clicks)
# conversation = pn.bind(cb.convchain, inp) 

# jpg_pane = pn.pane.Image( './img/convchain.jpg')

# tab1 = pn.Column(
#     pn.Row(inp),
#     pn.layout.Divider(),
#     pn.panel(conversation,  loading_indicator=True, height=300),
#     pn.layout.Divider(),
# )
# tab2= pn.Column(
#     pn.panel(cb.get_lquest),
#     pn.layout.Divider(),
#     pn.panel(cb.get_sources ),
# )
# tab3= pn.Column(
#     pn.panel(cb.get_chats),
#     pn.layout.Divider(),
# )
# tab4=pn.Column(
#     pn.Row( file_input, button_load, bound_button_load),
#     pn.Row( button_clearhistory, pn.pane.Markdown("Clears chat history. Can use to start a new topic" )),
#     pn.layout.Divider(),
#     pn.Row(jpg_pane.clone(width=400))
# )
# dashboard = pn.Column(
#     pn.Row(pn.pane.Markdown('# ChatWithYourData_Bot')),
#     pn.Tabs(('Conversation', tab1), ('Database', tab2), ('Chat History', tab3),('Configure', tab4))
# )
# dashboard
import streamlit as st
from PIL import Image
import pandas as pd  # Replace this with your preferred data loading library

class cbfs:
    def __init__(self):
        self.chat_history = []
        self.answer = ""
        self.db_query = ""
        self.db_response = []
        self.loaded_file = "docs/testdoc/MachineLearning.pdf"
        self.qa = load_db(self.loaded_file, "stuff", 4)

    def call_load_db(self, count):
        if count == 0 or not uploaded_file:  # init or no file specified:
            st.markdown(f"Loaded File: {self.loaded_file}")
        else:
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.read())
            self.loaded_file = "temp.pdf"
            st.markdown(f"Loaded File: {self.loaded_file}")
            self.qa = load_db("temp.pdf", "stuff", 4)
            st.button("Load DB", key="load_db_button")  # Clear loading indicator
            self.clr_history()

    def convchain(self, query):
        if not query:
            st.write("User: ", "")
        else:
            result = self.qa({"question": query, "chat_history": self.chat_history})
            self.chat_history.extend([(query, result["answer"])])
            self.db_query = result["generated_question"]
            self.db_response = result["source_documents"]
            self.answer = result['answer']
            st.write("User: ", query)
            st.write("ChatBot: ", self.answer)

    def get_lquest(self):
        if not self.db_query:
            st.markdown("Last question to DB:")
            st.write("no DB accesses so far")
        else:
            st.markdown("DB query:")
            st.write(self.db_query)

    def get_sources(self):
        if not self.db_response:
            return
        st.markdown("Result of DB lookup:")
        for doc in self.db_response:
            st.write(doc)

    def get_chats(self):
        if not self.chat_history:
            st.write("No History Yet")
        else:
            st.markdown("Current Chat History variable")
            for exchange in self.chat_history:
                st.write(exchange)

    def clr_history(self, count=0):
        self.chat_history = []

# # Replace this function with your actual data loading function
# def load_db(file_path, query, num_results):
#     # Example: Load data from a CSV file
#     data = pd.read_csv(file_path)
#     return {"question": query, "answer": "Sample answer", "generated_question": "Generated question", "source_documents": data.head(num_results)}

cb = cbfs()
uploaded_file = None

st.title('ChatWithYourData_Bot')
st.sidebar.title('Configure')

if st.sidebar.button("Load DB"):
    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

if st.sidebar.button("Clear History"):
    cb.clr_history()

query = st.text_input("User:", key="query_input")
if st.button("Send", key="send_button"):
    cb.convchain(query)

st.sidebar.write("Last question to DB:")
cb.get_lquest()

st.sidebar.write("Result of DB lookup:")
cb.get_sources()

st.sidebar.markdown("Chat History:")
cb.get_chats()
st.set_page_config(layout="wide")

# You can add more configuration or customization options for Streamlit here.

# Optionally, you can display an image or other media in your Streamlit app:
image = Image.open('./img/convchain.jpg')
st.image(image, caption='Conversation Chain', use_column_width=True)

# Finally, you can run the Streamlit app by adding the following line:
if __name__ == '__main__':
    st.sidebar.markdown("Chat History:")
    cb.get_chats()
