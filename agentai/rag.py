from typing import Literal
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from typing import List, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.chat_models import ChatDeepInfra
from langchain_community.embeddings import DeepInfraEmbeddings
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langgraph.graph.message import add_messages
from langchain import hub
from langchain.agents import create_react_agent, initialize_agent
from langchain_community.tools.tavily_search import TavilySearchResults
import json
import os
import warnings

# Ignorar todos os avisos de depreciação
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Defina sua chave de acesso 
load_dotenv()
API_KEY = os.getenv("API_KEY")


class RAG():
    def __init__(self, executor, document, persist_directory):
        self.executor = executor
        self.document = document
        self.persist_directory = persist_directory # folder name to save our vector store
     

    def rag(self):
        urls = [
            self.document
        ]
        
        loader = UnstructuredURLLoader(urls=urls)
        docs = loader.load()
        
        # create chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
        doc_splits = text_splitter.split_documents(docs)
        
        # using vectorStore Chroma
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="docs",
            embedding = DeepInfraEmbeddings(model_id="BAAI/bge-base-en-v1.5", deepinfra_api_token=API_KEY), # obligatory?
            persist_directory=self.persist_directory,
        )

        retriever = vectorstore.as_retriever()
        return retriever

    def retrieve_context(query: str):
        """Search for relevant context using RAG"""
        global retriever
        results = retriever.invoke(query)
        return "\n".join([doc.page_content for doc in results])
    





# I can use state for document. pdf or vectorstore file? is a thing to find out!
# retriever = rag(state["documento"])
