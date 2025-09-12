from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DeepInfraEmbeddings
from langchain_community.document_loaders import TextLoader
import os

# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()
API_KEY = os.getenv("DEEPINFRA_API_KEY")


class RAG():
    def __init__(self, persist_directory="./memoryDB", document="teste_rag"):
        self.document = document # I think it won't be needed in the future
        self.persist_directory = persist_directory
        self.collection_name = "long-term-memory"
        self.embedding = DeepInfraEmbeddings(model_id="BAAI/bge-base-en-v1.5", deepinfra_api_token=API_KEY) # which is better?

        if os.path.exists(self.persist_directory) and os.path.isdir(self.persist_directory):
            print(f"Directory '{self.persist_directory}' found. Loading existing database.")
            self._load()
        else:
            print(f"Directory '{self.persist_directory}' not found. Creating a new one.")
            self._first_run()
     

    def _first_run(self):

        loader = TextLoader(self.document, encoding="utf-8")
        docs = loader.load()
        
        # create chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
        doc_splits = text_splitter.split_documents(docs)
        
        # using vectorStore Chroma
        self.vectorstore = Chroma.from_documents( # OR chroma_client.create_collection() IF NOT FROM DOCUMENTS
            documents=doc_splits,
            collection_name=self.collection_name,
            embedding = self.embedding,
            persist_directory=self.persist_directory,
        )

        self.retriever = self.vectorstore.as_retriever()

    def _load(self):
        try:
            self.vectorstore = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding)
            self.retriever = self.vectorstore.as_retriever()
            return "Vector database loaded successfully."
        except Exception as e:
            return f"Error while loading vector database: {str(e)}"


    def store(self, texts: list[str]):
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
        document_chunks = splitter.create_documents(texts)
        self.vectorstore.add_documents(documents=document_chunks)


    def retrieve(self, query: str):
        results = self.retriever.invoke(query)
        return "\n".join([doc.page_content for doc in results])
    