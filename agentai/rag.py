from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import DeepInfraEmbeddings
from langchain_community.document_loaders import TextLoader
import os
import shutil

# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()
API_KEY = os.getenv("DEEPINFRA_API_KEY")


class RAG():
    def __init__(self, persist_directory="./memoryDB", document="./agentai/teste_rag.txt", force_rebuild: bool = False):
        self.document = document # I think it won't be needed in the future
        self.persist_directory = persist_directory
        self.collection_name = "long-term-memory"
        self.embedding = DeepInfraEmbeddings(model_id="BAAI/bge-base-en-v1.5", deepinfra_api_token=API_KEY) # which is better?
        self.retriever = None

        if force_rebuild and os.path.exists(self.persist_directory):
            print(f"force_rebuild is True. Removing existing directory: {self.persist_directory}")
            shutil.rmtree(self.persist_directory)

        if os.path.exists(self.persist_directory) and os.path.isdir(self.persist_directory):
            print(f"Directory '{self.persist_directory}' found. Loading existing database.")
            self._load()
        else:
            print(f"Directory '{self.persist_directory}' not found. Creating a new one.")
            self._build()
     

    def _build(self):
        if not os.path.exists(self.document):
            print(f"Error: Document file not found at {self.document}")
            return
        
        try:
            loader = TextLoader(self.document, encoding="utf-8")
            docs = loader.load()
            
            # create chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            doc_splits = text_splitter.split_documents(docs)
            
            # using vectorStore Chroma
            self.vectorstore = Chroma.from_documents( # OR chroma_client.create_collection() IF NOT FROM DOCUMENTS
                documents=doc_splits,
                collection_name=self.collection_name,
                embedding = self.embedding,
                persist_directory=self.persist_directory,
            )

            self.retriever = self.vectorstore.as_retriever(search_kwargs={'k': 6})
            print("Vector database built.\n")
        except Exception as e:
            print(f"Error while building vector database: {e}")

    def _load(self):
        try:
            self.vectorstore = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding)
            self.retriever = self.vectorstore.as_retriever(search_kwargs={'k': 6})
            print("Vector database loaded successfully.")
        except Exception as e:
            print(f"Error while loading vector database: {str(e)}")


    def store(self, texts: list[str]):
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        document_chunks = splitter.create_documents(texts)
        self.vectorstore.add_documents(documents=document_chunks)

    
    def retrieve(self, query: str):
        # NOVIDADE: Adicionar a instrução para otimizar a busca
        instructional_query = f"Represent this sentence for searching relevant passages: {query}"
        
        print(f"\n\n RAG EXECUTED WITH INSTRUCTIONAL QUERY: '{instructional_query}'\n")
        
        # Usar a query com a instrução na busca
        results = self.retriever.invoke(instructional_query)
        
        if not results:
            print("No results found in RAG.\n\n")
            return "No relevant solution was found in the knowledge base. Please proceed with an alternative strategy."

        print(f"RAG Results Found: {[doc.page_content for doc in results]} \n\n")
        return "\n".join([doc.page_content for doc in results])
    