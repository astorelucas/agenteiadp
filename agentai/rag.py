from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import DeepInfraEmbeddings
from langchain_community.document_loaders import TextLoader
import os

# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()
API_KEY = os.getenv("DEEPINFRA_API_KEY")


class RAG():
    def __init__(self, persist_directory="./memoryDB", document="./agentai/teste_rag.txt"):
        self.document = document 
        self.persist_directory = persist_directory
        self.collection_name = "long-term-memory"
        self.embedding = DeepInfraEmbeddings(model_id="BAAI/bge-base-en-v1.5", deepinfra_api_token=API_KEY) # which is better?
        self.retriever = None

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
            self.vectorstore = Chroma(collection_name=self.collection_name, persist_directory=self.persist_directory, embedding_function=self.embedding)
            self.retriever = self.vectorstore.as_retriever(search_kwargs={'k': 6})
            print("Vector database loaded successfully.")
        except Exception as e:
            print(f"Error while loading vector database: {str(e)}")


    def store(self, texts: list[str]):
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        document_chunks = splitter.create_documents(texts)
        self.vectorstore.add_documents(documents=document_chunks)
        
        self.retriever = self.vectorstore.as_retriever(search_kwargs={'k': 6})
    
        print("Novos documentos foram adicionados e o retriever foi atualizado.")

    
    def retrieve(self, query: str):
        instructional_query = f"Represent this sentence for searching relevant passages: {query}"
        
        print(f"\n\n RAG EXECUTED WITH INSTRUCTIONAL QUERY: '{instructional_query}'\n")
        
        results = self.retriever.invoke(instructional_query)
        
        if not results:
            print("No results found in RAG.\n\n")
            return "No relevant solution was found in the knowledge base. Please proceed with an alternative strategy."

        return "\n".join([doc.page_content for doc in results])
    