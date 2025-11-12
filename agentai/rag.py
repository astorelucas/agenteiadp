from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DeepInfraEmbeddings
from langchain_community.document_loaders import TextLoader
import os
import threading

load_dotenv()
API_KEY = os.getenv("DEEPINFRA_API_KEY")


class RAG():

    #fazendo inicialização lazy, pra evitar problemas
    def __init__(self, document="./agentai/base_rag.txt", collection_name="long-term-memory", persist_directory="./agentai/FAISS_DB"):
        self.document = document 
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.index_path = os.path.join(self.persist_directory, self.collection_name)
        
        self._embedding = None
        self._vectorstore = None
        self._retriever = None
        
        # evitar o fim da execução por problema de threading
        self._init_lock = threading.Lock()

    def _initialize_components(self):
        if self._retriever:
            return

        with self._init_lock:
            if self._retriever:
                return

            try:
                self._embedding = DeepInfraEmbeddings(model_id="BAAI/bge-base-en-v1.5", deepinfra_api_token=API_KEY)

                if os.path.exists(self.index_path):
                    print(f"Diretório de FAISS já existente {self.index_path}")
                    self._vectorstore = FAISS.load_local(
                        self.index_path, 
                        self._embedding,
                        allow_dangerous_deserialization=True
                    )
                else:
                    print(f"Nenhum índice FAISS encontrado. Construindo novo índice de {self.document}")
                    self._build_from_docs()

                if self._vectorstore:
                    self._retriever = self._vectorstore.as_retriever(search_kwargs={'k': 5})
                else:
                    print("Falha ao inicializar o vectorstore.")
            
            except Exception as e:
                print(f"Erro durante a inicialização do RAG: {e}")
                self._embedding = None
                self._vectorstore = None
                self._retriever = None

    def _build_from_docs(self):
        if not os.path.exists(self.document):
            print(f"Erro, documento não encontrado em {self.document}")
            return
        
        try:
            loader = TextLoader(self.document, encoding="utf-8")
            docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            doc_splits = text_splitter.split_documents(docs)
            
            self._vectorstore = FAISS.from_documents(doc_splits, self._embedding)
            
            os.makedirs(self.persist_directory, exist_ok=True)
            self._vectorstore.save_local(self.index_path)
            
        except Exception as e:
            print(f"Erro ao construir o índice FAISS: {e}")

    # @property é bom pra evitar inicializações inválidos e evitar dependência de IFs.
    @property
    def retriever(self):
        if self._retriever is None:
            self._initialize_components()
        return self._retriever

    @property
    def vectorstore(self):
        if self._vectorstore is None:
            self._initialize_components()
        return self._vectorstore

    def store(self, text_content: str):
        if self.vectorstore is None:
             print("Erro: Vectorstore não pôde ser inicializado. 'store' falhou.")
             return

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)        
        chunks = splitter.split_text(text_content)
        
        self.vectorstore.add_texts(texts=chunks)
        self.vectorstore.save_local(self.index_path)
        self._retriever = self.vectorstore.as_retriever(search_kwargs={'k': 5})
        print("Novos documentos foram adicionados e o retriever foi atualizado.")

    
    def retrieve(self, query: str):
        if self.retriever is None:
             return "RAG retriever failed to initialize."

        instructional_query = f"Represent this sentence for searching relevant passages: {query}"
        print(f"\n\n[RAG] EXECUTED WITH INSTRUCTIONAL QUERY: '{instructional_query}'\n")
        
        try:
            results = self.retriever.invoke(instructional_query)
            
            if not results:
                print("No results found in RAG.\n\n")
                return "No relevant solution was found in the knowledge base."

            print(f"Rag result:\n{results}\n")
            return "\n".join([doc.page_content for doc in results])
        
        except Exception as e:
            print(f"[RAG] Erro durante a execução de 'invoke': {e}")
            return f"RAG retrieve failed during invoke: {e}"