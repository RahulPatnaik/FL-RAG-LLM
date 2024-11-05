from langchain_core.globals import set_verbose, set_debug
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

set_debug(True)
set_verbose(True)

class ChatPDF:
    def __init__(self, llm_model: str = "mistral"):
        self.model = ChatOllama(
            model=llm_model,
            num_gpu=1,
            num_thread=4,
            temperature=0.7,
            num_predict=2048,
            top_k=30,
            repeat_penalty=1.1
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50
        )
        
        self.prompt = ChatPromptTemplate.from_template("""
            You are a helpful assistant that provides concise answers about the PDF document.
            
            Context: {context}
            Question: {question}
            
            Please provide a clear and concise answer based on the context provided above.
            If the context doesn't contain relevant information, please say so.
            """)
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}  # Changed to CPU for reliability
        )

        self.vector_store = None
        self.retriever = None
        self.chain = None

    def ingest(self, pdf_file_path: str):
        try:
            logger.info(f"Loading PDF from: {pdf_file_path}")
            docs = PyPDFLoader(file_path=pdf_file_path).load()
            logger.info(f"Loaded {len(docs)} pages")
            
            chunks = self.text_splitter.split_documents(docs)
            logger.info(f"Split into {len(chunks)} chunks")
            
            chunks = filter_complex_metadata(chunks)
            
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory="chroma_db"
            )
            self.vector_store.persist()
            logger.info("Vector store created and persisted")
            
            return True
        except Exception as e:
            logger.error(f"Error in ingest: {str(e)}")
            raise e

    def ask(self, query: str):
        try:
            if not self.vector_store:
                logger.info("Loading existing vector store")
                self.vector_store = Chroma(
                    persist_directory="chroma_db", 
                    embedding_function=self.embeddings
                )
            
            # Get relevant documents
            retrieved_docs = self.vector_store.similarity_search(
                query,
                k=4
            )
            
            if not retrieved_docs:
                logger.warning("No relevant documents found")
                return "I couldn't find any relevant information in the document to answer your question."
            
            # Extract the content from retrieved documents
            context = "\n".join([doc.page_content for doc in retrieved_docs])
            
            # Create the chain for this query
            chain = (
                self.prompt | 
                self.model | 
                StrOutputParser()
            )
            
            # Run the chain with the retrieved context
            response = chain.invoke({
                "context": context,
                "question": query
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error in ask: {str(e)}")
            return f"An error occurred while processing your question: {str(e)}"

    def clear(self):
        try:
            if self.vector_store:
                self.vector_store = None
            logger.info("Cleared vector store")
        except Exception as e:
            logger.error(f"Error in clear: {str(e)}")
            raise e