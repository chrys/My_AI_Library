import os
import threading
import datetime
from sqlalchemy import make_url
import pandas as pd
from my_library.utilities import read_config
from my_library.logger import setup_logger
from my_library.wp_scraper import scrape_website


# Initialize the logger
logger = setup_logger()


from llama_index.llms.openai import OpenAI
from llama_index.core import (
      VectorStoreIndex,
      Settings,
      SimpleDirectoryReader,
      StorageContext,
      Document,
  )

from llama_index.core.node_parser import (
    SemanticSplitterNodeParser,
)

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from google.genai.types import EmbedContentConfig
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.google_genai import GoogleGenAI

#from llama_index.readers.web import SimpleWebPageReader

from my_library.parse_csv import parse_qa_csv
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

import time
from google.api_core.exceptions import ResourceExhausted


MY_EMBED_DIMENSION = 768

# gemini-embedding-exp-03-07
# it says that output dimension size can be 3072, 1536, or 768
GEMINI_EMBED_DIMENSION = 768 # it seems that the gemini embedding model is 768 dimensions


  # Singleton instance store with thread-safety
class RAGServiceManager:
    _instances = {}
    _lock = threading.RLock()

    @classmethod
    def get_instance(cls, table_name, model_name):
        key = f"{table_name}:{model_name}"
        with cls._lock:
            if key not in cls._instances:
                cls._instances[key] = RAGService(table_name, model_name)
                cls._instances[key].initialize()
            return cls._instances[key]

# Main RAG service class
class RAGService:
    def __init__(self, table_name, model_name):
        self.table_name = table_name
        self.model_name = model_name
        self.logger = setup_logger()

        # Instance attributes (replacing globals)
        self.vector_store = None
        self.index = None
        self.retriever = None
        self.llm = None
        self.embed_model = None
        self.chat_engine = None
        self.initialized = False

        # Thread-local storage for request context
        self.local = threading.local()

    def initialize(self):
        """Initialize RAG components"""
        if self.initialized:
            return True
        self.logger.info(f"Initializing RAG components for {self.model_name}...")
        try:
            # Get connection string
            my_connection_string = read_config('CONNECTIONS', 'postgres2')
            if not my_connection_string:
                self.logger.error("PostgreSQL connection string not found")
                return False
            # Get table name
            self.table_name = read_config('RAG', 'table_name')
            if not self.table_name:
                self.logger.error("Table name not found in environment variables")
                return False
             
            # Initialize LLM based on model selection
            self._initialize_llm()

            # Initialize embeddings
            # self.embed_model = GeminiEmbedding(
            #     model="models/text-embedding-004",
            #     dimensions=EMBED_DIMENSION
            # )
            # if self.model_name.lower() == "gemini":
            #     self.embed_model = GoogleGenAIEmbedding(
            #         model="models/text-embedding-004",
            #         dimensions=MY_EMBED_DIMENSION,
            #     )
            # elif self.model_name.lower() == "local":
            #     self.embed_model = OllamaEmbedding(
            #         model_name="mxbai-embed-large:latest",
            #         embed_batch_size=MY_EMBED_DIMENSION
            #     )
            # else:
            #     logger.error("Invalid model name. Use 'gemini' or 'local'")
            
            # Settings.embed_model = self.embed_model
            # Settings.llm = self.llm
            
            # Initialize vector store
            
            
            url = make_url(my_connection_string)
            if self.model_name.lower() == "gemini":
                embed_dim = GEMINI_EMBED_DIMENSION
                self.logger.info(f"Gemini embedding dimension: {embed_dim}")
            elif self.model_name.lower() == "local":
                embed_dim = MY_EMBED_DIMENSION
                self.logger.info(f"Local embedding dimension: {embed_dim}")
            else:
                self.logger.error("Invalid model name. Use 'gemini' or 'local'")
                return False
            
            self.vector_store = PGVectorStore.from_params(
                database=url.database,
                host=url.host,
                password=url.password,
                port=url.port,
                user=url.username,
                table_name=self.table_name,
                embed_dim=embed_dim,
            )

            # Create index and retriever
            self.index = VectorStoreIndex.from_vector_store(self.vector_store)
            if not self.index:
                self.logger.error("Failed to create index from vector store")
                return False

            self.retriever = self.index.as_retriever(
                similarity_top_k=3,
                search_kwargs={
                    "k": 3,
                    "score_threshold": None,  # Remove any threshold initially
                    "fetch_k": 10  # Fetch more candidates
                },
            )
            
            if not self.retriever:
                self.logger.error("Failed to create retriever")
                return False
            
            # Initialize ChatEngine
            
            self.chat_engine = self.index.as_chat_engine(
                chat_mode="context",
                verbose=True,
                system_prompt="""You are a helpful and friendly chatbot named VasiliasBot assisting customers of Vasilias Weddings.
                                You are a dedicated customer service representative whose primary goal is to provide information and assistance 
                                related to Vasilias Weddings based exclusively on the knowledge provided.

                                Guidelines:
                                1. Answer questions only based on the provided context. Do not use external knowledge.
                                2. If information isn't available, politely suggest contacting Vasilias Weddings directly.
                                3. Use a direct, helpful, and natural conversational style.
                                4. Be concise and avoid unnecessary greetings.
                                5. Show empathy while maintaining professionalism.
                                6. Prioritize accuracy based on the context."""
                            )

            self.initialized = True
            return True

        except Exception as e:
            self.logger.error(f"Error initializing RAG components: {str(e)}")
            return False

    def _initialize_llm(self):
        """Initialize the LLM based on model selection"""
        if self.model_name.lower() == "openai":
            api_key = read_config('AI KEYS', 'openai')
            if not api_key:
                raise ValueError("OpenAI keys not found in environment variables")
            self.llm = OpenAI(model="gpt-3.5-turbo-0125", api_key=api_key)
        elif self.model_name.lower() == "claude":
            api_key = read_config('AI KEYS', 'claude')
            if not api_key:
                raise ValueError("Claude keys not found in environment variables")
            os.environ["ANTHROPIC_API_KEY"] = api_key
            self.llm = Anthropic(model="claude-3-5-sonnet-latest")
        elif self.model_name.lower() == "gemini":
            api_key = read_config('AI KEYS', 'gemini')
            if not api_key:
                raise ValueError("Gemini keys not found in environment variables")
            os.environ["GOOGLE_API_KEY"] = api_key
            self.llm = GoogleGenAI(model="models/gemini-2.5-flash-lite-preview-06-17")
            self.embed_model = GoogleGenAIEmbedding(
            model="gemini-embedding-exp-03-07",
            )
        elif self.model_name.lower() == "local":
            self.llm = Ollama(model="phi3:latest")
            self.embed_model = OllamaEmbedding(
                model_name="mxbai-embed-large:latest",
                embed_batch_size=MY_EMBED_DIMENSION
            )
        else:
            raise ValueError("Invalid model name. Use 'openai', 'claude', or 'gemini'")

        if not self.llm:
            self.logger.error("Failed to initialize LLM")
            raise ValueError("Failed to initialize LLM")
        else: 
            self.logger.info(f"{self.model_name} LLM initialized successfully")
            Settings.llm = self.llm
        
        if not self.embed_model:
            self.logger.error("Failed to initialize embedding model")
            raise ValueError("Failed to initialize embedding model")
        else:
            self.logger.info(f"{self.model_name} embedding model initialized successfully")
            Settings.embed_model = self.embed_model
        
    def ask(self, message):
        """Thread-safe method to ask questions of the RAG system"""
        if not self.initialized and not self.initialize():
            return "System is not properly initialized. Please try again later."
        self.logger.info("New question received")
        try:
            # Validate input 
            if not message or not message.strip():
                return "Please provide a valid question"

            # Create a query-specific prompt (thread-safe)
            # Each thread gets its own prompt instance
            #if not hasattr(self.local, 'prompt'):
                #template = "Based on the following context, please answer the question: {message}\n\nContext: {context}"
                #google gemini template shouldn't exceed 4000 characters
                # template = """
                #     You are a helpful and friendly chatbot named VasiliasBot assisting customers of Vasilias Weddings.
                #     You are a dedicated customer service representative whose primary goal is to provide information and assistance related to Vasilias Weddings based *exclusively* on the knowledge provided in the context below. You must stick to the context. Do not hallucinate answers.

                #     **Instructions:**

                #     1.  **Knowledge Source:** You *must* answer questions *only* based on the information provided in the **Context** section below. Do not use any external knowledge or prior assumptions.
                #     2.  **Handling Missing Information:** If the answer to the user's query cannot be confidently determined from the provided **Context**, politely state that you don't have that specific information based on the available website content. Do not invent an answer. Suggest contacting Vasilias Weddings directly for specifics if appropriate (e.g., "I couldn't find the exact details about that in the information I have access to. For the most accurate information, please reach out to Vasilias Weddings directly.").
                #     3.  **Tone and Style:** Respond in a direct, helpful, and natural conversational style. Aim for a professional yet warm and approachable tone. *Do not use overly formal language or overly verbose explanations.* Be concise and to the point. *Do not include conversational fillers or greetings like "Hi" or "Hello" at the start of your response*.
                #     4.  **Focus and Relevance:** Respect the user's time by focusing on providing the most relevant information first. Only respond to the specific question asked in the **User Query**. Do not provide unsolicited extra information.
                #     5.  **Empathy and Support:** Acknowledge that wedding planning can be complex and potentially stressful. Offer support and understanding in your responses. Strive for a balance between showing support and providing clear, concise answers efficiently.
                #     6.  **Accuracy and Corrections:** Prioritize accuracy based on the **Context**. If the user states something incorrect according to the **Context**, gently correct them using an empathizing and supportive tone.

                #     **Context:**
                #     {context_str}

                #     **User Query:** {query_str}

                #     **Response:**
                #     """
                # self.local.prompt = PromptTemplate(template)

            # Retrieve documents
            logger.info(f"Using retriever with similarity_top_k={self.retriever.similarity_top_k}")
            
            retrieved_nodes = self.retriever.retrieve(message)
            logger.info(f"Retrieved {len(retrieved_nodes)} nodes")
            # Log the first retrieved node if any
            if retrieved_nodes:
                
                logger.info(f"First retrieved node text: {retrieved_nodes[0].node.text[:100]}...")
                logger.info(f"First retrieved node score: {retrieved_nodes[0].score}")
            else:
                logger.warning("No nodes retrieved!")
                return "I couldn't find any relevant information to answer your question."

            # # Format context from retrieved nodes
            # context = "\n".join([node.node.text for node in retrieved_nodes])

            # # Format prompt with context
            # formatted_prompt = self.local.prompt.format(
            #     context=context,
            #     query_str=message)
            
            # Get response using LLM
            #response = self.llm.complete(formatted_prompt)
            response = self.chat_engine.chat(message)
            
            # Return the response
            if not response or not response.response:
                return "I couldn't generate a response based on the provided context."
            else: 
                self.logger.info("Response generated successfully")
                return response.response

        except Exception as e:
            self.logger.error(f"Error processing question: {str(e)}")
            return "I'm sorry, but an error occurred while processing your question."

    def test_retrieval(self):
        """Add this method to RAGService class"""
        # Get a sample embedding
        try:
            sample_text = "Test query"
            embedding = Settings.embed_model.get_text_embedding(sample_text)
            logger.info(f"Query embedding dimension: {len(embedding)}")
            
            # Get table dimension using vec_dim() function
            url = make_url(read_config('CONNECTIONS', 'postgres2'))
            import psycopg2
            with psycopg2.connect(
                dbname=url.database,
                user=url.username,
                password=url.password,
                host=url.host,
                port=url.port
            ) as conn:
                with conn.cursor() as cur:
                    quoted_table = f'"{self.table_name}"'
                    # Also check if we have any vectors stored
                    cur.execute(f"SELECT COUNT(*) FROM {quoted_table}")
                    count = cur.fetchone()[0]
                    logger.info(f"Number of vectors stored: {count}")
                    
                    # Check total rows
                    cur.execute(f"SELECT COUNT(*) FROM {quoted_table}")
                    count = cur.fetchone()[0]
                    logger.info(f"Number of vectors stored: {count}")
                    
                    # Sample some actual data
                    cur.execute(f"SELECT id, text FROM {quoted_table} LIMIT 1")
                    sample = cur.fetchone()
                    if sample:
                        logger.info(f"Sample document ID: {sample[0]}")
                        logger.info(f"Sample text: {sample[1][:100]}...")
                        
            # 3. Test actual retrieval
            if self.retriever:
                logger.info("Testing retriever with sample query...")
                nodes = self.retriever.retrieve("wedding venue")
                logger.info(f"Retrieved {len(nodes)} nodes for test query")
                if nodes:
                    logger.info(f"First retrieved node score: {nodes[0].score}")
                    logger.info(f"First retrieved text: {nodes[0].node.text[:100]}...")
            else:
                logger.error("Retriever not initialized")
        except Exception as e:
            logger.error(f"Error in test_retrieval: {str(e)}")
            
    def cleanup(self):
        """Clean up resources"""
        # Implementation depends on what resources need cleanup
        # For example, close database connections
        self.initialized = False
        self.logger.info("RAG service resources cleaned up")

    # Convenience functions for backward compatibility
    def initialize_rag_components(table_name, model_name):
      """Legacy function for backward compatibility"""
      service = RAGServiceManager.get_instance(table_name, model_name)
      if service.initialized or service.initialize():
          return service.vector_store, service.index, service.retriever
      return None, None, None

    def ask_RAG(message, table_name="default_table", model_name="openai"):
      """Legacy function for backward compatibility"""
      service = RAGServiceManager.get_instance(table_name, model_name)
      return service.ask(message)

  #3. Usage in Django Views:

  # Example usage in Django view
#   from my_library.rag_functions import RAGServiceManager

#   def chatbot_view(request):
#       if request.method == 'POST':
#           user_message = request.POST.get('message', '')

#           # Get RAG service for this request
#           rag_service = RAGServiceManager.get_instance("my_vector_table", "openai")

#           # Get response
#           response = rag_service.ask(user_message)

#           return JsonResponse({"response": response})

#       return render(request, 'chatbot/chat.html')
def embed_with_retry(text, embed_model, max_retries=5, delay=1):
    for attempt in range(max_retries):
        try:
            return embed_model.get_text_embedding(text)
        except ResourceExhausted as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(delay * (attempt + 1))  # Exponential backoff
        except Exception as e:
            logger.error(f"Error during embedding: {e}")
            return None
    return None

def index_data(sample_data, input_type, llm_model, my_embed_model):
    """
    Index data from various sources using specified LLM model.
    
    Args:
        sample_data (str): Path to file/directory or URL
        input_type (str): Type of input ("file", "url", "csv", "directory")
        llm_model (str): Name of the LLM model to use ("gemini", "claude", "openai")
    
    Returns:
        list: List of processed documents
    
    Raises:
        ValueError: If invalid model name is provided
        FileNotFoundError: If the specified file doesn't exist
    """
   # Initialize LLM and Embedding models
    
    if llm_model.lower() == "openai":
        OPENAI_API_KEY = read_config('AI KEYS', 'openai')
        if not OPENAI_API_KEY:
            logger.error("OpenAI keys not found in environment variables")
            raise ValueError("OpenAI keys not found in environment variables")
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        my_llm = OpenAI(model="gpt-3.5-turbo-0125")
        Settings.llm = my_llm 
        logger.info("OpenAI LLM initialized successfully")
    elif llm_model.lower() == "claude":
        CLAUDE_API_KEY = read_config('AI KEYS', 'claude')
        if not CLAUDE_API_KEY:
            logger.error("Claude keys not found in environment variables")
            raise ValueError("Claude keys not found in environment variables")
        os.environ["ANTHROPIC_API_KEY"] = CLAUDE_API_KEY      
        my_llm = Anthropic(model="claude-3-5-sonnet-latest")
        Settings.llm = my_llm
        logger.info("Claude LLM initialized successfully")
    elif llm_model.lower() == "gemini":
        GOOGLE_API_KEY = read_config('AI KEYS', 'gemini')
        if not GOOGLE_API_KEY:
            logger.error("Gemini keys not found in environment variables")
            raise ValueError("Gemini keys not found in environment variables")
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        my_llm = GoogleGenAI(model = "/models/gemini-2.5-flash-lite-preview-06-17")
        Settings.llm = my_llm
        logger.info("Gemini LLM initialized successfully")
    elif llm_model.lower() == "local":
        Settings.llm = Ollama(model="phi4:latest")
        logger.info("Local LLM initialized successfully")
    else:
        logger.error("Invalid model name. Use 'gemini', 'claude', or 'openai'")
        raise ValueError("Invalid model name. Use 'gemini', 'claude', or 'local'")

    # Initialize Embedding model
    if llm_model.lower() == "openai":
        None
        #TODO code for OpenAI embedding model 
    elif llm_model.lower() == "claude":
        None
        #TODO code for Claude embedding model
    elif llm_model.lower() == "gemini":
        my_embed_model = GoogleGenAIEmbedding(
            model="gemini-embedding-exp-03-07",
        )
    elif llm_model.lower() == "local":
        my_embed_model = OllamaEmbedding(
            model_name= my_embed_model,
        )
        Settings.embed_model = my_embed_model
        logger.info("Local embedding model initialized successfully")
    else:
        raise ValueError("Invalid model name. Use 'gemini', 'claude', or 'local'")

    splitter = SemanticSplitterNodeParser(
        buffer_size=1, 
        breakpoint_percentile_threshold=95, 
        embed_model=my_embed_model,
    )
    Settings.node_parser = splitter
    Settings.embed_model = my_embed_model

     # Check if input is URL
    if input_type == "url":
        # Check if URL is valid
        if not sample_data.startswith("http://") or sample_data.startswith("https://"):
            logger.error("Invalid URL provided")
            raise ValueError("Invalid URL provided")
        try:
            #documents = SimpleWebPageReader(html_to_text=True).load_data(urls=[sample_data])
            documents = scrape_website(sample_data)
            if not documents:
                logger.error("No documents found in the URL")
                raise ValueError("No documents found in the URL")
            else:
                logger.info(f"Documents loaded successfully from {sample_data}")
                return documents
        except Exception as e:
            logger.error(f"\nError occurred: {str(e)}")
            return None 
    # Process local file or directory
    elif input_type == "file" or type == "directory":
        if os.path.isfile(sample_data):
            reader = SimpleDirectoryReader(input_files=[sample_data])
            documents = reader.load_data()
            if not documents:
                logger.error("No documents found in the file")
                raise ValueError("No documents found in the file")
            else:
                logger.info(f"Documents loaded successfully from {sample_data}")
                #log the beginning of the document
                #logger.info(f"First document: {documents[0].text[:50]}...")
            return documents
        elif os.path.isdir(sample_data):
            reader = SimpleDirectoryReader(input_dir=sample_data)
            documents = reader.load_data()
            if not documents:
                logger.error("No documents found in the directory")
                raise ValueError("No documents found in the directory")
            else:   
                logger.info(f"Documents loaded successfully from {sample_data}")
                #log the beginning of the document
                logger.info(f"First document: {documents[0].text[:50]}...")
                return documents
        else:
                logger.error("File or directory not found")
                raise FileNotFoundError(f"File or directory not found: {sample_data}")  
    elif input_type == "csv":
        # Parse CSV file
        df = pd.read_csv(sample_data)
        documents = []
        for index, row in df.iterrows():
            question = str(row['Questions'])
            answer = str(row['Answers'])
            # Create a structured text format
            doc_text = f"Question: {question}\nAnswer: {answer}"
            # Add metadata for potential filtering or reference
            metadata = {"faq_question": question, "source_row": index}
            documents.append(Document(text=doc_text, metadata=metadata))
        logger.info(f"Parsed {len(documents)} documents from CSV file {sample_data}")       
        if not documents:
            logger.error("No documents found in the CSV file")
            raise ValueError("No documents found in the CSV file")
        else:
            logger.info(f"Documents loaded successfully from {sample_data}")
            logger.info(f"First document: {documents[0].text[:50]}...")
            return documents
    else:
        logger
        raise ValueError("Invalid input. Please provide a url or a valid file or a valid directory path")
        return None
    

def store_documents(my_table_name, my_documents, my_model):
    """
    Store documents in the specified table.
    
    Args:
        my_table_name (str): Name of the table to store documents
        my_documents (list): List of documents to store
        my_model (str): Name of the LLM model to use ("gemini", "claude", "local")
    
    Returns:
        bool: True if documents are stored successfully, False otherwise
    """
    
    if my_model.lower() == "gemini":
        api_key = read_config('AI KEYS', 'gemini')
        if not api_key:
            raise ValueError("Gemini keys not found in environment variables")
        os.environ["GOOGLE_API_KEY"] = api_key
        llm = GoogleGenAI(model="models/gemini-2.5-flash-lite-preview-06-17")
        my_embed_model = GoogleGenAIEmbedding(
                model="gemini-embedding-exp-03-07",
            )
    elif my_model.lower() == "local":
        llm = Ollama(model="phi4:latest")
        my_embed_model = OllamaEmbedding(
            model_name="quentinz/bge-base-zh-v1.5:latest",
            )
    else:
        raise ValueError("Invalid model name. Use 'gemini', 'claude', or 'local'")      
    
    #TODO 
    # What should the system do if my_table_name already exists?
    
   
            
    Settings.embed_model = my_embed_model
    Settings.llm = llm
            
    try:
        # Get connection string
        my_connection_string = read_config('CONNECTIONS', 'postgres3')
        if not my_connection_string:
            logger.error("PostgreSQL connection string not found")
            raise ValueError("PostgreSQL connection string not found")
        # Initialize vector store
        url = make_url(my_connection_string)
        logger.info(f"Connecting to PostgreSQL database: {url.database} at {url.host}:{url.port} as user {url.username}")
        logger.info(f"Using table name: {my_table_name}")
        if my_model.lower() == "gemini":
            my_vector_store = PGVectorStore.from_params(
                database=url.database,
                host=url.host,
                password=url.password,
                port=url.port,
                user=url.username,
                table_name=my_table_name,
                embed_dim=GEMINI_EMBED_DIMENSION,
            )
        elif my_model.lower() == "local":
            my_vector_store = PGVectorStore.from_params(
                database=url.database,
                host=url.host,
                password=url.password,
                port=url.port,
                user=url.username,
                table_name=my_table_name,
                embed_dim=MY_EMBED_DIMENSION,
            )
        else:
            logger.error("Invalid model name. Use 'gemini' or 'local'")
            raise ValueError("Invalid model name. Use 'gemini' or 'local'")
       
        if not my_vector_store:
            logger.error("Failed to create vector store")
            raise ValueError("Failed to create vector store")
        logger.info(f"Vector store created successfully: {my_vector_store.table_name}")
        # Store documents
        my_storage_context = StorageContext.from_defaults(
            vector_store=my_vector_store,
        )
        if not my_storage_context:
            logger.error("Failed to create storage context")
            raise ValueError("Failed to create storage context")
        else:
            logger.info(f"Storing documents in table: {my_vector_store.table_name}")
            
        # Create embeddings with retry and add to vector store
        nodes = []
        for doc in my_documents:
            embedding = embed_with_retry(doc.text, my_embed_model)
            if embedding:
                # Create a LlamaIndex Node object
                node = Document(
                    text=doc.text,
                    embedding=embedding,
                    metadata=doc.metadata,
                    id_=doc.id_
                )
                nodes.append(node)
            else:
                logger.warning(f"Failed to embed document {doc.id_}")
        
        # Store embeddings in vector store
        if nodes:
            my_vector_store.add(nodes)
            logger.info(f"Stored {len(nodes)} documents in table: {my_vector_store.table_name}")
        else:
            logger.warning("No nodes to store in vector store.")
        return True
    except Exception as e:
        logger.error(f"Error storing documents: {str(e)}")
        return False
         
    #     my_index = VectorStoreIndex.from_documents(
    #         documents=my_documents,
    #         storage_context=my_storage_context, 
    #         show_progress=True,
    #     )
    #     if not my_index:
    #         logger.error("Failed to create index from documents")
    #         raise ValueError("Failed to create index from documents")
    #     else:
    #         logger.info(f"Index created successfully: {my_index.summary}")
    #         return my_index        
    # except Exception as e:
    #     logger.error(f"Error storing documents: {str(e)}")
    #     return False

def test_RAG(table, model):
    # Get RAG service for table and model "gemini"
    if model.lower() == "gemini":
        GOOGLE_API_KEY = read_config('AI KEYS', 'gemini')
        if not GOOGLE_API_KEY:
            logger.error("Gemini keys not found in environment variables")
            raise ValueError("Gemini keys not found in environment variables")
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        rag_service = RAGServiceManager.get_instance(table, "gemini")
    elif model.lower() == "local":
        rag_service = RAGServiceManager.get_instance(table, "local")
    else:
        logger.error("Invalid model name. Use 'gemini' or 'local'")
        raise ValueError("Invalid model name. Use 'gemini' or 'local'")
    
    logger.info(f"Testing RAG service for table: {table}")
    rag_service.test_retrieval()
    
    # List of questions to ask
    questions = [
        "Do you have experience with weddings for couples of different nationalities or religions?", 
        "Can you provide references or testimonials from past clients?",
        "How many guests does a cocktail jar serve? and what are the options?",
        #"what is the capital of France?", OK
    ]
    
    # Ask each question and print the reply
    for question in questions:
        response = rag_service.ask(question)
        print(f"Question: {question}")
        print(f"Reply: {response}\n")

def test_embedding():
    # Test the embedding model
    my_embed_model = OllamaEmbedding(
        model_name="mxbai-embed-large:latest",
        embed_batch_size=MY_EMBED_DIMENSION)
    Settings.embed_model = my_embed_model
    logger.info("Local embedding model initialized successfully")
    
    # Test the embedding model with a sample text
    sample_text = "This is a test sentence for embedding."
    embedding = my_embed_model.get_text_embedding(sample_text)
    logger.info(f"Embedding length: {len(embedding)}")
    
    GOOGLE_API_KEY = read_config('AI KEYS', 'gemini')
    if not GOOGLE_API_KEY:
        logger.error("Gemini keys not found in environment variables")
        raise ValueError("Gemini keys not found in environment variables")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    
    my_embed_model2 = GoogleGenAIEmbedding(
        model="models/text-embedding-004",
        output_dimensionality=GEMINI_EMBED_DIMENSION,
    )
    Settings.embed_model = my_embed_model2
    logger.info("Google GenAI embedding model initialized successfully")
    # Test the embedding model with a sample text
    sample_text2 = "This is a test sentence for embedding."
    embedding2 = my_embed_model2.get_text_embedding(sample_text2)
    logger.info(f"Embedding length: {len(embedding2)}")



                
def main(): 
    return None 

if __name__ == "__main__":
    main()
    #test_RAG("vasilias_weddings_2025_05_24", "gemini")
    today_date = datetime.datetime.now().strftime("%Y_%m_%d")
    my_table = "vasilias_weddings_" + today_date
    my_documents1 = index_data("./data/VW_dataMar25.txt", "file", "local", "quentinz/bge-base-zh-v1.5:latest") 
    store_documents(my_table, my_documents1, "local")
    my_documents2 = index_data("./data/VWFAQ24052025.csv", "csv", "local", "quentinz/bge-base-zh-v1.5:latest") 
    store_documents(my_table, my_documents2, "local")
    
    
    # vasilias_nikoklis_doc = index_data("https://vasilias.nikoklis.com/", "gemini")
    # cyprus_wedding_doc = index_data("https://vasilias.nikoklis.com/cyprus-wedding-venue/", "gemini")
    # my_documents = vasilias_nikoklis_doc + cyprus_wedding_doc
    # my_documents1 = index_data("https://vasilias.nikoklis.com/vasilias-nikoklis-history/", "gemini")
    # store_documents("vasilias_weddings3", my_documents1)
    # my_documents2 = index_data("https://vasilias.nikoklis.com/contact-us/", "gemini")
    # store_documents("vasilias_weddings3", my_documents2)
    # qa_docs = parse_qa_csv("VWFAQ.csv")
    # store_documents("vasilias_faq", qa_docs)
    # Manually create a document
    # manual_document = [
    #     {
    #         "text": "Vasilias Weddings is a premier wedding venue in Cyprus, offering a picturesque setting for your special day. We provide comprehensive wedding services, including catering, decoration, and guest accommodation.",
    #         "metadata": {"source": "manual_entry", "category": "venue_info"}
    #     }
    # ]

    # Store the manually created document
    #store_documents(table, manual_document)
    #my_documents = index_data("https://vasilias.nikoklis.com", "gemini")
   #store_documents(table, my_documents)
    # Test the RAG service
    
     
        