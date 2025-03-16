import os
import threading
from sqlalchemy import make_url
from my_library.utilities import read_config
from logger import setup_logger
# Initialize the logger
logger = setup_logger()

from llama_index.llms.openai import OpenAI
from llama_index.core import (
      VectorStoreIndex,
      Settings,
      SimpleDirectoryReader,
      PromptTemplate,
      StorageContext,
  )

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.gemini import Gemini

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
        self.initialized = False

        # Thread-local storage for request context
        self.local = threading.local()

    def initialize(self):
        """Initialize RAG components"""
        if self.initialized:
            return True
        self.logger.info(f"Initializing RAG components for {self.model_name}...")
        EMBED_DIMENSION = 768
        try:
            # Get connection string
            my_connection_string = read_config('CONNECTIONS', 'postgres')
            if not my_connection_string:
                self.logger.error("PostgreSQL connection string not found")
                return False

            # Initialize LLM based on model selection
            self._initialize_llm()

            # Initialize embeddings
            self.embed_model = GeminiEmbedding(
                model="models/text-embedding-004",
                dimensions=EMBED_DIMENSION
            )
            Settings.embed_model = self.embed_model

            # Initialize vector store
            url = make_url(my_connection_string)
            self.vector_store = PGVectorStore.from_params(
                database=url.database,
                host=url.host,
                password=url.password,
                port=url.port,
                user=url.username,
                table_name=self.table_name,
                embed_dim=EMBED_DIMENSION,
            )

            # Create index and retriever
            self.index = VectorStoreIndex.from_vector_store(self.vector_store)
            if not self.index:
                self.logger.error("Failed to create index from vector store")
                return False

            self.retriever = self.index.as_retriever()
            if not self.retriever:
                self.logger.error("Failed to create retriever")
                return False

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
            self.llm = Gemini(model="models/gemini-2.0-flash-lite")

        else:
            raise ValueError("Invalid model name. Use 'openai', 'claude', or 'gemini'")

        Settings.llm = self.llm
        self.logger.info(f"LLM ({self.model_name}) initialized successfully")

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
            if not hasattr(self.local, 'prompt'):
                template = "Based on the following context, please answer the question: {message}\n\nContext: {context}"
                self.local.prompt = PromptTemplate(template)

            # Retrieve documents
            retrieved_nodes = self.retriever.retrieve(message)

            if not retrieved_nodes:
                return "I couldn't find any relevant information to answer your question."

            # Format context from retrieved nodes
            context = "\n".join([node.node.text for node in retrieved_nodes])

            # Format prompt with context
            formatted_prompt = self.local.prompt.format(message=message, context=context)

            # Get response using LLM
            response = self.llm.complete(formatted_prompt)
            return response.text

        except Exception as e:
            self.logger.error(f"Error processing question: {str(e)}")
            return "I'm sorry, but an error occurred while processing your question."

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

def index_data(sample_data, llm_model):
    """
    Index data from various sources using specified LLM model.
    
    Args:
        sample_data (str): Path to file/directory or URL
        llm_model (str): Name of the LLM model to use ("gemini", "claude", "openai")
    
    Returns:
        list: List of processed documents
    
    Raises:
        ValueError: If invalid model name is provided
        FileNotFoundError: If the specified file doesn't exist
    """
   # Initialize LLM and Embedding models
    my_llm = None
    my_embed_model = None
    
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
        my_llm = Gemini(model = "models/gemini-2.0-flash-lite")
        Settings.llm = my_llm
        logger.info("Gemini LLM initialized successfully")

    # Initialize Embedding model
    if llm_model.lower() == "openai":
        None
        #TODO code for OpenAI embedding model 
    elif llm_model.lower() == "claude":
        None
        #TODO code for Claude embedding model
    elif llm_model.lower() == "gemini":
        my_embed_model = GeminiEmbedding(
            model="models/text-embedding-004",
            dimensions=768
        )
    else:
        raise ValueError("Invalid model name. Use 'gemini', 'claude', or 'openai'")

    Settings.embed_model = my_embed_model

     # Check if input is URL
    if sample_data.startswith(('http://', 'https://')):
        None
        try:
            # TODO code for scraping website
            # url = sample_data,
            # docs = scrape_website(
            #     url=url,
            #     max_pages=100,
            #     output_file="output/vasilias_content.json"
            # )

            #if docs:
            #    print(f"\nSuccessfully scraped {len(docs)} documents")
            #else:
            #    print("\nNo documents were scraped")
            pass
        except Exception as e:
            import traceback
            logger(f"\nError occurred: {str(e)}") 
    # Process local file or directory
    elif os.path.exists(sample_data):
        if os.path.isfile(sample_data):
            reader = SimpleDirectoryReader(input_files=[sample_data])
            documents = reader.load_data()
            return documents
        elif os.path.isdir(sample_data):
            reader = SimpleDirectoryReader(input_dir=sample_data)
            documents = reader.load_data()
            return documents  
    else:
        raise ValueError("Invalid input. Please provide a url or a valid file or a valid directory path")

    return documents

def store_documents(my_table_name, my_documents):
    """
    Store documents in the specified table.
    
    Args:
        my_table_name (str): Name of the table to store documents
        my_documents (list): List of documents to store
    
    Returns:
        bool: True if documents are stored successfully, False otherwise
    """
    try:
        # Get connection string
        my_connection_string = read_config('CONNECTIONS', 'postgres')
        if not my_connection_string:
            logger.error("PostgreSQL connection string not found")
            raise ValueError("PostgreSQL connection string not found")
        # Initialize vector store
        url = make_url(my_connection_string)
        my_vector_store = PGVectorStore.from_params(
            database=url.database,
            host=url.host,
            password=url.password,
            port=url.port,
            user=url.username,
            table_name=my_table_name,
            embed_dim=768,
        )
        # Store documents
        my_storage_context = StorageContext.from_defaults(
            vector_store=my_vector_store,
        ) 
        my_index = VectorStoreIndex.from_documents(
            documents=my_documents,
            storage_context=my_storage_context, 
            show_progress=True,
        )
        if not my_index:
            logger.error("Failed to create index from documents")
            raise ValueError("Failed to create index from documents")
        else:
            return my_index        
    except Exception as e:
        logger.error(f"Error storing documents: {str(e)}")
        return False

def test_RAG():
    # Get RAG service for "vasilias_weddings2" and model "gemini"
    rag_service = RAGServiceManager.get_instance("vasilias_weddings2", "gemini")
    
    # List of questions to ask
    questions = [
        "Where is Vasilias Weddings?",
        "When can I get married?",
        "What is the cost of a wedding?",
        "Do you offer catering services?",
    ]
    
    # Ask each question and print the reply
    for question in questions:
        response = rag_service.ask(question)
        print(f"Question: {question}")
        print(f"Reply: {response}\n")
            
def main(): 
    return None 

if __name__ == "__main__":
    main()
    
    # my_documents = index_data("VW_dataMar25.txt", "gemini")
    # store_documents("vasilias_weddings2", my_documents)
    
    test_RAG()
     
        