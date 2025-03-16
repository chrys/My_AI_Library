import os
import threading
from sqlalchemy import make_url
from my_library.utilities import read_config
from my_library.logger import setup_logger

from llama_index.llms.openai import OpenAI
from llama_index.core import (
      VectorStoreIndex,
      Settings,
      SimpleDirectoryReader,
      PromptTemplate,
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