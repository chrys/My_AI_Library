import os
import threading
from sqlalchemy import make_url
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
      PromptTemplate,
      StorageContext,
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

from llama_index.readers.web import SimpleWebPageReader

from my_library.parse_csv import parse_qa_csv


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
            # self.embed_model = GeminiEmbedding(
            #     model="models/text-embedding-004",
            #     dimensions=EMBED_DIMENSION
            # )
            self.embed_model = GoogleGenAIEmbedding(
                model="models/text-embedding-004",
                dimensions=EMBED_DIMENSION,
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

            self.retriever = self.index.as_retriever(similarity_top_k=3) #let's start with 3 top chunks
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
            self.llm = GoogleGenAI(model="models/gemini-2.0-flash-lite")

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
                #template = "Based on the following context, please answer the question: {message}\n\nContext: {context}"
                #google gemini template shouldn't exceed 4000 characters
                template = """
                    You are a helpful and friendly chatbot named VasiliasBot assisting customers of Vasilias Weddings.
                    You are a dedicated customer service representative whose primary goal is to provide information and assistance related to Vasilias Weddings based *exclusively* on the knowledge provided in the context below. You must stick to the context. Do not hallucinate answers.

                    **Instructions:**

                    1.  **Knowledge Source:** You *must* answer questions *only* based on the information provided in the **Context** section below. Do not use any external knowledge or prior assumptions.
                    2.  **Handling Missing Information:** If the answer to the user's query cannot be confidently determined from the provided **Context**, politely state that you don't have that specific information based on the available website content. Do not invent an answer. Suggest contacting Vasilias Weddings directly for specifics if appropriate (e.g., "I couldn't find the exact details about that in the information I have access to. For the most accurate information, please reach out to Vasilias Weddings directly.").
                    3.  **Tone and Style:** Respond in a direct, helpful, and natural conversational style. Aim for a professional yet warm and approachable tone. *Do not use overly formal language or overly verbose explanations.* Be concise and to the point. *Do not include conversational fillers or greetings like "Hi" or "Hello" at the start of your response*.
                    4.  **Focus and Relevance:** Respect the user's time by focusing on providing the most relevant information first. Only respond to the specific question asked in the **User Query**. Do not provide unsolicited extra information.
                    5.  **Empathy and Support:** Acknowledge that wedding planning can be complex and potentially stressful. Offer support and understanding in your responses. Strive for a balance between showing support and providing clear, concise answers efficiently.
                    6.  **Accuracy and Corrections:** Prioritize accuracy based on the **Context**. If the user states something incorrect according to the **Context**, gently correct them using an empathizing and supportive tone.

                    **Context:**
                    {context_str}

                    **User Query:** {query_str}

                    **Response:**
                    """
                self.local.prompt = PromptTemplate(template)

            # Retrieve documents
            retrieved_nodes = self.retriever.retrieve(message)

            if not retrieved_nodes:
                return "I couldn't find any relevant information to answer your question."

            # Format context from retrieved nodes
            context = "\n".join([node.node.text for node in retrieved_nodes])

            # Format prompt with context
            formatted_prompt = self.local.prompt.format(
                context=context,
                query_str=message)
            
            # Get response using LLM
            response = self.llm.complete(formatted_prompt)
            # Return the response
            if not response or not response.text:
                return "I couldn't generate a response based on the provided context."
            else: 
                self.logger.info("Response generated successfully")
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
        my_llm = GoogleGenAI(model = "models/gemini-2.0-flash-lite")
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
        my_embed_model = GoogleGenAIEmbedding(
            model="models/text-embedding-004",
            dimensions=768
        )
    else:
        raise ValueError("Invalid model name. Use 'gemini', 'claude', or 'openai'")

    splitter = SemanticSplitterNodeParser(
        buffer_size=1, 
        breakpoint_percentile_threshold=95, 
        embed_model=my_embed_model,
    )
    Settings.node_parser = splitter
    Settings.embed_model = my_embed_model

     # Check if input is URL
    if sample_data.startswith(('http://', 'https://')):
        None
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
    EMBED_DIMENSION = 768
    
    api_key = read_config('AI KEYS', 'gemini')
    if not api_key:
        raise ValueError("Gemini keys not found in environment variables")
    os.environ["GOOGLE_API_KEY"] = api_key
    llm = GoogleGenAI(model="models/gemini-2.0-flash-lite")
    
    my_embed_model = GoogleGenAIEmbedding(
                model="models/text-embedding-004",
                dimensions=EMBED_DIMENSION,
            )
            
    Settings.embed_model = my_embed_model
    Settings.llm = llm
            
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

def test_RAG(table):
    # Get RAG service for table and model "gemini"
    rag_service = RAGServiceManager.get_instance(table, "gemini")
    
    # List of questions to ask
    questions = [
        "Can you recommend wedding suppliers?",  
        "What is the average cost of a wedding package?", 
        #"Can you assist with accommodation for our guests?", OK 
        #"what is the capital of France?", OK
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
    table = "vasilias_weddings290325"
    #my_documents = index_data("VW_dataMar25.txt", "gemini")
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
    test_RAG(table)
     
        