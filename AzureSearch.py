import os
import openai
import time
from azure.search.documents import SearchClient
from azure.search.documents.models import Vector
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

# Class to handle Azure Search operations
class Config:
    def __init__(self, path='./environment.env'):
        # Load environment variables from specified file path
        load_dotenv(dotenv_path=path)

        # Azure Search Service configurations
        self.endpoint = os.getenv("COG_ENDPOINT")
        self.vector_config_name = os.getenv("VECTOR_CONFIG_NAME")
        self.vector_field = os.getenv("VECTOR_FIELD_NAME")
        self.index_name = os.getenv("INDEX_NAME")
        self.top_k = os.getenv("TOP_K")

        # OpenAI API configurations
        self.open_ai_endpoint = os.getenv("OPENAI_ENDPOINT")
        self.engine = os.getenv("EMBEDDING")
        self.gpt = os.getenv("GPT")

        # Additional OpenAI configurations
        self.openai_api_base = os.getenv("OPENAI_API_BASE")
        self.openai_api_version = os.getenv("OPENAI_API_VERSION")

class AzureSearch:
    def __init__(self, config, index_name):
        # Initialize with configurations
        self.endpoint = config.endpoint
        self.open_ai_endpoint = config.open_ai_endpoint
        self.engine = config.engine
        self.index_name = index_name
        self.vector_field = config.vector_field

        # Use Azure Default Credential for Azure Search Client
        azure_cred = DefaultAzureCredential()
        self.search_client = SearchClient(endpoint=self.endpoint, index_name=self.index_name, credential=azure_cred)

        # Use Azure Default Credential for OpenAI
        token_credential = azure_cred.get_token("https://cognitiveservices.azure.com/")
        openai.api_key = token_credential.token
        openai.api_version = config.openai_api_version
        openai.api_type = "azure"
        openai.api_base = config.openai_api_base
    
    # Method to generate embeddings for a given text using OpenAI
    def generate_embeddings(self, text):
    
        # Trim the text if it exceeds the token limit
        words = text.split()
        if len(words) > 6000:
            words = words[:6000]
            trimmed_text = ' '.join(words)
        else:
            trimmed_text = text

        retries = 0
        wait_time = 30  # Starting wait time in seconds
        max_retries = 5

        while retries < max_retries:
            try:
                response = openai.Embedding.create(
                    input=trimmed_text,
                    engine=self.engine
                )
                return response['data'][0]['embedding']
            except Exception as e:  # Catch the specific exception for rate limits if possible
                retries += 1
                if retries < max_retries:
                    print(f"Rate limit exceeded, waiting for {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    # Optionally increase wait_time if needed, or keep it constant
                else:
                    print("Maximum retries reached. Returning error.")
                    return {'error': str(e)}
    
    # Method to perform a vector search on the Azure Cognitive Search index
    def vector_search(self, query, filter, k=5, select_fields=None, vector_search=True): #### ADD SEMANTIC RANKER ####################
        #######################################################
        ####################
        ###############
        ####################    
        # Initialize parameters
        vectors_param = None
        order_by_param = "search.score() desc"

        # Check if vector search is enabled
        if vector_search:
            vector = Vector(value=self.generate_embeddings(query), k=k, fields=self.vector_field)
            vectors_param = [vector]
            #query = None # Pure Vector Search
            order_by_param = None  # Removing the order_by clause if vector_search is true

        # Set default fields if none are provided
        if select_fields is None:
            select_fields = "*"

        # Parse results    
        results = self.search_client.search(
            search_text=query,
            top=k,
            vectors=vectors_param,
            select=select_fields,
            order_by=order_by_param,
            filter = filter if filter else None
        )

        # Convert each result to a dictionary and append to search_output
        search_output = [dict(result) for result in results]

        return search_output
