import ollama
import logging
from retreiver import setup_chromadb, embed_query, find_most_similar_chunk

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AIModelInvoker:
    def __init__(self, query, context):
        """
        Initialize with the query and context.
        """
        self.query = query
        self.context = context
        logging.info("AIModelInvoker initialized with query and context.")

    def generate_system_prompt(self):
        """
        Generates the static system prompt.
        """
        logging.info("Generating system prompt.")
        return "You are an AI assistant designed as part of a Retrieval-Augmented Generation (RAG) system. only use the context provided below to answer the query"

    def generate_query_prompt(self):
        """
        Generates the prompt using the query and context.
        """
        logging.info("Generating query prompt.")
        return (
            f"Query: {self.query}\n"
            f"Context: {self.context}\n"
        )

    def invoke_model(self):
        """
        Invokes the Ollama model with the provided query and context and returns the response.
        """
        logging.info("Invoking the Ollama model.")
        sys_prompt = self.generate_system_prompt()
        query_prompt = self.generate_query_prompt()
        
        response = ollama.chat(
            model='llama3.2:latest',
            messages=[
                {'role': 'system', 'content': sys_prompt},
                {'role': 'user', 'content': query_prompt}
            ],
            stream=True,
        )
        complete_response = ""
        for chunk in response:
            complete_response += chunk['message']['content']
        logging.info("Model invocation completed.")
        return complete_response
    
    

