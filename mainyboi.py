# In your_rag_chatbot_module.py

import logging
from retreiver import setup_chromadb, embed_query, find_most_similar_chunk
from rag import AIModelInvoker  # Assuming AIModelInvoker is in a separate file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load ChromaDB once during initialization
chromadb_path = "E:/insurance/coding_project/embeddings"
collection_name = "test_3"
collection = setup_chromadb(chromadb_path, collection_name)

def get_response(user_message):
    """
    Retrieves relevant context and generates a response for the user's message.
    """
    # Step 1: Embed the Query
    logging.info("Embedding the query.")
    query_embedding = embed_query(user_message)

    # Step 2: Find the Most Similar Chunk
    logging.info("Finding the most similar chunk from the collection.")
    most_similar_chunk = find_most_similar_chunk(query_embedding, collection)

    # Step 3: Use the retrieved context to invoke the model
    logging.info("Using the retrieved context to invoke the model.")
    context = most_similar_chunk
    logging.info(f"Retrieved chunk: {context}")

    # Step 4: Create an instance of AIModelInvoker and generate a response
    ai_invoker = AIModelInvoker(user_message, context)
    response = ai_invoker.invoke_model()

    return response
