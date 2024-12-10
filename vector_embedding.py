import logging
from chunking import process_document
import ollama
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Setup logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Custom embedding function using Ollama's API
class OllamaEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, model_name='nomic-embed-text'):
        """Initialize the embedding function."""
        self.model_name = model_name
        logging.info(f"Initialized OllamaEmbeddingFunction with model: {self.model_name}")

    def __call__(self, input: Documents) -> Embeddings:
        """Embed the input documents."""
        return [ollama.embeddings(model=self.model_name, prompt=text)['embedding'] for text in input]

def setup_chromadb(path: str, collection_name: str):
    """Setup ChromaDB client and collection."""
    logging.info(f"Setting up ChromaDB at path: {path} with collection name: {collection_name}")
    client = chromadb.PersistentClient(path=path)
    embedding_function = OllamaEmbeddingFunction()
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedding_function)
    logging.info(f"ChromaDB collection '{collection_name}' setup successfully")
    return collection

def ingest_data_to_chromadb(collection, documents, document_ids):
    """Ingest data into ChromaDB."""
    logging.info(f"Ingesting {len(documents)} documents into ChromaDB collection")
    collection.upsert(ids=document_ids, documents=documents)
    logging.info(f"Successfully ingested {len(documents)} documents into the collection")

def main():
    file_path = "docu_3.txt"
    chromadb_path = "E:/insurance/coding_project/embeddings"
    collection_name = "test_3"
    
    logging.info(f"Starting the document processing for file: {file_path}")
    # Use the process_document function to generate text chunks
    buffer_size = 2
    breakpoint_percentile_threshold = 50
    chunks = process_document(file_path, buffer_size, breakpoint_percentile_threshold)
    logging.warning(f"Generated {len(chunks)} chunks from the document")
    
    # Create document IDs for the chunks
    document_ids = [f"id_{i}" for i in range(len(chunks))]
    logging.warning(f"Created {len(document_ids)} document IDs for the chunks")
    
    # Setup ChromaDB
    collection = setup_chromadb(chromadb_path, collection_name)
    
    # Ingest the chunks into ChromaDB
    ingest_data_to_chromadb(collection, chunks, document_ids)
    logging.warning("Completed ingestion of document chunks into ChromaDB")

if __name__ == "__main__":
    main()
