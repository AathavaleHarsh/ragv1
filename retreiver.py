import ollama
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

# Custom embedding function using Ollama's API
class OllamaEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, model_name='nomic-embed-text'):
        """Initialize the embedding function."""
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        """Embed the input documents."""
        return [ollama.embeddings(model=self.model_name, prompt=text)['embedding'] for text in input]

def setup_chromadb(path: str, collection_name: str):
    """Setup ChromaDB client and collection."""
    client = chromadb.PersistentClient(path=path)
    embedding_function = OllamaEmbeddingFunction()
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedding_function)
    return collection

def embed_query(query: str, model_name='nomic-embed-text'):
    """Embed the query using Ollama embedding function."""
    embedding_function = OllamaEmbeddingFunction(model_name)
    return embedding_function([query])[0]

def find_most_similar_chunk(query_embedding, collection, n_results=1):
    """Find the most similar chunk in the ChromaDB collection to the given query embedding."""
    res = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )
    return res['documents'][0]
