import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ollama

# Ollama Embedding Function
class OllamaEmbeddingFunction:
    def __init__(self, model_name='nomic-embed-text'):
        """Initialize the embedding function with a model name."""
        self.model_name = model_name

    def embed_documents(self, documents):
        """Embed the input documents using Ollama."""
        return [ollama.embeddings(model=self.model_name, prompt=text)['embedding'] for text in documents]

# Function to load essay text
def load_essay(file_path):
    with open(file_path) as file:
        essay = file.read()
    return essay

# Function to split essay into sentences
def split_into_sentences(essay):
    single_sentences_list = re.split(r'(?<=[.?!])\s+', essay)
    print(f"{len(single_sentences_list)} sentences were found")
    return [{'sentence': x, 'index': i} for i, x in enumerate(single_sentences_list)]

# Function to combine sentences with a buffer size
def combine_sentences(sentences, buffer_size=1):
    for i in range(len(sentences)):
        combined_sentence = ''

        # Add sentences before the current one
        for j in range(i - buffer_size, i):
            if j >= 0:
                combined_sentence += sentences[j]['sentence'] + ' '

        # Add the current sentence
        combined_sentence += sentences[i]['sentence']

        # Add sentences after the current one
        for j in range(i + 1, i + 1 + buffer_size):
            if j < len(sentences):
                combined_sentence += ' ' + sentences[j]['sentence']

        sentences[i]['combined_sentence'] = combined_sentence

    return sentences

# Function to generate embeddings for combined sentences using Ollama
def generate_embeddings_with_ollama(sentences):
    """Generate embeddings for combined sentences using the Ollama embedding model."""
    ollama_embeds = OllamaEmbeddingFunction()
    embeddings = ollama_embeds.embed_documents([sentence['combined_sentence'] for sentence in sentences])

    for sentence, embedding in zip(sentences, embeddings):
        sentence['combined_sentence_embedding'] = embedding

    return sentences

# Function to calculate cosine distances between sentence embeddings
def calculate_cosine_distances(sentences):
    distances = []
    for i in range(len(sentences) - 1):
        similarity = cosine_similarity([sentences[i]['combined_sentence_embedding']], [sentences[i + 1]['combined_sentence_embedding']])[0][0]
        distance = 1 - similarity
        distances.append(distance)
        sentences[i]['distance_to_next'] = distance
    return distances, sentences

# Function to determine breakpoints and count distances above threshold
def determine_breakpoints(distances, breakpoint_percentile_threshold=95):
    breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)
    num_distances_above_threshold = len([x for x in distances if x > breakpoint_distance_threshold])
    indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]
    return breakpoint_distance_threshold, num_distances_above_threshold, indices_above_thresh

# Function to create chunks of sentences
def create_chunks(sentences, indices_above_thresh):
    start_index = 0
    chunks = []
    for index in indices_above_thresh:
        end_index = index
        group = sentences[start_index:end_index + 1]
        combined_text = ' '.join([d['sentence'] for d in group])
        chunks.append(combined_text)
        start_index = index + 1

    # Add the last group if any sentences remain
    if start_index < len(sentences):
        combined_text = ' '.join([d['sentence'] for d in sentences[start_index:]])
        chunks.append(combined_text)

    return chunks

# Main function to orchestrate the flow of the code
def process_document(file_path, buffer_size=1, breakpoint_percentile_threshold=95):
    essay = load_essay(file_path)
    sentences = split_into_sentences(essay)
    sentences = combine_sentences(sentences, buffer_size=buffer_size)
    sentences = generate_embeddings_with_ollama(sentences)
    distances, sentences = calculate_cosine_distances(sentences)
    breakpoint_distance_threshold, num_distances_above_threshold, indices_above_thresh = determine_breakpoints(distances, breakpoint_percentile_threshold)
    chunks = create_chunks(sentences, indices_above_thresh)
    return chunks
