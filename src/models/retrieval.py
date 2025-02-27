# retrieval.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def retrieve_similar_images(query_embedding, database_embeddings, top_k=5):
    """
    Retrieve indices of the top_k similar images given a query embedding.
    
    Args:
        query_embedding (numpy.ndarray): 1D array of shape (embed_dim,) representing the query image embedding.
        database_embeddings (numpy.ndarray): 2D array of shape (num_images, embed_dim) containing embeddings for all images.
        top_k (int): Number of top similar images to return.
    
    Returns:
        top_indices (list): List of indices corresponding to the most similar images.
        top_similarities (list): List of cosine similarity scores for these images.
    """
    # Reshape query embedding to a 2D array for similarity computation
    query_embedding = query_embedding.reshape(1, -1)
    
    # Compute cosine similarity between the query embedding and each image in the database
    similarities = cosine_similarity(query_embedding, database_embeddings)[0]
    
    # Get indices of the top_k most similar images (sorted in descending order)
    top_indices = similarities.argsort()[-top_k:][::-1]
    top_similarities = similarities[top_indices]
    
    return top_indices.tolist(), top_similarities.tolist()

if __name__ == "__main__":
    # Dummy example for testing
    embed_dim = 768
    num_images = 100
    # Create a random database of embeddings (in practice, these would be precomputed)
    database_embeddings = np.random.rand(num_images, embed_dim)
    
    # Create a random query embedding
    query_embedding = np.random.rand(embed_dim)
    
    indices, sims = retrieve_similar_images(query_embedding, database_embeddings, top_k=5)
    print("Top similar image indices:", indices)
    print("Similarity scores:", sims)
