from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

import random

def generate_random_genre(n):
    """
    Generate a dataset of random movie genres.

    Args:
        n (int): The number of movie IDs to generate data for.

    Returns:
        dict: A dictionary containing movie IDs as keys and random genre strings as values.
    """
    genres_list = [
        "Action", "Adventure", "Animation", "Children's", "Comedy", "Drama", "Fantasy",
        "Romance", "Sci-Fi", "Thriller", "Horror", "Mystery", "Documentary", "Musical", "War",
        "Western"
    ]

    # Create a larger dataset
    larger_dataset = {}
    for i in range(1, n):  # Generate data for n movie IDs
        num_genres = random.randint(5, 10)  # Choose a random number of genres for each movie
        random_genres = random.sample(genres_list, num_genres)
        genre_string = '|'.join(random_genres)
        larger_dataset[str(i)] = genre_string

    return larger_dataset

def cosine_similarity_matrix(data):
    """
    Calculate the cosine similarity matrix for a list of documents.

    Args:
        data (list): A list of strings representing the documents.

    Returns:
        numpy.ndarray: The cosine similarity matrix.
    """
    # Create a TF-IDF vectorizer and transform the data into TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data)

    # Calculate the cosine similarity matrix
    cosine_sim_matrix = cosine_similarity(
        tfidf_matrix,
        tfidf_matrix
    )

    return cosine_sim_matrix

if __name__=='__main__':
    # Call the function with the sample data
    similarity_matrix = cosine_similarity_matrix(generate_random_genre(10000).values())
    movie_scores = list(enumerate(similarity_matrix[0]))
    movie_scores = [ i for i in movie_scores if i[1] != 1]
    sorted_score = sorted(movie_scores, key=lambda x: x[1], reverse=True)
    print([i[0] for i in sorted_score[0:11] ])



