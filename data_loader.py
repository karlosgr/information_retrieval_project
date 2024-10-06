import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class DataLoader:
    def __init__(self, ratings_file, movies_file, tags_file):
        self.ratings_file = ratings_file
        self.movies_file = movies_file
        self.tags_file = tags_file

    def load_data(self):
        ratings = pd.read_csv(self.ratings_file)
        movies = pd.read_csv(self.movies_file)
        tags = pd.read_csv(self.tags_file)

        # Combinar datos de películas y tags
        movies = pd.merge(movies, tags, on='movieId', how='left')
        movies['tag'] = movies['tag'].fillna('')
        movies['metadata'] = movies['genres'] + ' ' + movies['tag']

        return ratings, movies

    def preprocess_movies(self, movies):
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(movies['metadata'])
        return tfidf_matrix
