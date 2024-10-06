# content_based.py

from sklearn.metrics.pairwise import linear_kernel
import numpy as np


class ContentBasedRecommender:
    def __init__(self, tfidf_matrix, movies, ratings):
        self.tfidf_matrix = tfidf_matrix
        self.movies = movies.reset_index()
        self.ratings = ratings
        self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    def recommend(self, user_id, top_n=10):
        # Obtener las películas valoradas por el usuario
        user_ratings = self.ratings[self.ratings["userId"] == user_id]
        if user_ratings.empty:
            return []

        # Obtener los índices de las películas valoradas por el usuario
        rated_movie_indices = self.movies.index[
            self.movies["movieId"].isin(user_ratings["movieId"])
        ].tolist()

        # Calcular las similitudes ponderadas por la calificación del usuario
        weighted_sim_scores = np.zeros(self.cosine_sim.shape[0])
        for row in user_ratings.itertuples():
            movie_id = row.movieId
            rating = row.rating
            idx = self.movies.index[self.movies["movieId"] == movie_id].tolist()
            if idx:
                weighted_sim_scores += self.cosine_sim[idx[0]] * rating

        # Excluir películas ya valoradas por el usuario
        for idx in rated_movie_indices:
            weighted_sim_scores[idx] = 0

        # Ordenar por puntaje de similitud ponderado
        sim_scores = list(enumerate(weighted_sim_scores))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Obtener los índices de las películas recomendadas
        sim_scores = sim_scores[:top_n]
        movie_indices = [i[0] for i in sim_scores]
        return self.movies.iloc[movie_indices]["movieId"].values

    def update_data(self, ratings):
        self.ratings = ratings
