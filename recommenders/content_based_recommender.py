# content_based.py
from recommenders.recommender_base import Recommender
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class ContentBasedRecommender(Recommender):
    def __init__(self, movies, ratings):
        """
        Inicializa el recomendador basado en contenido con los datos de películas y calificaciones.

        Parámetros:
        movies (DataFrame): Datos de las películas, incluyendo la información de metadatos.
        ratings (DataFrame): Datos de las calificaciones de los usuarios para las películas.
        """
        # Inicializa la clase base con los datos de películas y calificaciones
        super().__init__(movies, ratings)

        # Calcula la matriz TF-IDF basada en la columna 'metadata' de las películas
        # Utiliza palabras en inglés como stopwords
        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform(movies["metadata"])

        # Calcula la similitud del coseno entre todas las películas
        self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    def recommend(self, user_id, top_n=10):
        """
        Recomienda películas a un usuario basándose en la similitud de contenido con las películas que ya ha valorado.

        Parámetros:
        user_id (int): ID del usuario para el que se harán las recomendaciones.
        top_n (int): Número de películas a recomendar.

        Devuelve:
        numpy.ndarray: IDs de las películas recomendadas.
        """
        # Obtener las películas valoradas por el usuario
        user_ratings = self.ratings[self.ratings["userId"] == user_id]
        if user_ratings.empty:
            return []

        # Obtener los índices de las películas valoradas por el usuario
        rated_movie_indices = self.movies.index[
            self.movies["movieId"].isin(user_ratings["movieId"])
        ].tolist()

        # Calcular las similitudes ponderadas por la calificación del usuario
        # Inicializa un vector de puntajes de similitud ponderados con ceros
        weighted_sim_scores = np.zeros(self.cosine_sim.shape[0])
        for row in user_ratings.itertuples():
            movie_id = row.movieId
            rating = row.rating
            # Encuentra el índice de la película en el DataFrame de películas
            idx = self.movies.index[self.movies["movieId"] == movie_id].tolist()
            if idx:
                # Suma la similitud ponderada por la calificación del usuario
                weighted_sim_scores += self.cosine_sim[idx[0]] * rating

        # Excluir películas ya valoradas por el usuario, estableciendo sus puntajes a 0
        for idx in rated_movie_indices:
            weighted_sim_scores[idx] = 0

        # Ordenar por puntaje de similitud ponderado en orden descendente
        sim_scores = list(enumerate(weighted_sim_scores))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Obtener los índices de las películas recomendadas
        sim_scores = sim_scores[:top_n]
        movie_indices = [i[0] for i in sim_scores]
        return self.movies.iloc[movie_indices]["movieId"].values

    def update_data(self, ratings):
        """
        Actualiza los datos de calificaciones.

        Parámetros:
        ratings (DataFrame): Datos de las calificaciones de los usuarios para las películas.
        """
        # Actualiza las calificaciones almacenadas con los nuevos datos
        self.ratings = ratings
