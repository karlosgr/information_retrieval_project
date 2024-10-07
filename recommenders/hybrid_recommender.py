# hybrid_recommender.py
from recommenders.recommender_base import Recommender
import numpy as np


class HybridRecommender(Recommender):
    def __init__(self, recommenders, movies, ratings, weights=None):
        """
        Inicializa el recomendador híbrido con una lista de recomendadores, datos de películas y calificaciones.

        Parámetros:
        recommenders (list): Lista de clases de recomendadores que se utilizarán en el recomendador híbrido.
        movies (DataFrame): Datos de las películas.
        ratings (DataFrame): Datos de las calificaciones de los usuarios para las películas.
        weights (list, opcional): Lista de pesos para cada recomendador. Si no se proporcionan, se asignan pesos iguales.
        """
        # Inicializa la clase base con los datos de películas y calificaciones
        super().__init__(movies, ratings)

        # Inicializa los recomendadores con los datos de películas y calificaciones
        self.recommenders = [
            recommender(movies, ratings) for recommender in recommenders
        ]

        # Si no se proporcionan pesos, se asignan pesos iguales a todos los recomendadores
        if weights is None:
            self.weights = [1 / len(recommenders)] * len(recommenders)
        else:
            self.weights = weights

    def recommend(self, user_id, top_n=10):
        """
        Recomienda películas a un usuario combinando las recomendaciones de los recomendadores base.

        Parámetros:
        user_id (int): ID del usuario para el que se harán las recomendaciones.
        top_n (int): Número de películas a recomendar.

        Devuelve:
        list: IDs de las películas recomendadas.
        """
        # Si el usuario es nuevo, recomendar películas basadas en popularidad
        if user_id not in self.recommenders[1].user_item_matrix.index:
            return self.recommend_popular(top_n)

        # Lista para almacenar las recomendaciones de cada recomendador
        scores_list = []
        for recommender in self.recommenders:
            # Obtiene más recomendaciones de las necesarias para combinar mejor
            scores = recommender.recommend(user_id, top_n * 2)
            scores_list.append(scores)

        # Unir todas las recomendaciones de los recomendadores base y calcular puntuaciones
        all_rec = np.concatenate(scores_list)

        # Obtener las películas únicas recomendadas y contar cuántas veces se recomienda cada una
        unique_rec, counts = np.unique(all_rec, return_counts=True)
        rec_scores = dict(zip(unique_rec, counts))

        # Ordenar las recomendaciones por puntuación en orden descendente
        sorted_rec = sorted(rec_scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = [rec[0] for rec in sorted_rec[:top_n]]
        return recommendations

    def recommend_popular(self, top_n=10):
        """
        Recomienda las películas más populares en base a la cantidad de calificaciones.

        Parámetros:
        top_n (int): Número de películas a recomendar.

        Devuelve:
        list: IDs de las películas más populares.
        """
        # Calcular la popularidad de cada película según la cantidad de calificaciones
        movie_popularity = (
            self.recommenders[1]
            .ratings["movieId"]
            .value_counts()
            .sort_values(ascending=False)
        )

        # Seleccionar las top_n películas más populares
        popular_movies = movie_popularity.head(top_n).index.tolist()
        return popular_movies

    def update_data(self, ratings):
        """
        Actualiza los datos de calificaciones en todos los recomendadores base.

        Parámetros:
        ratings (DataFrame): Datos de las calificaciones de los usuarios para las películas.
        """
        # Actualizar los datos en cada uno de los recomendadores base
        for recommender in self.recommenders:
            recommender.update_data(ratings)
