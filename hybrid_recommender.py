# hybrid_recommender.py

import numpy as np


class HybridRecommender:
    def __init__(
        self, content_recommender, collaborative_recommender, sequential_recommender
    ):
        """
        Inicializa el recomendador híbrido con los recomendadores de contenido, colaborativo y secuencial.

        Parámetros:
        content_recommender (ContentBasedRecommender): Recomendador basado en contenido.
        collaborative_recommender (CollaborativeFilteringRecommender): Recomendador de filtrado colaborativo.
        sequential_recommender (SequentialRecommender): Recomendador secuencial.
        """
        self.content_recommender = content_recommender
        self.collaborative_recommender = collaborative_recommender
        self.sequential_recommender = sequential_recommender

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
        if user_id not in self.collaborative_recommender.user_item_matrix.index:
            return self.recommend_popular(top_n)

        # Obtener recomendaciones de cada uno de los recomendadores base
        content_rec = self.content_recommender.recommend(user_id, top_n * 2)
        collab_rec = self.collaborative_recommender.recommend(user_id, top_n * 2)
        seq_rec = self.sequential_recommender.recommend(user_id, top_n * 2)

        # Unir todas las recomendaciones y calcular puntuaciones
        all_rec = np.concatenate((content_rec, collab_rec, seq_rec))

        # Excluir películas ya valoradas por el usuario
        user_rated_movies = (
            self.collaborative_recommender.user_item_matrix.loc[user_id].dropna().index
        )
        all_rec = [movie for movie in all_rec if movie not in user_rated_movies]

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
            self.collaborative_recommender.ratings["movieId"]
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
        self.content_recommender.update_data(ratings)
        self.collaborative_recommender.update_data(ratings)
        self.sequential_recommender.update_data(ratings)
