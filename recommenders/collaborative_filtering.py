from recommenders.recommender_base import Recommender
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


class CollaborativeFilteringRecommender(Recommender):
    def __init__(self, movies, ratings):
        """
        Inicializa el recomendador de filtrado colaborativo con los datos de películas y calificaciones.

        Parámetros:
        movies (DataFrame): Datos de las películas.
        ratings (DataFrame): Datos de las calificaciones de los usuarios para las películas.
        """
        # Inicializa la clase base con los datos de películas y calificaciones
        super().__init__(movies, ratings)

        # Crea una matriz usuario-item donde las filas son usuarios, las columnas son películas y los valores son calificaciones
        self.user_item_matrix = ratings.pivot_table(
            index="userId", columns="movieId", values="rating"
        )

        # Calcula la matriz de similitud entre usuarios utilizando similitud coseno, rellenando los valores faltantes con 0
        self.user_sim_matrix = cosine_similarity(self.user_item_matrix.fillna(0))

        # Convierte la matriz de similitud en un DataFrame para un acceso fácil
        self.user_sim_df = pd.DataFrame(
            self.user_sim_matrix,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index,
        )

    def recommend(self, user_id, top_n=10):
        """
        Recomienda películas a un usuario en función de las calificaciones de los usuarios más similares.

        Parámetros:
        user_id (int): ID del usuario para el que se harán las recomendaciones.
        top_n (int): Número de películas a recomendar.

        Devuelve:
        numpy.ndarray: IDs de las películas recomendadas.
        """
        # Si el ID del usuario no está en el DataFrame de similitud, devuelve una lista vacía
        if user_id not in self.user_sim_df.index:
            return []

        # Obtén los usuarios similares al usuario dado, ordenados por similitud en orden descendente
        # Omite la primera entrada ya que será el propio usuario (similitud de 1)
        sim_users = self.user_sim_df[user_id].sort_values(ascending=False)[1:]

        # Obtén los N usuarios más similares
        top_users = sim_users.index[:top_n]

        # Obtén las calificaciones de los usuarios más similares para todas las películas
        top_users_ratings = self.user_item_matrix.loc[top_users]

        # Calcula la calificación ponderada para cada película entre los usuarios más similares
        # Multiplica las calificaciones de cada usuario por su similitud con el usuario objetivo
        weighted_ratings = top_users_ratings.mul(sim_users[top_users], axis=0)
        # Calcula la media ponderada dividiendo la suma de las calificaciones ponderadas por la suma de las similitudes
        mean_ratings = weighted_ratings.sum(axis=0) / sim_users[top_users].sum()

        # Obtén la lista de películas que el usuario ya ha calificado
        user_rated_movies = self.user_item_matrix.loc[user_id].dropna().index

        # Filtra las películas que el usuario ya ha calificado y ordena las películas restantes por calificación media
        recommendations = (
            mean_ratings.drop(
                user_rated_movies
            )  # Elimina las películas ya calificadas por el usuario
            .sort_values(
                ascending=False
            )  # Ordena las películas por calificación media en orden descendente
            .head(top_n)  # Selecciona las top_n películas con mayor calificación
        )

        # Devuelve los IDs de las películas recomendadas
        return recommendations.index.values

    def update_data(self, ratings):
        """
        Actualiza los datos de calificaciones y recalcula la matriz de similitud entre usuarios.

        Parámetros:
        ratings (DataFrame): Datos de las calificaciones de los usuarios para las películas.
        """
        # Actualiza las calificaciones almacenadas con los nuevos datos
        self.ratings = ratings

        # Recrea la matriz usuario-item con las calificaciones actualizadas
        self.user_item_matrix = ratings.pivot_table(
            index="userId", columns="movieId", values="rating"
        )

        # Recalcula la matriz de similitud entre usuarios con la matriz usuario-item actualizada
        # Rellena los valores faltantes con 0 antes de calcular la similitud
        self.user_sim_matrix = cosine_similarity(self.user_item_matrix.fillna(0))

        # Convierte la matriz de similitud actualizada en un DataFrame para un acceso fácil
        self.user_sim_df = pd.DataFrame(
            self.user_sim_matrix,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index,
        )
