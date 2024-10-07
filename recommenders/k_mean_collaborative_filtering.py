# k_mean_collaborative_filtering.py

from recommenders.recommender_base import Recommender
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class KMeansCollaborativeFilteringRecommender(Recommender):
    def __init__(
        self, movies: pd.DataFrame, ratings: pd.DataFrame, num_clusters: int = 10
    ):
        """
        Inicializa el recomendador de filtrado colaborativo con clusterización K-Means.

        Args:
            movies (pd.DataFrame): DataFrame con información de las películas.
            ratings (pd.DataFrame): DataFrame con calificaciones de los usuarios.
            num_clusters (int, optional): Número de clusters para K-Means. Por defecto es 10.
        """
        super().__init__(movies, ratings)
        self.num_clusters = num_clusters
        self.user_item_matrix = self._create_user_item_matrix()
        self.user_sim_df = self._compute_pearson_similarity()
        self.cluster_labels = self._cluster_users()

    def _create_user_item_matrix(self) -> pd.DataFrame:
        """
        Crea la matriz usuario-item a partir de las calificaciones.

        Returns:
            pd.DataFrame: Matriz pivotada usuario-item.
        """
        user_item = self.ratings.pivot_table(
            index="userId", columns="movieId", values="rating"
        )
        return user_item

    def _compute_pearson_similarity(self) -> pd.DataFrame:
        """
        Calcula la matriz de similitud de Pearson entre usuarios.

        Returns:
            pd.DataFrame: Matriz de similitud usuario-usuario.
        """
        # Rellenar los valores faltantes con la media de cada usuario
        user_item_filled = self.user_item_matrix.apply(
            lambda row: row.fillna(row.mean()), axis=1
        )
        # Calcular la matriz de correlación de Pearson
        similarity = user_item_filled.T.corr(method="pearson", min_periods=5)
        similarity = similarity.fillna(0)  # Reemplazar NaNs con 0
        return similarity

    def _cluster_users(self) -> pd.Series:
        """
        Aplica K-Means para clusterizar a los usuarios.

        Returns:
            pd.Series: Series con las etiquetas de cluster para cada usuario.
        """
        # Rellenar los valores faltantes con la media de cada usuario
        user_item_filled = self.user_item_matrix.apply(
            lambda row: row.fillna(row.mean()), axis=1
        )
        # Normalizar los datos
        user_item_normalized = (
            user_item_filled - user_item_filled.mean()
        ) / user_item_filled.std()
        user_item_normalized = user_item_normalized.fillna(
            0
        )  # Reemplazar NaNs resultantes

        # Aplicar K-Means
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        kmeans.fit(user_item_normalized)
        labels = pd.Series(
            kmeans.labels_, index=self.user_item_matrix.index, name="Cluster"
        )
        return labels

    def evaluate(self, user_id: int) -> pd.Series:
        """
        Evalúa todas las películas para un usuario específico y devuelve una serie de puntuaciones.

        Args:
            user_id (int): ID del usuario para el cual generar puntuaciones.

        Returns:
            pd.Series: Serie de puntuaciones con movieId como índice.
        """
        if user_id not in self.user_sim_df.index:
            # Usuario nuevo, devolver puntuaciones neutras
            scores = pd.Series(0, index=self.user_item_matrix.columns)
            return scores

        # Obtener el cluster del usuario
        user_cluster = self.cluster_labels.loc[user_id]

        # Filtrar usuarios que pertenecen al mismo cluster
        users_in_cluster = self.cluster_labels[
            self.cluster_labels == user_cluster
        ].index
        if len(users_in_cluster) < 2:
            # Si el cluster tiene solo al usuario, expandir a todos los usuarios
            similar_users = (
                self.user_sim_df[user_id]
                .sort_values(ascending=False)[1 : self.num_clusters + 1]
                .index
            )
        else:
            # Obtener la similitud con otros usuarios en el mismo cluster
            sim_scores = self.user_sim_df.loc[users_in_cluster, user_id]
            sim_scores = sim_scores.drop(
                labels=[user_id], errors="ignore"
            )  # Eliminar al usuario mismo
            top_similar_users = (
                sim_scores.sort_values(ascending=False).head(self.num_clusters).index
            )
            similar_users = top_similar_users

        # Obtener las calificaciones de los usuarios similares
        top_users_ratings = self.user_item_matrix.loc[similar_users]
        mean_ratings = top_users_ratings.mean(axis=0)

        # Excluir películas ya calificadas por el usuario
        user_rated_movies = self.user_item_matrix.loc[user_id].dropna().index
        recommendations = mean_ratings.drop(
            user_rated_movies, errors="ignore"
        ).sort_values(ascending=False)

        # Retornar una serie con las puntuaciones
        return recommendations
