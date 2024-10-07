from recommenders.recommender_base import Recommender
import pandas as pd


class SequentialRecommender(Recommender):
    def __init__(self, movies, ratings):
        """
        Inicializa el recomendador secuencial con los datos de películas y calificaciones.

        Parámetros:
        movies (DataFrame): Datos de las películas.
        ratings (DataFrame): Datos de las calificaciones de los usuarios para las películas.
        """
        # Inicializa la clase base con los datos de películas y calificaciones
        super().__init__(movies, ratings)

        # Ordena las calificaciones por 'userId' y 'timestamp' para obtener una secuencia temporal
        self.ratings = ratings.sort_values(by=["userId", "timestamp"])

        # Agrupa las calificaciones por 'userId' y crea una lista de películas vistas por cada usuario
        self.user_sequences = self.ratings.groupby("userId")["movieId"].apply(list)

    def recommend(self, user_id, top_n=10):
        """
        Recomienda películas a un usuario basándose en la secuencia de las películas vistas por otros usuarios.

        Parámetros:
        user_id (int): ID del usuario para el que se harán las recomendaciones.
        top_n (int): Número de películas a recomendar.

        Devuelve:
        numpy.ndarray: IDs de las películas recomendadas.
        """
        # Si el ID del usuario no está en las secuencias de usuarios, devuelve una lista vacía
        if user_id not in self.user_sequences.index:
            return []

        # Obtén la secuencia de películas vistas por el usuario dado
        user_sequence = self.user_sequences[user_id]

        # Obtén la última película vista por el usuario
        last_movie = user_sequence[-1]

        # Lista para almacenar las próximas películas vistas después de la última película
        next_movies = []

        # Busca en las secuencias de todos los usuarios para encontrar qué película vieron después de la última película
        for seq in self.user_sequences:
            if last_movie in seq:
                idx = seq.index(last_movie)
                # Si hay una película después de la última película en la secuencia, añádela a la lista de próximas películas
                if idx + 1 < len(seq):
                    next_movies.append(seq[idx + 1])

        # Si no se encuentran próximas películas, devuelve una lista vacía
        if not next_movies:
            return []

        # Cuenta cuántas veces aparece cada próxima película y ordena por frecuencia
        next_movie_counts = pd.Series(next_movies).value_counts()

        # Devuelve las películas recomendadas (las más frecuentes) hasta un máximo de 'top_n'
        recommendations = next_movie_counts.head(top_n).index.values
        return recommendations

    def update_data(self, ratings):
        """
        Actualiza los datos de calificaciones y recalcula las secuencias de usuarios.

        Parámetros:
        ratings (DataFrame): Datos de las calificaciones de los usuarios para las películas.
        """
        # Actualiza las calificaciones y las ordena por 'userId' y 'timestamp' para mantener la secuencia temporal
        self.ratings = ratings.sort_values(by=["userId", "timestamp"])

        # Recalcula las secuencias de usuarios con los datos actualizados
        self.user_sequences = self.ratings.groupby("userId")["movieId"].apply(list)
