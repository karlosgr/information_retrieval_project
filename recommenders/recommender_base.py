# recommender_base.py


class Recommender:
    def __init__(self, movies, ratings):
        """
        Clase base para los recomendadores.

        Args:
            movies (pd.DataFrame): DataFrame con información de las películas.
            ratings (pd.DataFrame): DataFrame con calificaciones de los usuarios.
        """
        self.movies = movies
        self.ratings = ratings

    def evaluate(self, user_id):
        """
        Evalúa todas las películas para un usuario dado y devuelve una serie de puntuaciones.

        Args:
            user_id (int): ID del usuario.

        Returns:
            pd.Series: Serie con IDs de películas como índice y puntuaciones como valores.
        """
        raise NotImplementedError(
            "El método evaluate debe ser implementado por las subclases."
        )
