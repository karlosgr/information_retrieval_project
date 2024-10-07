import pandas as pd


class DataLoader:
    def __init__(self, ratings_file, movies_file, tags_file):
        """
        Inicializa el cargador de datos con las rutas de los archivos de calificaciones, películas y tags.

        Parámetros:
        ratings_file (str): Ruta del archivo CSV que contiene las calificaciones.
        movies_file (str): Ruta del archivo CSV que contiene los datos de las películas.
        tags_file (str): Ruta del archivo CSV que contiene los tags de las películas.
        """
        self.ratings_file = ratings_file
        self.movies_file = movies_file
        self.tags_file = tags_file

    def load_data(self):
        """
        Carga los datos de calificaciones, películas y tags desde los archivos CSV.

        Devuelve:
        tuple: Un par que contiene el DataFrame de calificaciones y el DataFrame de películas con metadatos combinados.
        """
        # Cargar los datos de calificaciones desde el archivo CSV
        ratings = pd.read_csv(self.ratings_file)

        # Cargar los datos de películas desde el archivo CSV
        movies = pd.read_csv(self.movies_file)

        # Cargar los datos de tags desde el archivo CSV
        tags = pd.read_csv(self.tags_file)

        # Combinar datos de películas y tags utilizando el ID de la película
        movies = pd.merge(movies, tags, on="movieId", how="left")

        # Rellenar los valores nulos en la columna 'tag' con una cadena vacía
        movies["tag"] = movies["tag"].fillna("")

        # Crear una columna 'metadata' que combine los géneros y los tags de cada película
        movies["metadata"] = movies["genres"] + " " + movies["tag"]

        return ratings, movies
