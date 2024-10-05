import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

# Datos ficticios de usuarios, calificaciones y contenido de películas
movies = pd.DataFrame(
    {
        "movieId": list(range(1, 101)),
        "title": [
            "Toy Story",
            "Jumanji",
            "Grumpy Old Men",
            "Waiting to Exhale",
            "Father of the Bride",
            "The Lion King",
            "Pulp Fiction",
            "Forrest Gump",
            "The Shawshank Redemption",
            "The Matrix",
            "Jurassic Park",
            "The Godfather",
            "The Dark Knight",
            "Inception",
            "Fight Club",
            "The Lord of the Rings",
            "Star Wars",
            "The Empire Strikes Back",
            "Return of the Jedi",
            "Titanic",
            "Gladiator",
            "The Green Mile",
            "Braveheart",
            "The Departed",
            "Schindler's List",
            "The Silence of the Lambs",
            "Se7en",
            "The Usual Suspects",
            "Saving Private Ryan",
            "The Prestige",
            "Interstellar",
            "Whiplash",
            "Mad Max: Fury Road",
            "The Avengers",
            "Black Panther",
            "Guardians of the Galaxy",
            "Iron Man",
            "Spider-Man",
            "The Incredible Hulk",
            "Doctor Strange",
            "Captain Marvel",
            "Thor",
            "Avengers: Infinity War",
            "Avengers: Endgame",
            "Deadpool",
            "Logan",
            "X-Men",
            "X2",
            "X-Men: The Last Stand",
            "Days of Future Past",
            "First Class",
            "Apocalypse",
            "Dark Phoenix",
            "Batman Begins",
            "The Dark Knight Rises",
            "Superman",
            "Man of Steel",
            "Wonder Woman",
            "Aquaman",
            "Justice League",
            "The Hobbit",
            "The Desolation of Smaug",
            "The Battle of the Five Armies",
            "The Hunger Games",
            "Catching Fire",
            "Mockingjay Part 1",
            "Mockingjay Part 2",
            "Django Unchained",
            "Once Upon a Time in Hollywood",
            "Inglourious Basterds",
            "Kill Bill",
            "Reservoir Dogs",
            "Jackie Brown",
            "Psycho",
            "Vertigo",
            "North by Northwest",
            "Rear Window",
            "The Birds",
            "Harry Potter and the Philosopher's Stone",
            "Harry Potter and the Chamber of Secrets",
            "Harry Potter and the Prisoner of Azkaban",
            "Harry Potter and the Goblet of Fire",
            "Harry Potter and the Order of the Phoenix",
            "Harry Potter and the Half-Blood Prince",
            "Harry Potter and the Deathly Hallows Part 1",
            "Harry Potter and the Deathly Hallows Part 2",
            "Fantastic Beasts and Where to Find Them",
            "Fantastic Beasts: The Crimes of Grindelwald",
            "The Great Gatsby",
            "Memento",
            "American Psycho",
            "The Wolf of Wall Street",
            "The Social Network",
            "Shutter Island",
            "The Martian",
            "Gravity",
            "Alien",
            "Blade Runner",
            "The Terminator",
            "The Godfather Part II",
        ],
        "genres": [
            "Animation|Children|Comedy",
            "Adventure|Children|Fantasy",
            "Comedy|Romance",
            "Comedy|Drama",
            "Comedy",
            "Animation|Adventure|Drama",
            "Crime|Drama|Thriller",
            "Drama|Romance",
            "Drama|Crime",
            "Action|Sci-Fi",
            "Action|Adventure|Sci-Fi",
            "Crime|Drama",
            "Action|Crime|Drama",
            "Action|Adventure|Sci-Fi",
            "Drama|Thriller",
            "Action|Adventure|Fantasy",
            "Action|Adventure|Fantasy",
            "Action|Adventure|Sci-Fi",
            "Action|Adventure|Sci-Fi",
            "Drama|Romance",
            "Action|Drama",
            "Crime|Drama|Fantasy",
            "Action|Biography|Drama",
            "Crime|Drama|Thriller",
            "Biography|Drama|History",
            "Crime|Drama|Thriller",
            "Crime|Drama|Mystery",
            "Crime|Drama|Mystery",
            "Drama|War",
            "Drama|Mystery|Sci-Fi",
            "Adventure|Drama|Sci-Fi",
            "Drama|Music",
            "Action|Adventure|Sci-Fi",
            "Action|Adventure|Sci-Fi",
            "Action|Adventure|Sci-Fi",
            "Action|Adventure|Sci-Fi",
            "Action|Adventure|Sci-Fi",
            "Action|Adventure|Fantasy",
            "Action|Adventure|Fantasy",
            "Action|Adventure|Fantasy",
            "Action|Adventure|Fantasy",
            "Action|Adventure|Fantasy",
            "Action|Adventure|Sci-Fi",
            "Action|Adventure|Sci-Fi",
            "Action|Comedy|Fantasy",
            "Action|Drama|Sci-Fi",
            "Action|Adventure|Fantasy",
            "Action|Adventure|Fantasy",
            "Action|Adventure|Sci-Fi",
            "Action|Adventure|Fantasy",
            "Action|Adventure|Fantasy",
            "Action|Adventure|Fantasy",
            "Action|Adventure|Sci-Fi",
            "Action|Adventure|Fantasy",
            "Action|Crime|Drama",
            "Action|Adventure|Fantasy",
            "Action|Adventure|Fantasy",
            "Action|Adventure|Sci-Fi",
            "Action|Adventure|Fantasy",
            "Action|Adventure|Fantasy",
            "Adventure|Drama|Fantasy",
            "Adventure|Drama|Fantasy",
            "Adventure|Drama|Fantasy",
            "Adventure|Sci-Fi",
            "Adventure|Sci-Fi",
            "Adventure|Sci-Fi",
            "Adventure|Sci-Fi",
            "Drama|Western",
            "Comedy|Drama",
            "Adventure|Drama|War",
            "Action|Crime|Drama",
            "Action|Crime|Thriller",
            "Crime|Drama|Thriller",
            "Horror|Mystery|Thriller",
            "Mystery|Romance|Thriller",
            "Adventure|Mystery|Thriller",
            "Mystery|Thriller",
            "Horror|Mystery|Thriller",
            "Adventure|Family|Fantasy",
            "Adventure|Family|Fantasy",
            "Adventure|Family|Fantasy",
            "Adventure|Family|Fantasy",
            "Adventure|Family|Fantasy",
            "Adventure|Family|Fantasy",
            "Adventure|Family|Fantasy",
            "Adventure|Family|Fantasy",
            "Adventure|Family|Fantasy",
            "Adventure|Family|Fantasy",
            "Drama|Romance",
            "Mystery|Thriller",
            "Crime|Drama|Horror",
            "Biography|Comedy|Crime",
            "Biography|Drama",
            "Mystery|Thriller",
            "Adventure|Drama|Sci-Fi",
            "Drama|Sci-Fi",
            "Action|Sci-Fi",
            "Action|Sci-Fi|Thriller",
            "Crime|Drama|Thriller",
            "Sci-Fi|Thriller",
        ],
    }
)


ratings = pd.DataFrame(
    {
        "userId": [1, 1, 1, 1],
        "movieId": [86, 85, 84, 1],
        "rating": [5, 5, 5, 1],
    }
)

# ratings = pd.DataFrame(
#     {
#         "userId": np.random.randint(1, 21, 4),
#         "movieId": np.random.randint(1, 101, 4),
#         "rating": np.random.randint(1, 6, 4),
#     }
# )

# Filtrado basado en contenido
# Convertimos los géneros a una representación TF-IDF
tfidf = TfidfVectorizer(token_pattern="[a-zA-Z0-9]+")
tfidf_matrix = tfidf.fit_transform(movies["genres"])

# Calculamos la similitud del coseno entre todas las películas
content_similarity = cosine_similarity(tfidf_matrix)

# Filtrado colaborativo - matriz de utilidad de usuarios y películas
user_movie_matrix = csr_matrix(
    ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0).values
)

# Calculamos la similitud del coseno entre los usuarios
user_similarity = cosine_similarity(user_movie_matrix)


# Recomendador híbrido
def hybrid_recommendation(user_id, movie_id, alpha=0.5, top_n=10):
    # Similitud basada en el contenido de la película
    movie_idx = movies[movies["movieId"] == movie_id].index[0]
    content_scores = content_similarity[movie_idx]

    # Similitud basada en usuarios
    user_idx = user_id - 1  # Los índices empiezan desde 0
    user_scores = user_similarity[user_idx]

    # Aseguramos que las dimensiones sean compatibles
    if len(content_scores) != len(user_scores):
        min_length = min(len(content_scores), len(user_scores))
        content_scores = content_scores[:min_length]
        user_scores = user_scores[:min_length]

    # Filtrado colaborativo: puntuaciones ponderadas de otros usuarios
    weighted_user_scores = user_scores @ user_movie_matrix.toarray()

    # Aseguramos que las dimensiones sean compatibles
    if len(content_scores) != len(weighted_user_scores):
        min_length = min(len(content_scores), len(weighted_user_scores))
        content_scores = content_scores[:min_length]
        weighted_user_scores = weighted_user_scores[:min_length]

    # Combinamos ambos enfoques
    hybrid_scores = alpha * content_scores + (1 - alpha) * weighted_user_scores

    # Ordenamos y mostramos las mejores recomendaciones
    recommended_idx = np.argsort(hybrid_scores)[::-1][:top_n]
    recommended_movies = movies.iloc[recommended_idx]
    return recommended_movies[["title", "genres"]]


# Ejemplo de recomendación para el usuario 1 y la película 1
recommended_movies = hybrid_recommendation(user_id=1, movie_id=1)
print(recommended_movies)
