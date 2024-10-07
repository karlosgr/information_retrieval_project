import tkinter as tk
from tkinter import ttk
import csv
import os

# Rutas de los archivos CSV
CSV_PATH_MOVIES = os.path.join("data", "movies.csv")
CSV_PATH_RATINGS = os.path.join("data", "ratings.csv")


# Función para agregar película
def add_movie():
    movie_title = entry_movie_title.get()
    movie_genre = entry_movie_genre.get()

    # Leer el último ID de la película del archivo CSV
    try:
        with open(CSV_PATH_MOVIES, "r", newline="", encoding="utf-8-sig") as csvfile:
            reader = csv.DictReader(csvfile)
            movie_ids = [int(row["movieId"]) for row in reader]
            new_id = max(movie_ids) + 1 if movie_ids else 1
    except FileNotFoundError:
        new_id = 1

    # Agregar la nueva película al archivo CSV
    with open(CSV_PATH_MOVIES, "a", newline="", encoding="utf-8-sig") as csvfile:
        fieldnames = ["movieId", "title", "genres"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if new_id == 1:
            writer.writeheader()
        writer.writerow(
            {"movieId": new_id, "title": movie_title, "genres": movie_genre}
        )

    print(f"Película agregada: {movie_title}, Género: {movie_genre}")
    entry_movie_title.delete(0, tk.END)
    entry_movie_genre.delete(0, tk.END)


# Función para agregar valoración
def add_rating():
    user_id = entry_user_id.get()
    movie_id = entry_movie_id.get()
    rating = entry_rating.get()

    # Agregar la nueva valoración al archivo CSV
    with open(CSV_PATH_RATINGS, "a", newline="", encoding="utf-8-sig") as csvfile:
        fieldnames = ["userId", "movieId", "rating", "timestamp"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if os.path.getsize(CSV_PATH_RATINGS) == 0:
            writer.writeheader()
        writer.writerow(
            {
                "userId": user_id,
                "movieId": movie_id,
                "rating": rating,
                "timestamp": "NA",
            }
        )

    print(
        f"Valoración agregada: Usuario ID: {user_id}, Película ID: {movie_id}, Valoración: {rating}"
    )
    entry_user_id.delete(0, tk.END)
    entry_movie_id.delete(0, tk.END)
    entry_rating.delete(0, tk.END)


# Crear la ventana principal
window = tk.Tk()
window.title("Gestión de Películas y Valoraciones")
window.geometry("600x400")

# Sección de Película
frame_movie = ttk.LabelFrame(window, text="Agregar Película")
frame_movie.pack(padx=10, pady=10, fill="x")

ttk.Label(frame_movie, text="Título de la película:").grid(
    row=0, column=0, padx=5, pady=5
)
entry_movie_title = ttk.Entry(frame_movie)
entry_movie_title.grid(row=0, column=1, padx=5, pady=5)

ttk.Label(frame_movie, text="Género:").grid(row=1, column=0, padx=5, pady=5)
entry_movie_genre = ttk.Entry(frame_movie)
entry_movie_genre.grid(row=1, column=1, padx=5, pady=5)

btn_add_movie = ttk.Button(frame_movie, text="Agregar Película", command=add_movie)
btn_add_movie.grid(row=2, columnspan=2, pady=10)

# Sección de Valoraciones
frame_rating = ttk.LabelFrame(window, text="Agregar Valoración")
frame_rating.pack(padx=10, pady=10, fill="x")

ttk.Label(frame_rating, text="ID de Usuario:").grid(row=0, column=0, padx=5, pady=5)
entry_user_id = ttk.Entry(frame_rating)
entry_user_id.grid(row=0, column=1, padx=5, pady=5)

ttk.Label(frame_rating, text="ID de Película:").grid(row=1, column=0, padx=5, pady=5)
entry_movie_id = ttk.Entry(frame_rating)
entry_movie_id.grid(row=1, column=1, padx=5, pady=5)

ttk.Label(frame_rating, text="Valoración:").grid(row=2, column=0, padx=5, pady=5)
entry_rating = ttk.Entry(frame_rating)
entry_rating.grid(row=2, column=1, padx=5, pady=5)

btn_add_rating = ttk.Button(frame_rating, text="Agregar Valoración", command=add_rating)
btn_add_rating.grid(row=3, columnspan=2, pady=10)

# Iniciar la aplicación
window.mainloop()
