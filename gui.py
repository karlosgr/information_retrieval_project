# gui.py

import sys
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QLineEdit,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
)
from PyQt5.QtCore import Qt


# Widget personalizado para mostrar título y estrellas
class MovieRatingWidget(QWidget):
    def __init__(self, movie_title, stars, parent=None):
        super().__init__(parent)

        # Crear etiquetas
        self.title_label = QLabel(movie_title)
        self.stars_label = QLabel(stars)

        # Opcional: Configurar propiedades de las etiquetas
        self.title_label.setStyleSheet(
            """
            background: transparent;
            font-size: 13px;
            color: #2c3e50;  /* Color de la fuente para contraste */
        """
        )
        self.stars_label.setStyleSheet(
            """
            background: transparent;
            font-size: 13px;
            color: #2c3e50;  /* Color de la fuente para contraste */
        """
        )

        # Layout horizontal sin mucho espaciado
        layout = QHBoxLayout()
        layout.addWidget(self.title_label)
        layout.addStretch()  # Espacio flexible entre título y estrellas
        layout.addWidget(self.stars_label)
        layout.setContentsMargins(
            5, 1, 5, 1
        )  # Reducir los márgenes alrededor del contenido
        layout.setSpacing(10)  # Espaciado entre elementos en la misma fila

        self.setLayout(layout)


class RecommenderGUI(QWidget):
    def __init__(self, hybrid_recommender, movies, ratings):
        super().__init__()
        self.hybrid_recommender = hybrid_recommender
        self.movies = movies
        self.ratings = ratings
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Sistema de Recomendación de Películas")
        self.setGeometry(100, 100, 600, 600)

        # Widgets
        self.user_label = QLabel("ID de Usuario:")
        self.user_input = QLineEdit()
        self.n_label = QLabel("Número de Recomendaciones:")
        self.n_input = QLineEdit("10")  # Valor por defecto
        self.recommend_button = QPushButton("Recomendar")
        self.recommend_button.clicked.connect(self.get_recommendations)

        self.rated_movies_label = QLabel("Películas valoradas por el usuario:")
        self.rated_movies_list = QListWidget()
        self.recommendations_label = QLabel("Recomendaciones:")
        self.recommendations_list = QListWidget()

        # Layouts
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.user_label)
        input_layout.addWidget(self.user_input)
        input_layout.addWidget(self.n_label)
        input_layout.addWidget(self.n_input)
        input_layout.addWidget(self.recommend_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(input_layout)
        main_layout.addWidget(self.rated_movies_label)
        main_layout.addWidget(self.rated_movies_list)
        main_layout.addWidget(self.recommendations_label)
        main_layout.addWidget(self.recommendations_list)

        self.setLayout(main_layout)

        # Aplicar estilos (opcional)
        self.setStyleSheet(
            """
            QWidget {
                background-color: #2c3e50;
                color: #ecf0f1;
                font-family: Arial;
            }
            QLabel {
                font-size: 14px;
            }
            QLineEdit {
                background-color: #ecf0f1;
                color: #2c3e50;
                padding: 5px;
                border-radius: 5px;
            }
            QPushButton {
                background-color: #e74c3c;
                color: #ecf0f1;
                padding: 10px;
                border: none;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QListWidget {
                background-color: #ecf0f1;
                color: #2c3e50;
                border: none;
                padding: 5px;
                border-radius: 5px;
            }
        """
        )

    def get_star_rating(self, rating, max_stars=5):
        full_star = "★"
        half_star = (
            "½"  # Puedes usar otro símbolo si prefieres una media estrella visual
        )
        empty_star = "☆"

        # Convertir la calificación en número de estrellas llenas y medias
        full_stars = int(rating)
        half_star_flag = False
        if (rating - full_stars) >= 0.5:
            half_star_flag = True

        stars = full_star * full_stars
        if half_star_flag:
            stars += half_star
        stars += empty_star * (max_stars - full_stars - (1 if half_star_flag else 0))

        return stars

    def get_recommendations(self):
        user_id_text = self.user_input.text()
        n_text = self.n_input.text()
        try:
            user_id = int(user_id_text)
            top_n = int(n_text)
        except ValueError:
            QMessageBox.warning(
                self,
                "Error",
                "Por favor, introduce un ID de usuario y un número válido.",
            )
            return

        # Mostrar películas valoradas por el usuario
        user_ratings = self.ratings[self.ratings["userId"] == user_id]
        if user_ratings.empty:
            QMessageBox.information(
                self, "Información", "El usuario no ha valorado ninguna película."
            )
            self.rated_movies_list.clear()
        else:
            self.rated_movies_list.clear()
            for _, row in user_ratings.iterrows():
                movie_id = row["movieId"]
                rating = row["rating"]
                movie_title = self.movies.loc[
                    self.movies["movieId"] == movie_id, "title"
                ].values
                if movie_title.size > 0:
                    title = movie_title[0]
                else:
                    title = f"ID {movie_id}"

                stars = self.get_star_rating(rating)

                # Crear un nuevo QListWidgetItem
                item = QListWidgetItem()
                # Crear el widget personalizado
                widget = MovieRatingWidget(title, stars)
                # Establecer el tamaño del ítem según el widget
                item.setSizeHint(widget.sizeHint())
                # Agregar el ítem a la lista
                self.rated_movies_list.addItem(item)
                self.rated_movies_list.setItemWidget(item, widget)

        # Obtener recomendaciones
        recommendations = self.hybrid_recommender.recommend(user_id, top_n=top_n)
        if not recommendations:
            QMessageBox.information(
                self, "Sin recomendaciones", "No se encontraron recomendaciones."
            )
            self.recommendations_list.clear()
            return

        self.recommendations_list.clear()
        for rec_movie_id in recommendations:
            movie_title = self.movies.loc[
                self.movies["movieId"] == rec_movie_id, "title"
            ].values
            if movie_title.size > 0:
                self.recommendations_list.addItem(f"{movie_title[0]}")
            else:
                self.recommendations_list.addItem(
                    f"ID {rec_movie_id} (Título no disponible)"
                )
