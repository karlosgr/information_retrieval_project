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
    QMessageBox,
)
from PyQt5.QtCore import Qt


class RecommenderGUI(QWidget):
    def __init__(self, hybrid_recommender, movies, ratings):
        super().__init__()
        self.hybrid_recommender = hybrid_recommender
        self.movies = movies
        self.ratings = ratings
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Sistema de Recomendación de Películas")
        self.setGeometry(100, 100, 800, 600)

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
                    self.rated_movies_list.addItem(
                        f"{movie_title[0]} (Calificación: {rating})"
                    )
                else:
                    self.rated_movies_list.addItem(
                        f"ID {movie_id} (Calificación: {rating})"
                    )

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
