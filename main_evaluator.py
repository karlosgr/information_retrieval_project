# main_evaluator.py

from data_loader import DataLoader
from content_based_recommender import ContentBasedRecommender
from collaborative_filtering import CollaborativeFilteringRecommender
from sequential_recommender import SequentialRecommender
from hybrid_recommender import HybridRecommender
from evaluation import Evaluator
from evaluation_gui import EvaluationGUI
import sys
from PyQt5.QtWidgets import QApplication


def main():
    # Cargar y preparar los datos
    data_loader = DataLoader("ratings.csv", "movies.csv", "tags.csv")
    ratings, movies = data_loader.load_data()
    tfidf_matrix = data_loader.preprocess_movies(movies)

    # Inicializar los recomendadores
    content_recommender = ContentBasedRecommender(tfidf_matrix, movies, ratings)
    collaborative_recommender = CollaborativeFilteringRecommender(ratings)
    sequential_recommender = SequentialRecommender(ratings)

    hybrid_recommender = HybridRecommender(
        content_recommender, collaborative_recommender, sequential_recommender
    )

    # Inicializar el evaluador
    evaluator = Evaluator(hybrid_recommender, ratings, k=10)

    # Inicializar la aplicación Qt
    app = QApplication(sys.argv)
    eval_gui = EvaluationGUI(evaluator)
    eval_gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
