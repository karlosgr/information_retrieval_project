# main_evaluator.py

from data_loader import DataLoader
from recommenders.content_based_recommender import ContentBasedRecommender
from recommenders.collaborative_filtering import CollaborativeFilteringRecommender
from recommenders.sequential_recommender import SequentialRecommender
from recommenders.hybrid_recommender import HybridRecommender
from evaluation.evaluation import Evaluator
from evaluation.evaluation_gui import EvaluationGUI
import sys
from PyQt5.QtWidgets import QApplication


def main():
    """
    Función principal que carga los datos, inicializa los recomendadores y la interfaz gráfica para evaluar las recomendaciones.
    """
    # Cargar y preparar los datos
    data_loader = DataLoader("data/ratings.csv", "data/movies.csv", "data/tags.csv")
    ratings, movies = data_loader.load_data()

    # Inicializar los recomendadores
    recommenders = [
        ContentBasedRecommender,
        CollaborativeFilteringRecommender,
        SequentialRecommender,
    ]

    # Crear el recomendador híbrido utilizando los recomendadores base
    hybrid_recommender = HybridRecommender(recommenders, movies, ratings)

    # Inicializar el evaluador con el recomendador híbrido
    evaluator = Evaluator(hybrid_recommender, ratings, k=10)

    # Inicializar la aplicación Qt para la interfaz de evaluación
    app = QApplication(sys.argv)
    eval_gui = EvaluationGUI(evaluator)
    eval_gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
