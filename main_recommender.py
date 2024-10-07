# main.py

from data_loader import DataLoader
from recommenders.content_based_recommender import ContentBasedRecommender
from recommenders.collaborative_filtering import CollaborativeFilteringRecommender
from recommenders.sequential_recommender import SequentialRecommender
from recommenders.hybrid_recommender import HybridRecommender
from gui import RecommenderGUI
import sys
from PyQt5.QtWidgets import QApplication


def main():
    data_loader = DataLoader("data/ratings.csv", "data/movies.csv", "data/tags.csv")
    ratings, movies = data_loader.load_data()

    recommenders = [
        ContentBasedRecommender,
        CollaborativeFilteringRecommender,
        SequentialRecommender,
    ]

    hybrid_recommender = HybridRecommender(recommenders, movies, ratings)

    app = QApplication(sys.argv)
    gui = RecommenderGUI(hybrid_recommender, movies, ratings)
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
