# main.py

from data_loader import DataLoader
from content_based_recommender import ContentBasedRecommender
from collaborative_filtering import CollaborativeFilteringRecommender
from sequential_recommender import SequentialRecommender
from hybrid_recommender import HybridRecommender
from gui import RecommenderGUI
import sys
from PyQt5.QtWidgets import QApplication


def main():
    data_loader = DataLoader("ratings.csv", "movies.csv", "tags.csv")
    ratings, movies = data_loader.load_data()
    tfidf_matrix = data_loader.preprocess_movies(movies)

    content_recommender = ContentBasedRecommender(tfidf_matrix, movies, ratings)
    collaborative_recommender = CollaborativeFilteringRecommender(ratings)
    sequential_recommender = SequentialRecommender(ratings)

    hybrid_recommender = HybridRecommender(
        content_recommender, collaborative_recommender, sequential_recommender
    )

    app = QApplication(sys.argv)
    gui = RecommenderGUI(hybrid_recommender, movies, ratings)
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
