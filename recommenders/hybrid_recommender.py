# hybrid_recommender.py
from recommenders.recommender_base import Recommender
import numpy as np


class HybridRecommender(Recommender):
    def __init__(self, recommenders, movies, ratings, weights=None):
        super().__init__(movies, ratings)
        self.recommenders = [
            recommender(movies, ratings) for recommender in recommenders
        ]
        if weights is None:
            self.weights = [1 / len(recommenders)] * len(recommenders)
        else:
            self.weights = weights

    def recommend(self, user_id, top_n=10):

        scores_list = []
        for recommender in self.recommenders:
            scores = recommender.recommend(user_id, top_n * 2)
            scores_list.append(scores)

        # Unir todas las recomendaciones y calcular puntuaciones
        all_rec = np.concatenate(scores_list)

        unique_rec, counts = np.unique(all_rec, return_counts=True)
        rec_scores = dict(zip(unique_rec, counts))
        # Ordenar las recomendaciones por puntuación
        sorted_rec = sorted(rec_scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = [rec[0] for rec in sorted_rec[:top_n]]
        return recommendations

    def update_data(self, ratings):
        # Actualizar los datos en los recomendadores
        for recommender in self.recommenders:
            recommender.update_data(ratings)
