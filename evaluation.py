# evaluation.py

import pandas as pd
import numpy as np


class Evaluator:
    def __init__(self, recommender, ratings, k=10):
        self.recommender = recommender
        self.ratings = ratings
        self.k = k
        # Dividir los datos en conjuntos de entrenamiento y prueba
        self.train_data, self.test_data = self.train_test_split()
        # Actualizar el recommender con los datos de entrenamiento
        self.recommender.update_data(self.train_data)

    def train_test_split(self, test_size=0.5):
        # Dividir los datos por usuario
        test_data = self.ratings.groupby("userId").apply(
            lambda x: x.sample(frac=test_size)
        )
        test_data.index = test_data.index.droplevel()
        train_data = self.ratings.drop(test_data.index)
        return train_data, test_data

    def precision_at_k(self):
        precisions = []
        for user_id in self.test_data["userId"].unique():
            actual = self.test_data[self.test_data["userId"] == user_id][
                "movieId"
            ].tolist()
            if not actual:
                continue
            recommended = self.recommender.recommend(user_id, top_n=self.k)
            if not any(recommended):
                continue
            recommended_set = set(recommended)
            actual_set = set(actual)
            precision = len(recommended_set & actual_set) / len(recommended_set)
            precisions.append(precision)
        return np.mean(precisions)

    def recall_at_k(self):
        recalls = []
        for user_id in self.test_data["userId"].unique():
            actual = self.test_data[self.test_data["userId"] == user_id][
                "movieId"
            ].tolist()
            if not actual:
                continue
            recommended = self.recommender.recommend(user_id, top_n=self.k)
            if not any(recommended):
                continue
            recommended_set = set(recommended)
            actual_set = set(actual)
            recall = len(recommended_set & actual_set) / len(actual_set)
            recalls.append(recall)
        return np.mean(recalls)

    def f1_score_at_k(self):
        precision = self.precision_at_k()
        recall = self.recall_at_k()
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)

    def ndcg_at_k(self):
        ndcgs = []
        for user_id in self.test_data["userId"].unique():
            actual = self.test_data[self.test_data["userId"] == user_id][
                "movieId"
            ].tolist()
            if not actual:
                continue
            recommended = self.recommender.recommend(user_id, top_n=self.k)
            if not any(recommended):
                continue
            dcg = 0.0
            for i, rec in enumerate(recommended):
                if rec in actual:
                    dcg += 1 / np.log2(i + 2)  # i+2 porque i empieza en 0
            idcg = sum([1 / np.log2(i + 2) for i in range(min(len(actual), self.k))])
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcgs.append(ndcg)
        return np.mean(ndcgs)

    def evaluate(self):
        precision = self.precision_at_k()
        recall = self.recall_at_k()
        f1_score = self.f1_score_at_k()
        ndcg = self.ndcg_at_k()
        return {
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1_score,
            "NDCG": ndcg,
        }
