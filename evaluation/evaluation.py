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

    def train_test_split(self, test_size=0.8):
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

    def map_at_k(self):
        average_precisions = []
        for user_id in self.test_data["userId"].unique():
            actual = self.test_data[self.test_data["userId"] == user_id][
                "movieId"
            ].tolist()
            if not actual:
                continue
            recommended = self.recommender.recommend(user_id, top_n=self.k)
            if not any(recommended):
                continue
            hits = 0
            sum_precisions = 0
            for i, rec in enumerate(recommended):
                if rec in actual:
                    hits += 1
                    precision_i = hits / (i + 1)
                    sum_precisions += precision_i
            if hits > 0:
                average_precision = sum_precisions / len(actual)
                average_precisions.append(average_precision)
        return np.mean(average_precisions) if average_precisions else 0

    def mrr_at_k(self):
        reciprocal_ranks = []
        for user_id in self.test_data["userId"].unique():
            actual = self.test_data[self.test_data["userId"] == user_id][
                "movieId"
            ].tolist()
            if not actual:
                continue
            recommended = self.recommender.recommend(user_id, top_n=self.k)
            if not any(recommended):
                continue
            rank = next(
                (i + 1 for i, rec in enumerate(recommended) if rec in actual), None
            )
            if rank:
                reciprocal_ranks.append(1 / rank)
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0

    def hit_rate_at_k(self):
        hits = 0
        total = 0
        for user_id in self.test_data["userId"].unique():
            actual = self.test_data[self.test_data["userId"] == user_id][
                "movieId"
            ].tolist()
            if not actual:
                continue
            recommended = self.recommender.recommend(user_id, top_n=self.k)
            if not any(recommended):
                continue
            if set(recommended) & set(actual):
                hits += 1
            total += 1
        return hits / total if total > 0 else 0

    def coverage(self):
        recommended_items = set()
        all_items = set(self.train_data["movieId"].unique())
        for user_id in self.train_data["userId"].unique():
            recommended = self.recommender.recommend(user_id, top_n=self.k)
            recommended_items.update(recommended)
        return len(recommended_items) / len(all_items) if all_items else 0

    def evaluate(self):
        precision = self.precision_at_k()
        recall = self.recall_at_k()
        f1_score = self.f1_score_at_k()
        ndcg = self.ndcg_at_k()
        map_k = self.map_at_k()
        mrr_k = self.mrr_at_k()
        hit_rate = self.hit_rate_at_k()
        coverage = self.coverage()
        return {
            "Precision@k": precision,
            "Recall@k": recall,
            "F1-Score@k": f1_score,
            "NDCG@k": ndcg,
            "MAP@k": map_k,
            "MRR@k": mrr_k,
            "Hit Rate@k": hit_rate,
            "Coverage": coverage,
        }
