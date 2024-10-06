from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


class CollaborativeFilteringRecommender:
    def __init__(self, ratings):
        self.ratings = ratings
        self.user_item_matrix = ratings.pivot_table(
            index="userId", columns="movieId", values="rating"
        )
        self.user_sim_matrix = cosine_similarity(self.user_item_matrix.fillna(0))
        self.user_sim_df = pd.DataFrame(
            self.user_sim_matrix,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index,
        )

    def recommend(self, user_id, top_n=10):
        if user_id not in self.user_sim_df.index:
            return []
        sim_users = self.user_sim_df[user_id].sort_values(ascending=False)[1:]
        top_users = sim_users.index[:top_n]
        top_users_ratings = self.user_item_matrix.loc[top_users]
        mean_ratings = top_users_ratings.mean(axis=0)
        user_rated_movies = self.user_item_matrix.loc[user_id].dropna().index
        recommendations = (
            mean_ratings.drop(user_rated_movies)
            .sort_values(ascending=False)
            .head(top_n)
        )
        return recommendations.index.values

    def update_data(self, ratings):
        self.ratings = ratings
        self.user_item_matrix = ratings.pivot_table(
            index="userId", columns="movieId", values="rating"
        )
        self.user_sim_matrix = cosine_similarity(self.user_item_matrix.fillna(0))
        self.user_sim_df = pd.DataFrame(
            self.user_sim_matrix,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index,
        )
