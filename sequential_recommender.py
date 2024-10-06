import pandas as pd


class SequentialRecommender:
    def __init__(self, ratings):
        self.ratings = ratings.sort_values(by=["userId", "timestamp"])
        self.user_sequences = self.ratings.groupby("userId")["movieId"].apply(list)

    def recommend(self, user_id, top_n=10):
        if user_id not in self.user_sequences.index:
            return []
        user_sequence = self.user_sequences[user_id]
        last_movie = user_sequence[-1]
        next_movies = []
        for seq in self.user_sequences:
            if last_movie in seq:
                idx = seq.index(last_movie)
                if idx + 1 < len(seq):
                    next_movies.append(seq[idx + 1])
        if not next_movies:
            return []
        next_movie_counts = pd.Series(next_movies).value_counts()
        recommendations = next_movie_counts.head(top_n).index.values
        return recommendations

    def update_data(self, ratings):
        self.ratings = ratings.sort_values(by=["userId", "timestamp"])
        self.user_sequences = self.ratings.groupby("userId")["movieId"].apply(list)
