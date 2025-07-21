import pandas as pd
import numpy as np
import joblib
import os
import random

class KKBoxRecommender:
    def __init__(self, model_path='models/', data_path='.'):
        # Load data
        self.songs = pd.read_csv(os.path.join(data_path, 'songs_cleaned_with_clusters.csv'))
        self.members = pd.read_csv(os.path.join(data_path, 'members.csv'))
        self.test = pd.read_csv(os.path.join(data_path, 'test_cleaned.csv'))

        # Load models and feature engineering results
        self.ensemble_models = joblib.load(os.path.join(model_path, 'ensemble_models.pkl'))
        self.ensemble_weights = joblib.load(os.path.join(model_path, 'ensemble_weights.pkl'))
        self.user_encoder = joblib.load(os.path.join(model_path, 'user_encoder.pkl'))
        self.song_encoder = joblib.load(os.path.join(model_path, 'song_encoder.pkl'))
        self.user_features_matrix = joblib.load(os.path.join(model_path, 'user_features_matrix.pkl'))
        self.song_features_matrix = joblib.load(os.path.join(model_path, 'song_features_matrix.pkl'))

    def recommend(self, user_id, n=10, candidate_pool_size=50):
        if user_id not in self.user_encoder.classes_:
            return []
        user_idx = self.user_encoder.transform([user_id])[0]
        user_predictions = []
        for model in self.ensemble_models:
            user_pred = model.predict(
                np.repeat(user_idx, len(self.song_encoder.classes_)),
                np.arange(len(self.song_encoder.classes_)),
                user_features=self.user_features_matrix,
                item_features=self.song_features_matrix
            )
            user_predictions.append(user_pred)
        ensemble_pred = np.average(user_predictions, axis=0, weights=self.ensemble_weights)
        # Get Top-N candidate pool
        candidate_indices = np.argsort(ensemble_pred)[::-1][:candidate_pool_size]
        # Randomly sample n songs
        if len(candidate_indices) > n:
            selected_indices = random.sample(list(candidate_indices), n)
        else:
            selected_indices = candidate_indices
        recommended_songs = self.song_encoder.inverse_transform(selected_indices)
        results = []
        for i, song_id in enumerate(recommended_songs):
            song_info = self.songs[self.songs['song_id'] == song_id]
            if not song_info.empty:
                song_data = song_info.iloc[0]
                results.append({
                    'song_id': song_id,
                    'name': song_data.get('name', 'Unknown'),
                    'artist': song_data.get('artist_name', 'Unknown'),
                    'genre': song_data.get('genre', 'Unknown'),
                    'cluster_id': song_data.get('cluster_id', 'N/A'),
                    'valence': song_data.get('valence', 0),
                    'energy': song_data.get('energy', 0),
                    'danceability': song_data.get('danceability', 0),
                    'ensemble_score': ensemble_pred[selected_indices[i]]
                })
        return results
    
    def get_test_users(self):
        """
        Get all user IDs from test_cleaned.csv
        :return: List of user IDs
        """
        return self.test['msno'].unique().tolist()