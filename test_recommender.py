import pandas as pd
from recommender import get_similar_songs, get_top_tracks

df = pd.read_csv("dataset.csv.zip", compression="zip")
features = ['danceability', 'energy', 'loudness', 'tempo', 'acousticness', 'valence', 'speechiness', 'liveness']

test_input = {
    'danceability': 0.6, 'energy': 0.7, 'loudness': -6.0, 'tempo': 120.0,
    'acousticness': 0.2, 'valence': 0.5, 'speechiness': 0.05, 'liveness': 0.15
}

print("Similar songs:")
print(get_similar_songs(test_input, df, features, genre="pop", top_n=5))

print("\nTop songs:")
print(get_top_tracks(df, genre="pop", top_n=5))