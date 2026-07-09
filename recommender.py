import numpy as np
import pandas as pd

def get_similar_songs(input_features_dict, df, features, genre=None, top_n=5):
    """
    input_features_dict: dict like {'danceability': 0.5, 'energy': 0.6, ...}
    df: the full dataset (must have track_name, artists, track_genre columns)
    features: list of the 8 audio feature column names
    genre: optional genre string to filter by
    """
    working_df = df.copy()
    if genre is not None:
        working_df = working_df[working_df['track_genre'] == genre]

    if working_df.empty:
        return pd.DataFrame()  # no songs in that genre, handle gracefully

    # Build input vector in the same feature order
    input_vector = np.array([input_features_dict[f] for f in features])

    # Normalize features so no single feature (e.g. loudness, tempo) dominates distance
    feature_matrix = working_df[features].values
    means = feature_matrix.mean(axis=0)
    stds = feature_matrix.std(axis=0) + 1e-6

    normalized_matrix = (feature_matrix - means) / stds
    normalized_input = (input_vector - means) / stds

    distances = np.linalg.norm(normalized_matrix - normalized_input, axis=1)
    working_df = working_df.copy()
    working_df['distance'] = distances

    result = working_df.sort_values('distance').head(top_n)
    return result[['track_name', 'artists', 'track_genre', 'popularity', 'distance']]


def get_top_tracks(df, genre=None, top_n=5):
    working_df = df.copy()
    if genre is not None:
        working_df = working_df[working_df['track_genre'] == genre]
    return working_df.sort_values('popularity', ascending=False).head(top_n)[
        ['track_name', 'artists', 'track_genre', 'popularity']
    ]