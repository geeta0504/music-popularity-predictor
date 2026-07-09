import librosa
import numpy as np


def extract_features(audio_path):
    """
    Extracts approximate Spotify-style audio features from a raw audio file
    using signal processing (Librosa). These are proxies, not exact matches
    to Spotify's proprietary algorithms.
    """
    y, sr = librosa.load(audio_path, sr=None)

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo) if np.isscalar(tempo) else float(tempo[0])

    # Loudness (RMS -> dB)
    rms = librosa.feature.rms(y=y)[0]
    loudness = float(20 * np.log10(np.mean(rms) + 1e-6))

    # Energy (normalized RMS as a proxy, 0-1 range)
    energy = float(np.mean(rms) / (np.max(rms) + 1e-6))

    # Acousticness proxy: harmonic vs percussive energy ratio
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    harmonic_energy = np.sum(y_harmonic ** 2)
    percussive_energy = np.sum(y_percussive ** 2)
    acousticness = float(harmonic_energy / (harmonic_energy + percussive_energy + 1e-6))

    # Speechiness proxy: zero-crossing rate
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)[0]))

    # Danceability proxy: beat strength consistency (steadier beat = more danceable)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    danceability_raw = np.std(onset_env) / (np.mean(onset_env) + 1e-6)
    danceability = float(1 / (1 + danceability_raw))

    # Valence proxy: spectral centroid as a rough "brightness" indicator (weakest proxy)
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
    valence = float(min(spec_centroid / (sr / 4), 1.0))

    # Liveness proxy: spectral flatness (weakest proxy)
    liveness = float(np.mean(librosa.feature.spectral_flatness(y=y)[0]))

    return {
        "danceability": danceability,
        "energy": energy,
        "loudness": loudness,
        "tempo": tempo,
        "acousticness": acousticness,
        "valence": valence,
        "speechiness": zcr,
        "liveness": liveness,
    }