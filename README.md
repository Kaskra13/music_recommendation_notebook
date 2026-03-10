# Music Recommender

A content-based music recommendation system built on the Spotify Tracks Dataset.
It compares songs using audio features and genre information, offering three
distinct recommendation strategies: Cosine Similarity, k-Nearest Neighbors, and KMeans Clustering.

---

## Dataset

- Source: Spotify Tracks Dataset (Kaggle)
  https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset/data
- Raw size: 114,000 tracks x 21 columns
- After deduplication (by track_id, then by artist + track name): ~81,343 tracks

---

## Requirements

Install dependencies with:

    pip install numpy pandas scikit-learn matplotlib seaborn

Python 3.8+ recommended.

---

## Project Structure

    music_recommendation_notebook/
    ├── dataset.csv                # Spotify Tracks Dataset (download from Kaggle)
    ├── music_recommender.ipynb    # Main notebook
    └── README.txt                 # This file

---

## Pipeline Overview

### 1. Load & Preprocess
- Load dataset.csv into a DataFrame
- Drop rows with missing values (artists, album_name, track_name)
- Deduplicate by track_id, then by (artists, track_name)
- Separate identity columns (track_id, artists, etc.) from numerical audio features

### 2. Genre Mapping
- 114 detailed Spotify genres are mapped to 12 broader categories:
    acoustic/folk, rock, metal, pop, rnb/soul/funk, hip-hop,
    electronic/dance, latin/world, asian, jazz/classical, functional, other
- Broad genres are one-hot encoded and weighted (genre_weight = 0.1)
  to softly influence recommendations without dominating audio features

### 3. Feature Engineering
- Numerical audio features used (11 total):
    danceability, energy, key, loudness, mode, speechiness,
    acousticness, instrumentalness, liveness, valence, tempo
- Scaled with MinMaxScaler
- Combined with weighted genre one-hot vectors → shape (81343, 23)
- L2-normalized per row for cosine-distance compatibility

---

## Recommendation Methods

### Cosine Similarity
Direct cosine similarity between the query track and all tracks in the dataset.
Returns the top-N most similar tracks by similarity score (1 = identical).

    recommend_cosine(track_idx, X_final, info, n_recs=10)

### k-Nearest Neighbors (KNN)
Fits a NearestNeighbors model with cosine metric. Efficiently retrieves
the N closest neighbors without computing the full similarity matrix.

    knn_recommender(track_idx, knn_model, X_final, info, n_recs=10)

### KMeans Clustering
Groups all tracks into k clusters. For a query track, only tracks within
the same cluster are considered, then ranked by cosine similarity.
Optimal k is selected via the Elbow Method (WCSS plot); k=4 was used.
Silhouette score for k=2: 0.1718.

    recommend_kmeans(track_idx, X_final, info, n_recs=10, n_clusters=4)

---

## Exploratory Analysis

- Histograms of key audio features (danceability, energy, acousticness,
  instrumentalness, valence, tempo, loudness)
- Pearson correlation heatmap of all numerical features
- Elbow Method plot for KMeans cluster selection
- PCA 2D scatter plot of KMeans cluster assignments

---

## Example Usage

    # Cosine similarity - "Comedy" by Gen Hoshino (index 0)
    recs = recommend_cosine(0, X_final, info, n_recs=5)

    # KNN - "The Nights" by Avicii (index 17177)
    recs = knn_recommender(17177, knn_model, X_final, info, n_recs=5)

    # KMeans - "Radioactive" by Imagine Dragons (index 65870)
    recs = recommend_kmeans(65870, X_final, info, n_recs=5, n_clusters=4)

---

## Notes

- The recommender is purely content-based: it uses audio features and genre,
  not user listening history or collaborative filtering.
- Genre weight (0.1) can be tuned: higher values bias recommendations
  toward same-genre tracks; lower values prioritise audio feature similarity.
- KMeans results are non-deterministic across runs unless random_state is fixed.
- PCA is used only for visualisation and cluster diagnostics, not for the
  final feature vectors used in recommendations.
