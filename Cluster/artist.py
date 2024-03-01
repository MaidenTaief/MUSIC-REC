import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
data_by_artist = pd.read_csv('/Users/taief/Desktop/MUSIC REC/data/data_w_genres.csv')
features = ['acousticness', 'danceability', 'energy', 'instrumentalness',
            'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'popularity']

# Check if 'artists' column exists
if 'artists' not in data_by_artist.columns:
    raise ValueError("The dataset does not have an 'artists' column.")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_by_artist[features])

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Choose the optimal number of clusters from silhouette analysis
optimal_num_clusters = 6

# Perform KMeans clustering
kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=10)
cluster_labels = kmeans.fit_predict(X_pca)

# Add the cluster labels to the original dataset for profiling
data_by_artist['cluster'] = cluster_labels

# Calculate the cluster profiles for each feature
cluster_profile = data_by_artist.groupby('cluster')[features].mean()


# Define the recommendation function
def recommend_artists(input_artists, data, num_recommendations=5):
    recommendations = {}

    for artist in input_artists:
        # Check if the artist is in the dataset
        if artist not in data['artists'].values:
            print(f"{artist} not found in the dataset.")
            recommendations[artist] = []
            continue

        # Get the cluster for the input artist
        artist_cluster = data[data['artists'] == artist]['cluster'].values[0]
        # Get other artists from the same cluster
        possible_recommendations = data[data['cluster'] == artist_cluster]['artists'].tolist()
        # Remove the input artist from the recommendation list
        possible_recommendations = [a for a in possible_recommendations if a != artist]
        # Select a number of recommendations
        recommendations[artist] = np.random.choice(possible_recommendations,
                                                   min(num_recommendations, len(possible_recommendations)),
                                                   replace=False).tolist()

    return recommendations


input_artists = ['The Weeknd', 'Metallica', 'Linkin Park', 'John Lennon', 'Taylor Swift', 'Eminem']
recommendations = recommend_artists(input_artists, data_by_artist)

# Print out recommendations
for input_artist, recs in recommendations.items():
    print(f"Recommendations for {input_artist}: {recs}")

# Visualization
plt.figure(figsize=(14, 10))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=data_by_artist['cluster'], palette='viridis', alpha=0.5)

for artist in input_artists:
    artist_index = data_by_artist[data_by_artist['artists'] == artist].index[0]
    pca_coords = X_pca[artist_index]
    plt.scatter(pca_coords[0], pca_coords[1], color='red', s=100, label=artist)
    plt.text(pca_coords[0], pca_coords[1], artist, horizontalalignment='right')

    # Annotate recommended artists
    for recommended_artist in recommendations[artist]:
        if recommended_artist in data_by_artist['artists'].values:
            reco_index = data_by_artist[data_by_artist['artists'] == recommended_artist].index[0]
            reco_pca_coords = X_pca[reco_index]
            plt.scatter(reco_pca_coords[0], reco_pca_coords[1], color='blue', s=30)
            plt.text(reco_pca_coords[0], reco_pca_coords[1], recommended_artist, horizontalalignment='left')

plt.title('PCA-reduced Space with Input Artists and Recommendations')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()
