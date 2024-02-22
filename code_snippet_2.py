import pandas as pd

file_path = '/Users/taief/Desktop/MUSIC REC/data/data_by_artist.csv'
data_by_artist = pd.read_csv(file_path)
data_by_artist.head()
# ----------------------------------------
#check for missing values
print(data_by_artist.isnull().sum())
#check for duplicates
duplicates = data_by_artist.duplicated()
print(f"Number of duplicate rows: {duplicates.sum()}")
# check missing rows
missing_rows = data_by_artist.isnull().sum()
print(f"Number of missing rows: {missing_rows.sum()}")
# ----------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# Setting visual style for plots
sns.set(style="whitegrid")

# List of features to plot
features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'popularity']

# Plotting distributions
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(20, 25))
for i, feature in enumerate(features):
    sns.histplot(data=data_by_artist[feature], ax=axes[i//2, i%2], kde=True)
    axes[i//2, i%2].set_title(f'Distribution of {feature}')
plt.tight_layout()

# save plot in plot folder with name distribution_of_features.png
plt.savefig('/Users/taief/Desktop/MUSIC REC/plot/distribution_of_features_artist.png')
# ----------------------------------------
# Plotting relation between features and popularity
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(20, 25))
for i, feature in enumerate(features):
    sns.scatterplot(data=data_by_artist, x=feature, y='popularity', ax=axes[i//2, i%2])
    axes[i//2, i%2].set_title(f'Relation between {feature} and popularity')
plt.tight_layout()

plt.savefig('/Users/taief/Desktop/MUSIC REC/plot/relation_between_features_and_popularity_artist.png')
# ----------------------------------------
# clustering KMeans
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Selecting features for clustering (all available features)
features_for_clustering = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'tempo', 'valence']

# Standardizing the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data_by_artist[features_for_clustering])

# Applying K-Means Clustering to the dataset

kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

# Adding the cluster labels to the original DataFrame
data_by_artist['cluster_label'] = clusters

# Plotting the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data_by_artist, x='danceability', y='energy', hue='cluster_label', palette='tab10')
plt.xlabel('Danceability')
plt.ylabel('Energy')
plt.title('Clusters of Artists')
# ----------------------------------------
from sklearn.metrics import silhouette_score

silhouette_scores = []
for i in range(4, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(scaled_features)
    silhouette_scores.append(silhouette_score(scaled_features, kmeans.labels_))

plt.figure(figsize=(10, 5))
plt.plot(range(4, 11), silhouette_scores)
plt.title('Silhouette')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.savefig('/Users/taief/Desktop/MUSIC REC/plot/silhouette_score.png')
plt.show()
# ----------------------------------------
kmeans = KMeans(n_clusters=6, random_state=42)
data_by_artist['cluster'] = kmeans.fit_predict(data_by_artist[features_for_clustering])

# Check the size of each cluster
print(data_by_artist['cluster'].value_counts())
# ----------------------------------------
# get the centroids of the clusters
centroids = kmeans.cluster_centers_

# create a DataFrame with the centroids
centroids_df = pd.DataFrame(centroids, columns=features_for_clustering)

print(centroids_df)
# ----------------------------------------
from sklearn.decomposition import PCA

# Apply PCA to reduce the dimensions to 2
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data_by_artist[features_for_clustering])

# Add the PCA results to the DataFrame
data_by_artist['pca_one'] = pca_result[:, 0]
data_by_artist['pca_two'] = pca_result[:, 1]

# Visualize the PCA result
plt.figure(figsize=(10, 8))
sns.scatterplot(x='pca_one', y='pca_two', hue='cluster', data=data_by_artist, palette='viridis')
plt.title('PCA - 2D Projection of artists')
plt.savefig('/Users/taief/Desktop/MUSIC REC/plot/PCA_of_artists.png')
plt.show()
# ----------------------------------------
def assign_cluster_name(centroid):
    # Define thresholds for categorizing features
    thresholds = {
        'high_instrumentalness': 0.7,
        'high_speechiness': 0.7,
        'medium_danceability': 0.5,
        'medium_energy': 0.5,
        'medium_valence': 0.5,
        'medium_acousticness': 0.5
    }

    # First check for highly instrumental or speechy music
    if centroid['instrumentalness'] > thresholds['high_instrumentalness']:
        return 'Highly Instrumental'
    if centroid['speechiness'] > thresholds['high_speechiness']:
        return 'Wordy'

    # Second, check for Acoustic and energetic music, and positive vibes
    labels = []
    if centroid['danceability'] > thresholds['medium_danceability'] and centroid['energy'] > thresholds['medium_energy']:
        labels.append('Energetic Dance')
    if centroid['valence'] > thresholds['medium_valence']:
        labels.append('Positive Vibes')
    if centroid['acousticness'] > thresholds['medium_acousticness']:
        labels.append('Acoustic')

    # Finally, check for vocally rich music
    if not labels:
        if centroid['instrumentalness'] < 0.3:
            return 'Vocally Rich'
        return 'Varied'  # Use 'Varied' for centroids that don't fit other categories

    # Return the combined labels, or 'Varied' if no specific characteristics stand out
    return ', '.join(labels)  # Using comma as separator for readability


# Example usage:
centroid_example = {
    'acousticness': 0.4,
    'danceability': 0.7,
    'energy': 0.6,
    'instrumentalness': 0.2,
    'liveness': 0.2,
    'speechiness': 0.4,
    'tempo': 120,
    'valence': 0.8
}

cluster_name = assign_cluster_name(centroid_example)
print(cluster_name)
# ----------------------------------------
# Apply the mapping to your DataFrame
data_by_artist['cluster_name'] = data_by_artist.apply(lambda row: assign_cluster_name(row[features_for_clustering]), axis=1)

# show random 10 rows
print(data_by_artist[['artists', 'cluster', 'cluster_name']].sample(10))

#print one artist cluster name
print(data_by_artist[data_by_artist['artists'] == 'Linkin Park']['cluster_name'])
# ----------------------------------------
# Save the enhanced data with cluster labels
data_by_artist.to_csv('/Users/taief/Desktop/MUSIC REC/data/data_by_artist_with_clusters.csv', index=False)
# ----------------------------------------

# ----------------------------------------
