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
plt.savefig('plot/distribution_of_features.png')
# ----------------------------------------
# Plotting relation between features and popularity
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(20, 25))
for i, feature in enumerate(features):
    sns.scatterplot(data=data_by_artist, x=feature, y='popularity', ax=axes[i//2, i%2])
    axes[i//2, i%2].set_title(f'Relation between {feature} and popularity')
plt.tight_layout()
# save plot in plot folder with name relation_between_features_and_popularity.png
plt.savefig('plot/relation_between_features_and_popularity.png')
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
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(scaled_features)
    silhouette_scores.append(silhouette_score(scaled_features, kmeans.labels_))

plt.figure(figsize=(10, 5))
plt.plot(range(2, 11), silhouette_scores)
plt.title('Silhouette')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.savefig('plot/silhouette_score.png')
plt.show()

# ----------------------------------------
kmeans = KMeans(n_clusters=6, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

# Adding the cluster labels to the original DataFrame
data_by_artist['cluster_label'] = clusters

# Plotting the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data_by_artist, x='danceability', y='energy', hue='cluster_label', palette='tab10')
plt.xlabel('Danceability')
plt.ylabel('Energy')
plt.title('Clusters of Artists')
plt.savefig('plot/clusters_of_artists.png')
plt.show()

# ----------------------------------------
from sklearn.decomposition import PCA

#2 for 2D
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_features)

# Create a new DataFrame for the principal components
principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

#PCA visualization
sns.scatterplot(x="PC1", y="PC2", data=principal_df, hue=data_by_artist['cluster_label'])
plt.title('PCA - Clusters Visualization')
plt.show()

# ----------------------------------------
import matplotlib.pyplot as plt

additional_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'valence', 'speechiness', 'tempo']

# Standardize the new set of features
additional_scaled_features = scaler.fit_transform(data_by_artist[additional_features])

# Apply K-Means clustering with the new features
additional_kmeans = KMeans(n_clusters=6, random_state=42)  
additional_clusters = additional_kmeans.fit_predict(additional_scaled_features)
data_by_artist['additional_cluster_label'] = additional_clusters

# Examine the centroids of the clusters
centroids = additional_kmeans.cluster_centers_

# Convert to a DataFrame for  interpretation
centroids_df = pd.DataFrame(centroids, columns=additional_features)

# Display the centroids to understand what each cluster represents
print(centroids_df)

# Project the new clustering results onto the PCA-reduced space
pca_result = PCA(n_components=2).fit_transform(additional_scaled_features)

# Create a new DataFrame for the PCA results
pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
pca_df['cluster'] = additional_clusters

# Plot the PCA results with the new cluster assignments
sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=pca_df, palette='tab10')
#plot title
plt.title('PCA - Additional Clusters Visualization')

#plotting the centroids
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=100)

plt.savefig('plot/pca_additional_clusters_visualization.png', dpi=300)

# Display the plot
plt.show()

# ----------------------------------------
# Define cluster labels based on the characteristics of the centroids

def assign_cluster_label(row):
    # Check the cluster of the row
    if row['instrumentalness'] > 0.5:
        return 'Instrumental'
    elif row['danceability'] > 0.5 and row['energy'] > 0.5:
        return 'Energetic Dance'
    elif row['valence'] > 0.5:
        return 'Positive Mood'
    elif row['speechiness'] > 0.5:
        return 'Speechy'
    elif row['acousticness'] > 0.5:
        return 'Acoustic'
    else:
        return 'Other'

# Map of cluster number to descriptive names
cluster_names = {
    0: 'Instrumental & Calm',
    1: 'Energetic & Popular',
    2: 'Mellow & Acoustic',
    3: 'Lyrical & Speech-Driven',
    4: 'Soft & Quiet',
    5: 'Upbeat & Danceable'
}

# Apply the mapping to your DataFrame
data_by_artist['cluster_name'] = data_by_artist['additional_cluster_label'].map(cluster_names)

# Verify the mapping
print(data_by_artist[['artists', 'additional_cluster_label', 'cluster_name']].head())

# ----------------------------------------
# Save the enhanced data with cluster labels
data_by_artist.to_csv('/Users/taief/Desktop/MUSIC REC/data/enhanced_data_with_clusters.csv', index=False)
# ----------------------------------------

# ----------------------------------------
