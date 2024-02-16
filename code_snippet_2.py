import pandas as pd

#load the data data_w_genres.csv
data_w_genres = pd.read_csv('/Users/taief/Desktop/MUSIC REC/data/data_w_genres.csv')
#print data head
print(data_w_genres.head())
# ----------------------------------------
# Check for missing values
print(data_w_genres.isnull().sum())

# Get a summary of the dataset
print(data_w_genres.describe())
# ----------------------------------------
import pandas as pd

data_w_genres = pd.read_csv('/Users/taief/Desktop/MUSIC REC/data/data_w_genres.csv')

# Replace empty lists with the string 'Unknown'
data_w_genres['genres'] = data_w_genres['genres'].apply(lambda x: 'Unknown' if x == '[]' else x)

# Check again for empty lists
empty_lists = data_w_genres['genres'].apply(lambda x: x == 'Unknown').sum()
print(f"Number of 'Unknown' genre entries: {empty_lists}")

# check for [] values in genres column
print(data_w_genres['genres'].head())
# ----------------------------------------
from sklearn.preprocessing import StandardScaler

# Fit and transform the data
scaler = StandardScaler()

# select features to scale

features_to_scale = ['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']

# Scale the features

data_w_genres[features_to_scale] = scaler.fit_transform(data_w_genres[features_to_scale])

# print the head of the data
print(data_w_genres.head())

# ----------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import ast  # Abstract Syntax Trees

# Load data
data_w_genres = pd.read_csv('/Users/taief/Desktop/MUSIC REC/data/data_w_genres.csv')

# Convert the 'genres' column to actual lists
data_w_genres['genres'] = data_w_genres['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Flatten the genre lists into one list
all_genres = [genre for sublist in data_w_genres['genres'] for genre in sublist]

# Count each genre's occurrence
genre_counts = Counter(all_genres)

# Create a DataFrame for the counts
genre_df = pd.DataFrame(genre_counts.items(), columns=['Genre', 'Count']).sort_values('Count', ascending=False)

# Plot the top 20 genres by count
plt.figure(figsize=(12, 8))
sns.barplot(x='Count', y='Genre', data=genre_df.head(20))
plt.title('Top 20 Genres by Count')
plt.savefig('/Users/taief/Desktop/MUSIC REC/plot/top_20_genres.png')
plt.show()

# ----------------------------------------
import seaborn as sns

numerical_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'popularity']

# Calculate the correlation matrix
corr_matrix = data_w_genres[numerical_features].corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('/Users/taief/Desktop/MUSIC REC/plot/correlation_heatmap_w_genres.png')
plt.show()

# ----------------------------------------
import seaborn as sns

# Selecting a subset of columns for pair plot visualization
selected_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'valence', 'popularity', 'loudness', 'speechiness']

# Creating pair plot
sns.pairplot(data_w_genres[selected_features])
plt.title('Pair Plot of Audio Features')
plt.savefig('/Users/taief/Desktop/MUSIC REC/plot/pair_plot_w_genres.png')
plt.show()

# ----------------------------------------
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data_w_genres = pd.read_csv('/Users/taief/Desktop/MUSIC REC/data/data_w_genres.csv')

# Feature Engineering based on identified correlations
data_w_genres['energy_acousticness_interaction'] = data_w_genres['energy'] * data_w_genres['acousticness']
data_w_genres['dance_valence_synergy'] = data_w_genres['danceability'] + data_w_genres['valence']
data_w_genres['loud_energy_interaction'] = data_w_genres['loudness'] * data_w_genres['energy']
data_w_genres['loud_acousticness_interaction'] = data_w_genres['loudness'] * data_w_genres['acousticness']
data_w_genres['loud_speech_interaction'] = data_w_genres['loudness'] * data_w_genres['speechiness']

# Standardize the data
scaler = StandardScaler()
features_to_scale = ['energy', 'acousticness', 'danceability', 'valence', 'loudness', 'speechiness', 
                     'energy_acousticness_interaction', 'dance_valence_synergy',
                     'loud_energy_interaction', 'loud_acousticness_interaction', 
                     'loud_speech_interaction']
scaled_features = scaler.fit_transform(data_w_genres[features_to_scale])

# KMeans Clustering with enhanced features
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(scaled_features)
data_w_genres['enhanced_cluster_label'] = clusters

# Saving the enhanced DataFrame with all new features and clusters
data_w_genres.to_csv('/Users/taief/Desktop/MUSIC REC/data/enhanced_data_with_clusters_w_genre.csv', index=False)

print("Fully enhanced dataset with comprehensive feature engineering and clusters created and saved successfully.")

# ----------------------------------------
from sklearn.metrics import silhouette_score

# Calculate silhouette scores for different values of k
silhouette_scores = []
for k in range(2, 11): 
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    score = silhouette_score(scaled_features, clusters)
    silhouette_scores.append((k, score))

# Find the optimal k with the highest silhouette score
optimal_k = max(silhouette_scores, key=lambda x: x[1])[0]
print(f"Optimal number of clusters: {optimal_k}")
#print silhouette scores for different values of k
print(silhouette_scores)
# ----------------------------------------
#import pca library
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Applying PCA for visualization
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_features)

# Plotting the clusters
plt.figure(figsize=(10, 6))
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=clusters, cmap='viridis', marker='o')
plt.title('PCA Cluster Visualization')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.colorbar(label='Cluster Label')
plt.show()

# ----------------------------------------
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
data_w_genres = pd.read_csv('/Users/taief/Desktop/MUSIC REC/data/data_w_genres.csv')

# Preprocessing: Replace empty genre lists with 'Unknown'
data_w_genres['genres'] = data_w_genres['genres'].apply(lambda x: 'Unknown' if x == '[]' else x)

# Select features for clustering
features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'valence', 'speechiness']
X = data_w_genres[features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
data_w_genres['cluster'] = kmeans.fit_predict(X_scaled)

# Calculate and interpret centroids
centroids = kmeans.cluster_centers_
centroids_df = pd.DataFrame(scaler.inverse_transform(centroids), columns=features)
print(centroids_df)

# Define a labeling function based on centroids
def label_genre_clusters(row):
    labels = []
    if row['acousticness'] > 0.5:
        labels.append('High Acousticness')
    if row['energy'] > 0.5:
        labels.append('High Energy')
    if row['danceability'] > 0.5:
        labels.append('High Danceability')
    if row['valence'] > 0.5:
        labels.append('High Valence')
    if row['speechiness'] > 0.5:
        labels.append('High Speechiness')
    
    # Combine labels if multiple conditions are met, or label as 'Mixed' if none
    return ', '.join(labels) if labels else 'Mixed'

centroids_df['label'] = centroids_df.apply(label_genre_clusters, axis=1)

# Map the labels from centroids to each genre in the original DataFrame
cluster_labels = centroids_df['label'].to_dict()
data_w_genres['cluster_label'] = data_w_genres['cluster'].map(cluster_labels)

# Visualize with PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
data_w_genres['pca_one'] = pca_result[:,0]
data_w_genres['pca_two'] = pca_result[:,1]

plt.figure(figsize=(10,8))
sns.scatterplot(
    x="pca_one", y="pca_two",
    hue="cluster",
    palette=sns.color_palette("hsv", 5),
    data=data_w_genres,
    legend="full",
    alpha=0.8
)

# Plot centroids
centroids_pca = pca.transform(centroids)
plt.scatter(centroids_pca[:,0], centroids_pca[:,1], c='red', s=50, marker='X')

# Save the enhanced dataset
data_w_genres.to_csv('/Users/taief/Desktop/MUSIC REC/data/enhanced_data_with_clusters_genre.csv', index=False)

plt.title('PCA Plot of Genres Clustered')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.savefig('/Users/taief/Desktop/MUSIC REC/plot/pca_genres_clusters_visualization.png', dpi=300)
plt.show()

# ----------------------------------------

# ----------------------------------------
