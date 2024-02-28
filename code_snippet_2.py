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
# Descriptive statistics
descriptive_stats = data_by_artist.describe()
print(descriptive_stats)
# ----------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'popularity']

# Select only the specified features from the DataFrame
features_data = data_by_artist[features]

# Compute the correlation matrix
correlation_matrix = features_data.corr()

# Plot the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix for Selected Features')
plt.show()

# ----------------------------------------
from sklearn.preprocessing import StandardScaler

# feature selection for clustering
features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
X = data_by_artist[features]

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# ----------------------------------------
# PCA

from sklearn.decomposition import PCA

# Keep 95% of variance
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# Number of components PCA chose after fitting the data
n_pca_components = X_pca.shape[1]
print(f"PCA optimal {n_pca_components}")

# print variance explained by each component
print(pca.explained_variance_ratio_)

# get the eigenvectors of the covariance matrix
loadings = (pca.components_)

# Create a DataFrame with the loadings and the feature names for better interpretability

loadings_df = pd.DataFrame(loadings.T, columns=[f'PC{i+1}' for i in range(loadings.shape[0])], index=features)

print(loadings_df)
# ----------------------------------------
import matplotlib.pyplot as plt
import numpy as np

loadings2 = pca.components_[:2].T

loadings_df2 = pd.DataFrame(loadings2, columns=['PC1', 'PC2'], index=features)

# create the biplot
plt.figure(figsize=(12, 9))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.3, color='lightgrey')  

# Determine the maximum range of PCA scores
x_max, y_max = np.max(np.abs(X_pca[:, 0])), np.max(np.abs(X_pca[:, 1]))

# Plot each feature as a quiver plot (arrow) from the origin to the coordinate of the feature for the first two principal components (PC1 and PC2)
for i in range(loadings_df2.shape[0]):
    plt.arrow(0, 0,
              loadings_df2.iloc[i, 0] * x_max,  # Use .iloc here
              loadings_df2.iloc[i, 1] * y_max,  # Use .iloc here
              color='r', width=0.01, head_width=0.05, head_length=0.1, overhang=0.2)
    plt.text(loadings_df2.iloc[i, 0] * x_max * 1.15,  # Use .iloc here
             loadings_df2.iloc[i, 1] * y_max * 1.15,  # Use .iloc here
             loadings_df2.index[i], color='r', ha='center', va='center')

# Set limits for the plot
plt.xlim(-x_max*1.2, x_max*1.2)
plt.ylim(-y_max*1.2, y_max*1.2)

# Add labels and a title
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Biplot')

# Ensure the aspect ratio is equal
plt.gca().set_aspect('equal', adjustable='box')

# Add gridlines for better readability
plt.grid(True)

# Add horizontal and vertical lines through the origin color them red for better readability
plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)

# Show the plot
plt.show()

# ----------------------------------------
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Range of possible clusters to evaluate
range_n_clusters = list(range(2, 11))

silhouette_scores = []  # To store silhouette scores for each n_clusters

for n_clusters in range_n_clusters:
    # Initialize KMeans with n_clusters
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X_pca)  # Use PCA-reduced data
    
    # Calculate the silhouette score and append to list
    silhouette_avg = silhouette_score(X_pca, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"For n_clusters = {n_clusters}, the silhouette score is: {silhouette_avg}")

# Plotting the silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, silhouette_scores, marker='o')
plt.title('Silhouette Score for Different Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()


optimal_clusters = range_n_clusters[silhouette_scores.index(max(silhouette_scores))]
print(f"The optimal number of clusters is: {optimal_clusters}")

# ----------------------------------------
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

# Assuming you have the PCA-reduced data X_pca and cluster labels cluster_labels

# Calculate Calinski-Harabasz Index
calinski_harabasz = calinski_harabasz_score(X_pca, cluster_labels)
print(f'Calinski-Harabasz Index: {calinski_harabasz}')

# Calculate Davies-Bouldin Index
davies_bouldin = davies_bouldin_score(X_pca, cluster_labels)
print(f'Davies-Bouldin Index: {davies_bouldin}')

# ----------------------------------------
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

# Choose the optimal number of clusters from silhouette analysis
optimal_num_clusters = 6  

# Perform KMeans clustering
kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=10)
cluster_labels = kmeans.fit_predict(X_pca)

# Visualize the clusters
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', marker='o')
plt.title('Clusters in PCA-reduced Space')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
plt.show()

# Profile the clusters

data_by_artist['cluster'] = cluster_labels  # add the cluster labels to your original dataset
cluster_profile = data_by_artist.groupby('cluster')[features].mean()  # calculate the mean of each feature for each cluster
print(cluster_profile)


# ----------------------------------------
# Get centroids and plot them
centroids = kmeans.cluster_centers_
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, alpha=0.5)  # The existing plot
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, c='red')  # Centroids
plt.title('PCA-reduced Data with Cluster Centroids')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# ----------------------------------------
from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm

# Assuming silhouette_scores and cluster_labels are already computed

# Compute silhouette scores for each sample
sample_silhouette_values = silhouette_samples(X_pca, cluster_labels)

# Create a subplot with 1 row and 2 columns
fig, ax1 = plt.subplots(1, 1)
fig.set_size_inches(18, 7)

y_lower = 10
for i in range(optimal_num_clusters):
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    ith_cluster_silhouette_values.sort()
    
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    
    color = cm.nipy_spectral(float(i) / optimal_num_clusters)
    ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
    y_lower = y_upper + 10

ax1.set_title("The silhouette plot for the various clusters.")
ax1.set_xlabel("The silhouette coefficient values")
ax1.set_ylabel("Cluster label")

# The vertical line for average silhouette score of all the values
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
plt.show()

# ----------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt

# Create a subplot layout
n_rows = len(features) // 2 + len(features) % 2
fig, axes = plt.subplots(nrows=n_rows, ncols=2, figsize=(20, 5 * n_rows))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Loop over each feature to create a line plot for each
for i, feature in enumerate(features):
    # Create a new dataframe with the mean values of the feature for each cluster
    feature_means = data_by_artist.groupby('cluster')[feature].mean().reset_index()
    sns.lineplot(data=feature_means, x='cluster', y=feature, ax=axes[i], marker='o')
    axes[i].set_title(f'Mean {feature} Value Across Clusters')
    axes[i].set_xlabel('Cluster')
    axes[i].set_ylabel(f'Mean {feature}')
    
# Adjust the layout
plt.tight_layout()
plt.show()



# ----------------------------------------
# automated cluster labeling
# Calculate the mean of each feature for each cluster

cluster_means = data_by_artist.groupby('cluster')[features].mean()

# Now, Normalize the feature values within each cluster and generate labels
feature_importance = cluster_means.apply(lambda row: row / row.sum(), axis=1)

for index, row in feature_importance.iterrows():
    top_features = row.sort_values(ascending=False).head(2).index.tolist()
    label = ' / '.join(top_features)
    print(f"Cluster {index} label suggestion: {label}")

# ----------------------------------------
# Calculate the mean of each feature within each cluster
cluster_means = data_by_artist.groupby('cluster')[features].mean()

# Normalize the feature values within each cluster
feature_importance = cluster_means.apply(lambda row: row / row.sum(), axis=1)

# Suggest labels by considering the second and third most defining features
for index, row in feature_importance.iterrows():
    # Sort the features by importance and select the top ones, excluding 'tempo'
    sorted_features = row.drop('tempo').sort_values(ascending=False)
    top_features = sorted_features.head(2).index.tolist()
    label = ' / '.join(top_features)
    print(f"Cluster {index} label suggestion: {label}")

# ----------------------------------------
import pandas as pd

# Define your cluster labels based on the discussion
cluster_labels = {
    0: "Instrumentalness / Valence",
    1: "Acousticness / Danceability",
    2: "Liveness / Energy",
    3: "Energy / Danceability",
    4: "Acousticness / Instrumentalness",
    5: "Danceability / Acousticness"
}

# Map the cluster numbers to the descriptive labels
data_by_artist['cluster_label'] = data_by_artist['cluster'].map(cluster_labels)

# ----------------------------------------
cluster_distribution = data_by_artist['cluster_label'].value_counts()

# Plotting the distribution
import matplotlib.pyplot as plt

cluster_distribution.plot(kind='bar')
plt.title('Distribution of artists Across Clusters')
plt.xlabel('Cluster Label')
plt.ylabel('Number of Songs')
plt.xticks(rotation=45, ha="right")
plt.show()

# ----------------------------------------
import pandas as pd
import numpy as np

def recommend_artists(input_artists, data, num_recommendations=5):
    recommendations = {}
    
    for artist in input_artists:
        # Check if the artist is in the dataset
        if artist not in data['artists'].values:
            print(f"{artist} not found in the dataset.")
            recommendations[artist] = []
            continue
        
        # Get the cluster for the input artist
        artist_cluster = data[data['artists'] == artist]['cluster'].values
        if len(artist_cluster) > 0:
            cluster = artist_cluster[0]
            # Get other artists from the same cluster
            possible_recommendations = data[data['cluster'] == cluster]['artists'].tolist()
            # Remove the input artist from the recommendation list
            possible_recommendations = [a for a in possible_recommendations if a != artist]
            # Select a number of recommendations
            if len(possible_recommendations) < num_recommendations:
                recommendations[artist] = possible_recommendations
            else:
                recommendations[artist] = np.random.choice(possible_recommendations, num_recommendations, replace=False).tolist()
        else:
            # If the cluster is not found, add an empty list of recommendations
            recommendations[artist] = []
    
    return recommendations


# Example usage:
input_artists = ['The Weeknd', 'Metallica', 'Linkin Park', 'Jhon Lennon', 'Taylor Swift', 'Eminem']
recommendations = recommend_artists(input_artists, data_by_artist)
for input_artist, recs in recommendations.items():
    print(f"Recommendations for {input_artist}: {recs}")

# ----------------------------------------
input_artists = ['The Weeknd', 'Metallica', 'Linkin Park', 'Jhon Lennon', 'Taylor Swift', 'Eminem']

def recommend_artists(input_artists, data, num_recommendations=5):
    recommendations = {}
    
    for artist in input_artists:
        # Check if the artist is in the dataset
        if artist not in data['artists'].values:
            print(f"{artist} not found in the dataset.")
            recommendations[artist] = []
            continue
        
        # Get the cluster for the input artist
        artist_cluster = data[data['artists'] == artist]['cluster'].values
        if len(artist_cluster) > 0:
            cluster = artist_cluster[0]
            # Get other artists from the same cluster
            possible_recommendations = data[data['cluster'] == cluster]['artists'].tolist()
            # Remove the input artist from the recommendation list
            possible_recommendations = [a for a in possible_recommendations if a != artist]
            # Select a number of recommendations
            if len(possible_recommendations) < num_recommendations:
                recommendations[artist] = possible_recommendations
            else:
                recommendations[artist] = np.random.choice(possible_recommendations, num_recommendations, replace=False).tolist()
        else:
            # If the cluster is not found, add an empty list of recommendations
            recommendations[artist] = []
    
    return recommendations

# ----------------------------------------

# ----------------------------------------
