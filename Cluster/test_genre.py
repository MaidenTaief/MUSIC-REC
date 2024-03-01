import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
data_w_genres = pd.read_csv('/Users/taief/Desktop/MUSIC REC/data/.csv')


# Select features for clustering
features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
X = data_w_genres[features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=6, random_state=10)
data_w_genres['cluster'] = kmeans.fit_predict(X_scaled)

# Calculate and interpret centroids
centroids = kmeans.cluster_centers_
centroids_df = pd.DataFrame(scaler.inverse_transform(centroids), columns=features)
print(centroids_df)

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



plt.title('PCA Plot of Genres Clustered')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.savefig('pca_genres_clusters_visualization.png', dpi=300)
plt.show()
