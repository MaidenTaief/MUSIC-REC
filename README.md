# Music Recommendation System Project

## Overview
This project is to create a robust Music Recommendation System. Leveraging the Dataset I collected from Kaggle. 
This project involves the following key steps:


## Datasets
The work so far has involved two primary datasets:

- **`data_by_artist.csv`**: Contains detailed audio features for music by various artists.
- **`data_w_genres.csv`**: Aggregates the audio features and associates them with music genres.

## Data Exploration and Preprocessing
The initial phase of the project involved deep exploratory data analysis (EDA) and preprocessing to understand the underlying patterns and structure of the datasets. Key steps included:

- Loading the data and performing sanity checks for data consistency.
- Handling missing values and duplicates to ensure data quality.
- Generating a correlation matrix to understand the relationships between different audio features.

## Feature Engineering
I have applied feature engineering techniques to enhance the datasets:

- Creating interaction features that capture the combined effects of correlated features.
- Standardizing features to prepare the data for clustering algorithms.

## Clustering
K-Means clustering was employed to segment artists and genres into distinct groups based on their audio features. This step was crucial for:

- Identifying groups of similar tracks or genres.
- Providing a basis for feature integration into the recommendation system.

## PCA Visualization
To visualize the high-dimensional clustering results, I applied Principal Component Analysis (PCA) for dimensionality reduction, allowing me to plot the data in a two-dimensional space.

## Next Steps
The project is ongoing, and future steps will include:

- Integration of additional datasets to enrich the feature space.
- Implementation of neural networks and graph embedding techniques for feature matrix optimization.
- Utilization of Language Model (LLM) for lyrical analysis and understanding.
- Comparison with the Million Song Dataset (MSD) for benchmarking and validation.
- Development of the recommendation algorithm using insights from the EDA and clustering results.

