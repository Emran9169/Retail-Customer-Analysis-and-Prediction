from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, Birch
from sklearn.metrics import silhouette_score, davies_bouldin_score
import pandas as pd
import matplotlib.pyplot as plt



def scale_features(data_frame, feature_columns):
    """
    Scales the given features of the dataframe using StandardScaler.
    
    Args:
    data_frame (pd.DataFrame): The dataframe containing the features to scale.
    feature_columns (list): List of column names to be scaled.
    
    Returns:
    pd.DataFrame: A dataframe with scaled features.
    """
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(data_frame[feature_columns])
    features_scaled_df = pd.DataFrame(features_scaled, columns=feature_columns)
    return features_scaled_df



def find_optimal_clusters_kmeans(features_scaled_df, max_k=10):
    """
    Uses the elbow method to find the optimal number of clusters for KMeans clustering.
    
    Args:
    features_scaled_df (pd.DataFrame): The dataframe containing the scaled features.
    max_k (int): Maximum number of clusters to test.
    
    Returns:
    None: Plots the elbow graph to visually inspect the optimal number of clusters.
    """
    distortions = []
    for k in range(1, max_k + 1):  # Corrected line
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features_scaled_df)
        distortions.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, max_k + 1), distortions, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal K (KMeans)')
    plt.xticks(range(1, max_k + 1))
    plt.grid(True)
    plt.show()


def perform_kmeans(features_scaled_df, n_clusters=4):
    """
    Performs KMeans clustering on the scaled features.

    Note: The 'n_clusters' parameter is determined after running 'find_optimal_clusters_kmeans'
    and determining the optimal number of clustersthrough the elbow method, manually update this
    parameter to reflect the optimal number.

    Args:
    features_scaled_df (pd.DataFrame): DataFrame containing the scaled features.
    n_clusters (int): Number of clusters to use for KMeans. Update based on elbow method analysis.

    Returns:
    array: Cluster labels assigned by KMeans.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    final_model_kmeans = kmeans.fit(features_scaled_df)
    cluster_labels_kmeans = final_model_kmeans.labels_
    return  cluster_labels_kmeans



def find_optimal_clusters_birch(features_scaled_df, range_n_clusters):
    """
    Uses silhouette analysis to find the optimal number of clusters for Birch clustering.
    
    Args:
    features_scaled_df (pd.DataFrame): The dataframe containing the scaled features.
    range_n_clusters (list): A list of the number of clusters to test.
    
    Returns:
    None: Plots silhouette scores for different numbers of clusters.
    """
    silhouette_avg_scores = []
    for n_clusters in range_n_clusters:
        birch = Birch(n_clusters=n_clusters)
        cluster_labels = birch.fit_predict(features_scaled_df)
        silhouette_avg = silhouette_score(features_scaled_df, cluster_labels)
        silhouette_avg_scores.append(silhouette_avg)
       
    
    plt.figure(figsize=(10, 5))
    plt.plot(range_n_clusters, silhouette_avg_scores, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Average Silhouette Score')
    plt.title('Silhouette Analysis for Optimal K (Birch)')
    plt.grid(True)
    plt.show()


def perform_birch(features_scaled_df, n_clusters=3):
    """
    Performs Birch clustering on the scaled features.

    Note: The 'n_clusters' parameter is initially set to a default value (e.g., 5).
    After running 'find_optimal_clusters_birch' and determining the optimal number of clusters
    through silhouette analysis, manually update this parameter to reflect the optimal number.

    Args:
    features_scaled_df (pd.DataFrame): DataFrame containing the scaled features.
    n_clusters (int): Number of clusters to use for Birch. Update based on silhouette analysis.

    Returns:
    array: Cluster labels assigned by Birch.
    """
    birch = Birch(n_clusters=n_clusters, threshold=0.05, branching_factor=50)
    final_model = birch.fit(features_scaled_df)
    cluster_labels_birch = final_model.labels_
    return cluster_labels_birch


def evaluate_clustering(features_scaled, cluster_labels):
    """
    Evaluates the clustering performance using silhouette and Davies-Bouldin scores.
    
    Args:
    features_scaled (np.array): Scaled feature array used for clustering.
    cluster_labels (array): Cluster labels.
    
    Prints:
    Silhouette Score and Davies-Bouldin Score.
    """
    silhouette_avg = silhouette_score(features_scaled, cluster_labels)
    davies_bouldin_avg = davies_bouldin_score(features_scaled, cluster_labels)
    print(f"Silhouette Score: {silhouette_avg}")
    print(f"Davies-Bouldin Score: {davies_bouldin_avg}")
