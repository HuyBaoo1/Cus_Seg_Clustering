import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score


def find_elbow(K: range, inertias: list) -> int:
    x = np.array(list(K))
    y = np.array(inertias)
    x_norm = (x - min(x)) / (max(x) - min(x))
    y_norm = (y - min(y)) / (max(y) - min(y))
    
    angles = []
    for i in range(1, len(x_norm)-1):
        # Vectors between points
        v1 = np.array([x_norm[i] - x_norm[i-1], y_norm[i] - y_norm[i-1]])
        v2 = np.array([x_norm[i+1] - x_norm[i], y_norm[i+1] - y_norm[i]])
        
        # Normalize vectors
        v1 = v1 / np.sqrt(np.sum(v1**2))
        v2 = v2 / np.sqrt(np.sum(v2**2))
        
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        angles.append(angle)
  
    elbow_idx = np.argmax(angles) + 1
    return K[elbow_idx]


def memory_safe_silhouette(X: np.ndarray, labels: np.ndarray, sample_size: int = 10000):
    n_samples = X.shape[0]
    try:
        if n_samples * n_samples > 3e8:
            raise MemoryError
        return silhouette_score(X, labels)
    except (MemoryError, ValueError):
        # fallback to sampling
        n = min(n_samples, sample_size)
        idx = np.random.choice(n_samples, size=n, replace=False)
        print(f"Silhouette: dataset too large, computing on a random sample of {n} rows")
        return silhouette_score(X[idx], labels[idx])


def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV file containing customer RFM scores."""
    df = pd.read_csv(file_path)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def visualize_rfm_distribution(df: pd.DataFrame):
    """Plot violin distributions for Recency, Frequency, and Monetary scores."""
    rfm_scores = df[['recency_score', 'frequency_score', 'monetary_score']]

    sns.set(style="whitegrid", palette="pastel", font_scale=1.1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    score_names = ['Recency', 'Frequency', 'Monetary']

    for i, col in enumerate(rfm_scores.columns):
        sns.violinplot(y=rfm_scores[col], ax=axes[i], color='lightcoral', inner='quartile')
        axes[i].set_title(f'{score_names[i]} Score Distribution')
        axes[i].set_ylabel('Score')
        axes[i].set_ylim(0.5, 5.5)

    plt.tight_layout()
    plt.show()


def kmeans_clustering(rfm_scores: pd.DataFrame):
    rfm_scores = rfm_scores.dropna().copy()
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_scores)

    inertia = []
    silhouette_scores = []
    K = range(2, 11)
    print("\nFinding optimal number of clusters...")
    for k in K:
        print(f"Testing k={k}...", end=" ")
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=4096)
        labels = kmeans.fit_predict(rfm_scaled)
        inertia.append(kmeans.inertia_)
        sil = memory_safe_silhouette(rfm_scaled, labels)
        silhouette_scores.append(sil)
        print(f"silhouette={sil:.3f}")

    # Plot both inertia and silhouette scores
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Inertia plot
    ax1.plot(K, inertia, 'bo-', linewidth=2)
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    ax1.grid(True)
    
    # Silhouette plot
    ax2.plot(K, silhouette_scores, 'ro-', linewidth=2)
    ax2.set_xlabel('Number of clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Scores')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

    # Find best K using elbow method
    elbow_k = find_elbow(K, inertia)
    best_sil_k = K[np.argmax(silhouette_scores)]

    if abs(best_sil_k - elbow_k) <= 1:
        sil_score_elbow = silhouette_scores[list(K).index(elbow_k)]
        sil_score_best = silhouette_scores[list(K).index(best_sil_k)]
        if sil_score_best > sil_score_elbow * 1.05:
            best_k = best_sil_k
        else:
            best_k = elbow_k
    else:
        best_k = elbow_k

    print(f"\nAutomatic cluster selection:")
    print(f"Elbow method suggests k={elbow_k}")
    print(f"Best silhouette score at k={best_sil_k}")
    print(f"Chosen K = {best_k}")

    best_kmeans = MiniBatchKMeans(n_clusters=best_k, random_state=42, n_init=10)
    print("Fitting KMeans...")
    import time
    t0 = time.time()
    labels = best_kmeans.fit_predict(rfm_scaled)
    t1 = time.time()
    rfm_scores.loc[:, 'KMeans_Cluster'] = labels

    silhouette = memory_safe_silhouette(rfm_scaled, labels)
    print(f"K-Means Silhouette Score: {silhouette:.3f}")
    print(f"KMeans fit finished in {t1 - t0:.2f} seconds")

    return rfm_scores, best_kmeans, silhouette


def run_pipeline(file_path: str):
    df = load_data(file_path)
    visualize_rfm_distribution(df)

    print("\nK-Means Clustering...")
    df_kmeans, kmeans_model, kmeans_score = kmeans_clustering(df[['recency_score', 'frequency_score', 'monetary_score']])

    # Export results
    output_path = "rfm_clustered_results.csv"
    df_kmeans.to_csv(output_path, index=False)
    print(f"\nClustered data saved to {output_path}")


if __name__ == "__main__":
    csv_path = "Data/factcustomermonthlysnapshot.csv"
    run_pipeline(csv_path)
