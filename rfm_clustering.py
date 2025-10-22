"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances
import time


def find_elbow(K: range, inertias: list) -> int:
    x = np.array(list(K))
    y = np.array(inertias)
    x_norm = (x - min(x)) / (max(x) - min(x))
    y_norm = (y - min(y)) / (max(y) - min(y))
    angles = []
    for i in range(1, len(x_norm) - 1):
        v1 = np.array([x_norm[i] - x_norm[i - 1], y_norm[i] - y_norm[i - 1]])
        v2 = np.array([x_norm[i + 1] - x_norm[i], y_norm[i + 1] - y_norm[i]])
        v1 = v1 / np.sqrt(np.sum(v1 ** 2))
        v2 = v2 / np.sqrt(np.sum(v2 ** 2))
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        angles.append(angle)
    elbow_idx = np.argmax(angles) + 1
    return K[elbow_idx]


def centroid_silhouette_proxy(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(labels)
    centers = np.asarray(centers, dtype=np.float32)

    D = pairwise_distances(X, centers, metric='euclidean')
    own = D[np.arange(D.shape[0]), labels]
    D[np.arange(D.shape[0]), labels] = np.inf
    nearest = D.min(axis=1)
    denom = np.maximum(own, nearest)
    s = np.zeros_like(denom)
    mask = denom > 0
    s[mask] = (nearest[mask] - own[mask]) / denom[mask]
    return float(np.nanmean(s))


def visualize_rfm_distribution(df: pd.DataFrame):
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


def kmeans_clustering(df: pd.DataFrame):

    rfm_scores = df[['recency_score', 'frequency_score', 'monetary_score']].dropna().copy()

    if rfm_scores.nunique().sum() <= 3:
        print("⚠️ All RFM scores are identical — skipping clustering.")
        df['KMeans_Cluster'] = 0
        return df, None, None
    

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_scores.values.astype(np.float32))

    best_k = 5

    print(f"\nUsing k={best_k}")
    best_kmeans = MiniBatchKMeans(
        n_clusters=best_k,
        random_state=42,
        batch_size=1024,
        n_init=5
    )

    labels = best_kmeans.fit_predict(rfm_scaled)

    # Align with original df
    df = df.loc[rfm_scores.index].copy()
    df['KMeans_Cluster'] = labels.astype(int)

    return df, best_kmeans, scaler



def create_segment_labels(df_kmeans, snapshot_date=None):
    #(1=Low → 3=High value)
    df_kmeans['rfm_key'] = (df_kmeans['recency_score'].astype(str) + '_' + 
                           df_kmeans['frequency_score'].astype(str) + '_' + 
                           df_kmeans['monetary_score'].astype(str))
    
    # Calculate value score for each unique RFM combination
    rfm_value_scores = {}
    for rfm_key in df_kmeans['rfm_key'].unique():
        r, f, m = map(float, rfm_key.split('_'))
        value_score = f * 0.4 + m * 0.4 + (6 - r) * 0.2
        rfm_value_scores[rfm_key] = value_score
    
    sorted_rfm = sorted(rfm_value_scores.items(), key=lambda x: x[1])
    segment_thirds = {k: (i // (len(sorted_rfm) // 3 + 1)) + 1 
                     for i, (k, _) in enumerate(sorted_rfm)}
    
    df_kmeans['segmentkey'] = df_kmeans['rfm_key'].map(segment_thirds)

    cluster_stats = []
    for cluster in sorted(df_kmeans['KMeans_Cluster'].unique()):
        subset = df_kmeans[df_kmeans['KMeans_Cluster'] == cluster]
        avg_r = subset['recency_score'].mean()
        avg_f = subset['frequency_score'].mean()
        avg_m = subset['monetary_score'].mean()
        
        # Calculate cluster average (weighted towards F and M)
        value_score = (
            avg_f * 0.4 +avg_m * 0.4 +(6 - avg_r) * 0.2
        )
        # segmentkey is relative to that single date. The same numeric RFM combination or same segmentkey may mean different things between dates (unless the clusters and scores happen to match).
        stats = {
            'cluster': cluster,
            'AvgRecencyScore': avg_r,
            'AvgFrequencyScore': avg_f,
            'AvgMonetaryScore': avg_m,
            'CustomerCount': len(subset),
            'value_score': value_score
        }
        if snapshot_date is not None:
            stats['snapshotdatekey'] = snapshot_date
        cluster_stats.append(stats)

    dim_segment = pd.DataFrame(cluster_stats)
    dim_segment = dim_segment.sort_values('value_score').reset_index(drop=True)
    dim_segment['segmentkey'] = dim_segment.index + 1
    
    mapping = dict(zip(dim_segment['cluster'], dim_segment['segmentkey']))
    df_kmeans['KMeans_Cluster'] = df_kmeans['KMeans_Cluster'].astype(int)
    df_kmeans['segmentkey'] = df_kmeans['KMeans_Cluster'].map(mapping).fillna(0).astype('int32')
    dim_segment = dim_segment.drop('value_score', axis=1)
    
    return df_kmeans, dim_segment


def run_pipeline(file_path: str):
    print("Loading data...")
    dtypes = {
        'customerkey': 'int32',
        'recency_score': 'float32',
        'frequency_score': 'float32',
        'monetary_score': 'float32',
        'snapshotdatekey': 'int32',
        'demographickey': 'int32',
        'geographickey': 'int32'
    }
    df = pd.read_csv(file_path, dtype=dtypes)

    if len(df) <= 1_000_000:
        visualize_rfm_distribution(df)

    all_results = []
    all_segments = []

    for date_key in sorted(df['snapshotdatekey'].unique()):
        print(f"\nProcessing snapshot date: {date_key}")
        df_snapshot = df[df['snapshotdatekey'] == date_key].copy()
        
        df_kmeans, model, scaler = kmeans_clustering(df_snapshot)
        df_kmeans, dim_segment = create_segment_labels(df_kmeans, date_key)
        all_segments.append(dim_segment)
        
        # Keep only necessary columns for results
        result = df_kmeans[['customerkey', 'segmentkey', 'snapshotdatekey']].copy()
        all_results.append(result)
        
        # Print segment summary for this date
        print(f"\nSegment summary for date {date_key}:")
        summary = dim_segment[['segmentkey', 'CustomerCount', 'AvgRecencyScore', 
                             'AvgFrequencyScore', 'AvgMonetaryScore']]
        print(summary.to_string())

    # Combine all results
    print("\n Results...")
    df_results = pd.concat(all_results, ignore_index=True)
    dim_segment_all = pd.concat(all_segments, ignore_index=True)

    # Diagnostic: ensure segmentkey exists and types align
    print("\ndf_results columns:", df_results.columns.tolist())
    if 'segmentkey' not in df_results.columns:
        print("Warning: 'segmentkey' column missing in df_results — adding default 0")
        df_results['segmentkey'] = 0

    try:
        df_results['customerkey'] = df_results['customerkey'].astype(df['customerkey'].dtype)
        df_results['snapshotdatekey'] = df_results['snapshotdatekey'].astype(df['snapshotdatekey'].dtype)
    except Exception:
        df_results['customerkey'] = df_results['customerkey'].astype('int32')
        df_results['snapshotdatekey'] = df_results['snapshotdatekey'].astype('int32')

    print("Original df columns:", df.columns.tolist())
    print("Original df dtypes:\n", df.dtypes)
    df_final = df.merge(df_results[['customerkey', 'segmentkey', 'snapshotdatekey']], 
                       on=['customerkey', 'snapshotdatekey'], 
                       how='left')
    print("df_final columns after merge:", df_final.columns.tolist())
    print("df_final sample columns dtypes:\n", df_final.dtypes.head(10))

    if 'segmentkey_y' in df_final.columns:
        df_final['segmentkey'] = df_final['segmentkey_y'].fillna(df_final.get('segmentkey_x', 0))
    elif 'segmentkey_x' in df_final.columns:
        df_final['segmentkey'] = df_final['segmentkey_x'].fillna(0)
    else:
        df_final['segmentkey'] = 0

    df_final['segmentkey'] = df_final['segmentkey'].astype('int32')
    df_final = df_final.drop([c for c in ['segmentkey_x', 'segmentkey_y'] if c in df_final.columns], axis=1)

    print("\nSaving results...")

    
    if 'segmentkey' in df_final.columns:
        if 'recency_score' in df_final.columns:
            cols = df_final.columns.tolist()
            cols.remove('segmentkey')
            rec_idx = cols.index('recency_score')
            cols.insert(rec_idx, 'segmentkey')
            df_final = df_final[cols]
        else:
            cols = df_final.columns.tolist()
            cols.remove('segmentkey')
            df_final = df_final[['segmentkey'] + cols]
    df_final.to_csv("factcustomermonthlysnapshot_updated.csv", index=False)
    dim_segment_all.to_csv("DimSegment.csv", index=False)

    print("\nUpdated 'factcustomermonthlysnapshot_updated.csv' created")
    print("Dimension table 'DimSegment.csv' created")
    

if __name__ == "__main__":
    csv_path = "Data/factcustomermonthlysnapshot.csv"
    run_pipeline(csv_path)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans


def visualize_rfm_distribution(df: pd.DataFrame):
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


def global_kmeans_training(df: pd.DataFrame, n_clusters: int = 5):
    """Train one global KMeans model using all RFM data."""
    rfm = df[['recency_score', 'frequency_score', 'monetary_score']].dropna().copy()
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=1024,
        n_init=5
    )
    kmeans.fit(rfm_scaled)
    return kmeans, scaler

def assign_clusters(df: pd.DataFrame, kmeans, scaler):
    rfm = df[['recency_score', 'frequency_score', 'monetary_score']].copy()
    rfm_scaled = scaler.transform(rfm)
    df['KMeans_Cluster'] = kmeans.predict(rfm_scaled)
    return df


def create_global_segments(df: pd.DataFrame):
    cluster_stats = []
    for cluster in sorted(df['KMeans_Cluster'].unique()):
        subset = df[df['KMeans_Cluster'] == cluster]
        avg_r = subset['recency_score'].mean()
        avg_f = subset['frequency_score'].mean()
        avg_m = subset['monetary_score'].mean()

        value_score = avg_f * 0.4 + avg_m * 0.4 + (6 - avg_r) * 0.2

        stats = {
            'cluster': cluster,
            'AvgRecencyScore': round(avg_r, 2),
            'AvgFrequencyScore': round(avg_f, 2),
            'AvgMonetaryScore': round(avg_m, 2),
            'CustomerCount': len(subset),
            'value_score': value_score
        }
        cluster_stats.append(stats)

    dim_segment = pd.DataFrame(cluster_stats)
    dim_segment = dim_segment.sort_values('value_score', ascending=True).reset_index(drop=True)
    dim_segment['segmentkey'] = dim_segment.index + 1

    mapping = dict(zip(dim_segment['cluster'], dim_segment['segmentkey']))
    df['segmentkey'] = df['KMeans_Cluster'].map(mapping).astype('int32')
    dim_segment = dim_segment.drop('value_score', axis=1)

    return df, dim_segment


def run_pipeline(file_path: str):
    print("Loading data...")
    dtypes = {
        'customerkey': 'int32',
        'recency_score': 'float32',
        'frequency_score': 'float32',
        'monetary_score': 'float32',
        'snapshotdatekey': 'int32',
        'demographickey': 'int32',
        'geographickey': 'int32'
    }
    df = pd.read_csv(file_path, dtype=dtypes)
    visualize_rfm_distribution(df)

    kmeans, scaler = global_kmeans_training(df, n_clusters=5)
    df_clustered = assign_clusters(df, kmeans, scaler)
    df_clustered, dim_segment = create_global_segments(df_clustered)


    df_clustered.to_csv("factcustomermonthlysnapshot_updated.csv", index=False)
    dim_segment.to_csv("DimSegment.csv", index=False)
    print("\nSaved to factcustomermonthlysnapshot_updated.csv")
    print("DimSegment.csv")


if __name__ == "__main__":
    csv_path = "Data/factcustomermonthlysnapshot.csv"
    run_pipeline(csv_path)
