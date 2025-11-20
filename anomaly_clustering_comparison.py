
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import yaml
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

# Define anomaly signatures from the problem description
UPLINK_INTERFERENCE_KPIS = [
    'interference_mean', 'interference_std', 'interference_median', 'interference_p95',
    'ul_sinr_pusch_mean', 'ul_sinr_pusch_std', 'ul_sinr_pusch_median', 'ul_sinr_pusch_p95',
    'ul_harq_failure_rate', 'erab_abnormal_drops'
]

MASS_EVENT_KPIS = [
    'rrc_connection_attempts', 'max_active_users_dl', 'prb_utilization_dl', 'rrc_success_rate',
    'pdcch_utilization_mean', 'pdcch_utilization_median', 'pdcch_utilization_p95'
]

# KPIs indicative of a sleeping cell
SLEEPING_CELL_KPIS = [
    'rach_success_rate', 'rrc_success_rate', 'erab_success_rate',
    'data_volume_dl', 'data_volume_ul', 'active_users_dl', 'active_users_ul',
    'handover_success_rate'
]

def load_4g_kpis_from_yaml(yaml_path='kpi_config_dbscan.yaml'):
    """Loads a dictionary of 4G KPI names to titles from the YAML configuration file."""
    try:
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
            return {kpi['name']: kpi['title'] for kpi in config.get('kpis_4g', [])}
    except FileNotFoundError:
        print(f"Error: The file {yaml_path} was not found.")
        return {}
    except Exception as e:
        print(f"An error occurred while reading the YAML file: {e}")
        return {}

def generate_synthetic_data(kpi_list, n_points_per_cluster=300, n_noise=200):
    """
    Generates realistic synthetic dataset with overlapping clusters, varying severity,
    and correlated KPI behaviors mimicking real network anomalies.
    """
    if not kpi_list:
        print("KPI list is empty. Cannot generate data.")
        return None, None, None

    n_kpis = len(kpi_list)
    total_points = 3 * n_points_per_cluster + n_noise
    print(f"Generating realistic overlapping data with {total_points} points and {n_kpis} KPIs (using ALL KPIs from config).")

    # Initialize with realistic baseline: most KPIs near normal with small variance
    data = np.random.gamma(shape=2.0, scale=0.05, size=(total_points, n_kpis))
    labels = np.zeros(total_points, dtype=int)

    # Map KPIs to their indices for efficient lookup
    kpi_to_idx = {kpi: idx for idx, kpi in enumerate(kpi_list)}

    def get_kpi_indices(kpi_names):
        """Get indices for KPIs that exist in the full list."""
        return [kpi_to_idx[kpi] for kpi in kpi_names if kpi in kpi_to_idx]

    # Get indices for each anomaly signature
    uplink_indices = get_kpi_indices(UPLINK_INTERFERENCE_KPIS)
    mass_event_indices = get_kpi_indices(MASS_EVENT_KPIS)
    sleeping_indices = get_kpi_indices(SLEEPING_CELL_KPIS)

    # Create correlated secondary effects (realistic network behavior)
    def add_secondary_effects(start, end, primary_indices, correlation_strength=0.3):
        """Add correlated changes to related KPIs (realistic cascading effects)."""
        for i in range(start, end):
            if np.random.rand() < correlation_strength:
                # Randomly select 3-5 non-primary KPIs to show mild effects
                all_other_indices = [idx for idx in range(n_kpis) if idx not in primary_indices]
                affected = np.random.choice(all_other_indices, size=min(5, len(all_other_indices)), replace=False)
                data[i, affected] += np.random.gamma(shape=1.5, scale=0.15, size=len(affected))

    # --- Cluster A: Uplink Interference ---
    # Characteristics: High interference, degraded SINR, increased HARQ failures
    # Severity varies: 30% severe, 40% moderate, 30% mild (overlapping with normal)
    start, end = 0, n_points_per_cluster
    labels[start:end] = 0

    severity_split = [int(0.3 * n_points_per_cluster), int(0.7 * n_points_per_cluster)]

    # Severe cases (high reconstruction error)
    for i in range(start, start + severity_split[0]):
        data[i, uplink_indices] = np.random.gamma(shape=3.0, scale=0.35, size=len(uplink_indices))
        data[i, uplink_indices] += np.random.normal(0.7, 0.2, size=len(uplink_indices))

    # Moderate cases (medium reconstruction error - overlaps with mild and normal)
    for i in range(start + severity_split[0], start + severity_split[1]):
        data[i, uplink_indices] = np.random.gamma(shape=2.5, scale=0.22, size=len(uplink_indices))
        data[i, uplink_indices] += np.random.normal(0.4, 0.18, size=len(uplink_indices))

    # Mild cases (low reconstruction error - significant overlap)
    for i in range(start + severity_split[1], end):
        data[i, uplink_indices] = np.random.gamma(shape=2.0, scale=0.15, size=len(uplink_indices))
        data[i, uplink_indices] += np.random.normal(0.2, 0.12, size=len(uplink_indices))

    add_secondary_effects(start, end, uplink_indices, correlation_strength=0.4)

    # --- Cluster B: Mass Event ---
    # Characteristics: High connection attempts, resource congestion, control channel saturation
    # More variable severity distribution
    start, end = n_points_per_cluster, 2 * n_points_per_cluster
    labels[start:end] = 1

    # Severe: sudden mass event (20%)
    severe_count = int(0.2 * n_points_per_cluster)
    for i in range(start, start + severe_count):
        data[i, mass_event_indices] = np.random.gamma(shape=3.5, scale=0.3, size=len(mass_event_indices))
        data[i, mass_event_indices] += np.random.normal(0.8, 0.15, size=len(mass_event_indices))

    # Moderate: building congestion (50%)
    moderate_count = int(0.5 * n_points_per_cluster)
    for i in range(start + severe_count, start + severe_count + moderate_count):
        data[i, mass_event_indices] = np.random.gamma(shape=2.8, scale=0.2, size=len(mass_event_indices))
        data[i, mass_event_indices] += np.random.normal(0.45, 0.2, size=len(mass_event_indices))

    # Mild: early stage or resolving (30%)
    for i in range(start + severe_count + moderate_count, end):
        data[i, mass_event_indices] = np.random.gamma(shape=2.2, scale=0.12, size=len(mass_event_indices))
        data[i, mass_event_indices] += np.random.normal(0.25, 0.15, size=len(mass_event_indices))

    add_secondary_effects(start, end, mass_event_indices, correlation_strength=0.5)

    # --- Cluster C: Sleeping Cell ---
    # Characteristics: Low success rates, minimal traffic, poor handovers
    # Wide severity range (hardest to detect)
    start, end = 2 * n_points_per_cluster, 3 * n_points_per_cluster
    labels[start:end] = 2

    # Critical failure (25%)
    critical_count = int(0.25 * n_points_per_cluster)
    for i in range(start, start + critical_count):
        data[i, sleeping_indices] = np.random.gamma(shape=4.0, scale=0.32, size=len(sleeping_indices))
        data[i, sleeping_indices] += np.random.normal(0.9, 0.18, size=len(sleeping_indices))

    # Degraded performance (45%)
    degraded_count = int(0.45 * n_points_per_cluster)
    for i in range(start + critical_count, start + critical_count + degraded_count):
        data[i, sleeping_indices] = np.random.gamma(shape=2.6, scale=0.18, size=len(sleeping_indices))
        data[i, sleeping_indices] += np.random.normal(0.5, 0.22, size=len(sleeping_indices))

    # Intermittent issues (30% - very hard to distinguish)
    for i in range(start + critical_count + degraded_count, end):
        data[i, sleeping_indices] = np.random.gamma(shape=2.0, scale=0.1, size=len(sleeping_indices))
        data[i, sleeping_indices] += np.random.normal(0.3, 0.18, size=len(sleeping_indices))

    add_secondary_effects(start, end, sleeping_indices, correlation_strength=0.35)

    # --- Noise Points: True Outliers and Ambiguous Cases ---
    # Mix of: random multi-KPI anomalies, edge cases, transient glitches
    start, end = 3 * n_points_per_cluster, total_points
    labels[start:end] = 3

    for i in range(start, end):
        # Random multi-dimensional anomalies
        n_affected = np.random.randint(2, 8)  # 2-7 KPIs affected randomly
        affected_kpis = np.random.choice(n_kpis, size=n_affected, replace=False)
        data[i, affected_kpis] = np.random.gamma(shape=2.5, scale=0.25, size=n_affected)
        data[i, affected_kpis] += np.random.uniform(0.1, 1.0, size=n_affected)

    # Add realistic measurement noise to ALL data
    noise_factor = np.random.normal(1.0, 0.08, size=(total_points, n_kpis))
    data = data * noise_factor

    data_unscaled = np.clip(data, 0, 2.5)  # Realistic max ceiling

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_unscaled)
    df_scaled = pd.DataFrame(data_scaled, columns=kpi_list)

    print(f"Data generation complete. Cluster overlap intentionally high for realism.")
    print(f"  - Cluster 0 (Uplink): {n_points_per_cluster} samples (30% severe, 40% moderate, 30% mild)")
    print(f"  - Cluster 1 (Mass Event): {n_points_per_cluster} samples (20% severe, 50% moderate, 30% mild)")
    print(f"  - Cluster 2 (Sleeping): {n_points_per_cluster} samples (25% critical, 45% degraded, 30% intermittent)")
    print(f"  - Noise: {n_noise} samples (random outliers)")

    return df_scaled, data_unscaled, np.array(labels)

def run_clustering_analysis(X, y_true):
    """
    Applies K-means, DBSCAN (Euclidean & Cosine), and HDBSCAN (Euclidean & Cosine).
    Compares performance across distance metrics.
    """
    print("\n" + "="*80)
    print("Starting Comprehensive Clustering Analysis with Distance Metric Comparison")
    print("="*80 + "\n")
    results = {}

    print("\n--- 1. K-means Analysis (uses Euclidean by default) ---\n")
    for k in [3, 4]:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        y_pred = kmeans.fit_predict(X)
        results[f'kmeans_{k}'] = y_pred
        ari = adjusted_rand_score(y_true, y_pred)
        silhouette = silhouette_score(X, y_pred)
        print(f"K-means (k={k}): ARI={ari:.4f}, Silhouette={silhouette:.4f}")

    # DBSCAN with BOTH distance metrics
    metrics_to_test = ['euclidean', 'cosine']

    for metric in metrics_to_test:
        print(f"\n--- 2. DBSCAN Analysis (Metric: {metric.upper()}) ---\n")

        # Calculate adaptive epsilon values based on k-nearest neighbors
        distances = pairwise_distances(X, metric=metric)
        # Get distance to 10th nearest neighbor (min_samples=10)
        k_distances = np.sort(distances, axis=1)[:, 10]
        k_distances_sorted = np.sort(k_distances)

        # Use elbow method percentiles for epsilon selection
        eps_values = {
            'small': np.percentile(k_distances_sorted, 25),
            'medium': np.percentile(k_distances_sorted, 50),
            'large': np.percentile(k_distances_sorted, 75)
        }

        print(f"Auto-calculated Epsilon values for {metric}:")
        print(f"  Small (P25):  ε = {eps_values['small']:.4f}")
        print(f"  Medium (P50): ε = {eps_values['medium']:.4f}")
        print(f"  Large (P75):  ε = {eps_values['large']:.4f}\n")

        for eps_name, eps in eps_values.items():
            dbscan = DBSCAN(eps=eps, min_samples=10, metric=metric)
            y_pred = dbscan.fit_predict(X)
            results[f'dbscan_{metric}_{eps_name}'] = y_pred

            n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
            n_noise = np.sum(y_pred == -1)
            ari = adjusted_rand_score(y_true, y_pred)

            # Calculate silhouette only if we have >1 cluster and not all noise
            if n_clusters > 1 and n_noise < len(X):
                try:
                    sil = silhouette_score(X, y_pred, metric=metric)
                    print(f"  DBSCAN-{metric} (ε={eps:.4f} '{eps_name}'): "
                          f"Clusters={n_clusters}, Noise={n_noise} ({n_noise/len(X):.1%}), "
                          f"ARI={ari:.4f}, Silhouette={sil:.4f}")
                except:
                    print(f"  DBSCAN-{metric} (ε={eps:.4f} '{eps_name}'): "
                          f"Clusters={n_clusters}, Noise={n_noise} ({n_noise/len(X):.1%}), "
                          f"ARI={ari:.4f}, Silhouette=N/A")
            else:
                print(f"  DBSCAN-{metric} (ε={eps:.4f} '{eps_name}'): "
                      f"Clusters={n_clusters}, Noise={n_noise} ({n_noise/len(X):.1%}), "
                      f"ARI={ari:.4f}")

    # HDBSCAN with BOTH distance metrics
    print(f"\n--- 3. HDBSCAN Analysis (comparing distance metrics) ---\n")

    for metric in metrics_to_test:
        algorithm = 'generic' if metric == 'cosine' else 'best'
        hdb = hdbscan.HDBSCAN(min_cluster_size=15, metric=metric, algorithm=algorithm)
        y_pred = hdb.fit_predict(X)
        results[f'hdbscan_{metric}'] = y_pred

        n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
        n_noise = np.sum(y_pred == -1)
        ari = adjusted_rand_score(y_true, y_pred)

        if n_clusters > 1 and n_noise < len(X):
            try:
                sil = silhouette_score(X, y_pred, metric=metric)
                print(f"  HDBSCAN-{metric}: Clusters={n_clusters}, Noise={n_noise} ({n_noise/len(X):.1%}), "
                      f"ARI={ari:.4f}, Silhouette={sil:.4f}")
            except:
                print(f"  HDBSCAN-{metric}: Clusters={n_clusters}, Noise={n_noise} ({n_noise/len(X):.1%}), "
                      f"ARI={ari:.4f}, Silhouette=N/A")
        else:
            print(f"  HDBSCAN-{metric}: Clusters={n_clusters}, Noise={n_noise} ({n_noise/len(X):.1%}), "
                  f"ARI={ari:.4f}")

    # Summary comparison
    print("\n" + "="*80)
    print("DISTANCE METRIC COMPARISON SUMMARY")
    print("="*80)
    print("\nBest performing configurations by ARI score:")

    # Find best performer per algorithm type
    ari_scores = {}
    for name, labels in results.items():
        ari_scores[name] = adjusted_rand_score(y_true, labels)

    sorted_results = sorted(ari_scores.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 5 Overall:")
    for i, (name, score) in enumerate(sorted_results[:5], 1):
        print(f"  {i}. {name:30s} ARI = {score:.4f}")

    return results

def visualize_results(X, y_true, clustering_results):
    """
    Generates comprehensive visualizations comparing clustering results.
    Shows ground truth + best performers from each algorithm family.
    """
    print("\nGenerating comprehensive scatter plot visualizations...")
    cluster_map_true = {0: 'Uplink Interference', 1: 'Mass Event', 2: 'Sleeping Cell', 3: 'Noise'}
    y_true_named = [cluster_map_true[l] for l in y_true]
    custom_palette = {name: color for name, color in zip(cluster_map_true.values(), sns.color_palette("viridis", 4))}
    custom_palette['Noise'] = (0.5, 0.5, 0.5)

    # Select best representatives from each algorithm family
    ari_scores = {name: adjusted_rand_score(y_true, labels) for name, labels in clustering_results.items()}

    # Pick best from each family
    best_algorithms = {'ground_truth': ('Ground Truth', None)}

    # K-means
    kmeans_results = {k: v for k, v in ari_scores.items() if k.startswith('kmeans')}
    if kmeans_results:
        best_kmeans = max(kmeans_results, key=kmeans_results.get)
        best_algorithms[best_kmeans] = (f"K-means (best: k={best_kmeans.split('_')[1]}, ARI={ari_scores[best_kmeans]:.3f})", ari_scores[best_kmeans])

    # DBSCAN Euclidean
    dbscan_euc = {k: v for k, v in ari_scores.items() if k.startswith('dbscan_euclidean')}
    if dbscan_euc:
        best_euc = max(dbscan_euc, key=dbscan_euc.get)
        eps_type = best_euc.split('_')[-1]
        best_algorithms[best_euc] = (f"DBSCAN-Euclidean (best: {eps_type} ε, ARI={ari_scores[best_euc]:.3f})", ari_scores[best_euc])

    # DBSCAN Cosine
    dbscan_cos = {k: v for k, v in ari_scores.items() if k.startswith('dbscan_cosine')}
    if dbscan_cos:
        best_cos = max(dbscan_cos, key=dbscan_cos.get)
        eps_type = best_cos.split('_')[-1]
        best_algorithms[best_cos] = (f"DBSCAN-Cosine (best: {eps_type} ε, ARI={ari_scores[best_cos]:.3f})", ari_scores[best_cos])

    # HDBSCAN Euclidean
    if 'hdbscan_euclidean' in ari_scores:
        best_algorithms['hdbscan_euclidean'] = (f"HDBSCAN-Euclidean (ARI={ari_scores['hdbscan_euclidean']:.3f})", ari_scores['hdbscan_euclidean'])

    # HDBSCAN Cosine
    if 'hdbscan_cosine' in ari_scores:
        best_algorithms['hdbscan_cosine'] = (f"HDBSCAN-Cosine (ARI={ari_scores['hdbscan_cosine']:.3f})", ari_scores['hdbscan_cosine'])

    plot_keys = list(best_algorithms.keys())
    n_plots = len(plot_keys)

    reducers = {'PCA': PCA(n_components=2, random_state=42), 't-SNE': TSNE(n_components=2, random_state=42, perplexity=50, max_iter=1000)}

    for reducer_name, reducer in reducers.items():
        print(f"  Applying {reducer_name} dimensionality reduction...")
        X_2d = reducer.fit_transform(X)

        # Create grid layout (2 rows, enough columns)
        n_cols = min(4, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 7*n_rows))
        fig.suptitle(f'Clustering Comparison: Best Algorithms per Family ({reducer_name} Reduction)', fontsize=22, y=0.98)

        if n_rows == 1:
            axes = axes.reshape(1, -1)
        ax_flat = axes.flatten()

        for i, key in enumerate(plot_keys):
            ax = ax_flat[i]
            title, score = best_algorithms[key]
            ax.set_title(title, fontsize=14, fontweight='bold')

            if key == 'ground_truth':
                sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=y_true_named, palette=custom_palette, ax=ax, s=50, alpha=0.6, edgecolor='none')
            else:
                y_pred = clustering_results[key]
                n_clusters_pred = len(set(y_pred))
                algo_palette = sns.color_palette("husl", n_clusters_pred)
                if -1 in y_pred:
                    cluster_labels = sorted(list(set(y_pred)))
                    color_map = {label: algo_palette[j] for j, label in enumerate(cluster_labels) if label != -1}
                    color_map[-1] = (0.3, 0.3, 0.3)  # Dark gray for noise
                    final_palette = [color_map.get(l) for l in y_pred]
                else:
                    final_palette = algo_palette
                sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=y_pred, palette=final_palette, ax=ax, s=50, alpha=0.6, edgecolor='none', legend='brief')

            ax.set(xlabel=f'{reducer_name}1', ylabel=f'{reducer_name}2')
            if ax.get_legend():
                ax.legend(loc='best', prop={'size': 9}, framealpha=0.7)

        # Hide unused subplots
        for i in range(n_plots, len(ax_flat)):
            ax_flat[i].axis('off')

        plt.tight_layout(rect=[0, 0.01, 1, 0.97])
        filename = f"clustering_comparison_best_{reducer_name}.png"
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        print(f"  Saved visualization to {filename}")
        plt.close()

    # Create additional metric comparison visualization
    print("\n  Creating distance metric comparison chart...")
    create_metric_comparison_plot(ari_scores)

def create_metric_comparison_plot(ari_scores):
    """Creates a bar chart comparing Euclidean vs Cosine distance metrics."""
    # Separate results by algorithm and metric
    comparison_data = []

    for name, score in ari_scores.items():
        if 'dbscan_euclidean' in name:
            eps_type = name.split('_')[-1]
            comparison_data.append({'Algorithm': f'DBSCAN-{eps_type}', 'Metric': 'Euclidean', 'ARI': score})
        elif 'dbscan_cosine' in name:
            eps_type = name.split('_')[-1]
            comparison_data.append({'Algorithm': f'DBSCAN-{eps_type}', 'Metric': 'Cosine', 'ARI': score})
        elif name == 'hdbscan_euclidean':
            comparison_data.append({'Algorithm': 'HDBSCAN', 'Metric': 'Euclidean', 'ARI': score})
        elif name == 'hdbscan_cosine':
            comparison_data.append({'Algorithm': 'HDBSCAN', 'Metric': 'Cosine', 'ARI': score})

    if not comparison_data:
        print("  No metric comparison data available.")
        return

    df_comparison = pd.DataFrame(comparison_data)

    # Create grouped bar chart
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))

    algorithms = df_comparison['Algorithm'].unique()
    x = np.arange(len(algorithms))
    width = 0.35

    euclidean_scores = []
    cosine_scores = []

    for algo in algorithms:
        euc = df_comparison[(df_comparison['Algorithm'] == algo) & (df_comparison['Metric'] == 'Euclidean')]
        cos = df_comparison[(df_comparison['Algorithm'] == algo) & (df_comparison['Metric'] == 'Cosine')]
        euclidean_scores.append(euc['ARI'].values[0] if len(euc) > 0 else 0)
        cosine_scores.append(cos['ARI'].values[0] if len(cos) > 0 else 0)

    bars1 = ax.bar(x - width/2, euclidean_scores, width, label='Euclidean Distance', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, cosine_scores, width, label='Cosine Distance', color='#A23B72', alpha=0.8)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Algorithm Configuration', fontsize=14, fontweight='bold')
    ax.set_ylabel('ARI Score (Adjusted Rand Index)', fontsize=14, fontweight='bold')
    ax.set_title('Distance Metric Comparison: Euclidean vs Cosine\n(Higher ARI = Better Agreement with Ground Truth)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=15, ha='right')
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, max(max(euclidean_scores), max(cosine_scores)) * 1.15])

    plt.tight_layout()
    plt.savefig('distance_metric_comparison.png', dpi=200, bbox_inches='tight')
    print(f"  Saved metric comparison to distance_metric_comparison.png")
    plt.close()

def visualize_kpi_patterns_beautifully(data, labels, kpi_names):
    """
    Generates a final, beautiful 4x1 line plot of KPI patterns with ultra-short, readable labels.
    """
    print("\nGenerating final beautiful KPI pattern visualization...")
    cluster_map = {0: 'Uplink Interference', 1: 'Mass Event', 2: 'Sleeping Cell', 3: 'Noise'}
    colors_map = {0: 'green', 1: 'red', 2: 'purple', 3: 'gray'}
    
    # Final, aggressive mapping for maximum readability in a wide format
    SUPER_SHORT_NAME_MAP = {
        'rach_success_rate': 'RACH SR', 'rrc_connection_attempts': 'RRC Att',
        'rrc_success_rate': 'RRC SR', 'erab_attempts': 'ERAB Att',
        'erab_success_rate': 'ERAB SR', 'erab_abnormal_drops': 'ERAB Drop',
        'erab_normal_drops': 'ERAB NormDrop', 'active_users_dl': 'ActiveUE (DL)',
        'max_active_users_dl': 'MaxUE (DL)', 'data_volume_dl': 'Vol (DL)',
        'prb_utilization_dl': 'PRB (DL)', 'active_users_ul': 'ActiveUE (UL)',
        'max_active_users_ul': 'MaxUE (UL)', 'data_volume_ul': 'Vol (UL)',
        'prb_utilization_ul': 'PRB (UL)', 'ul_sinr_pusch_mean': 'SINR(M)',
        'ul_sinr_pusch_std': 'SINR(S)', 'ul_sinr_pusch_median': 'SINR(Med)',
        'ul_sinr_pusch_p95': 'SINR(P95)', 'ul_sinr_pucch_mean': 'PUCCH SINR',
        'ul_pathloss_mean': 'Pathloss', 'ul_harq_failure_rate': 'HARQ Fail',
        'dl_cqi_mean': 'CQI(M)', 'dl_cqi_std': 'CQI(S)', 'dl_bler': 'DL BLER',
        'interference_mean': 'Interf(M)', 'interference_std': 'Interf(S)',
        'interference_median': 'Interf(Med)', 'interference_p95': 'Interf(P95)',
        'handover_success_rate': 'HO SR', 'pdcch_utilization_mean': 'PDCCH(M)',
        'pdcch_utilization_median': 'PDCCH(Med)', 'pdcch_utilization_p95': 'PDCCH(P95)',
        'dl_packet_latency': 'Pkt Latency', 'mme_initiated_abnormal_drops': 'MME Drop'
    }
    short_kpi_names = [SUPER_SHORT_NAME_MAP.get(kpi, kpi) for kpi in kpi_names]
    
    # Use a 4x1 layout for maximum width
    fig, axes = plt.subplots(4, 1, figsize=(24, 30))
    
    for i, (label_num, label_name) in enumerate(cluster_map.items()):
        ax = axes[i]
        errors = data[labels == label_num]
        if len(errors) == 0: continue

        n_samples_to_show = min(20, len(errors))
        for j in range(n_samples_to_show):
            ax.plot(range(len(kpi_names)), errors[j, :], alpha=0.15, color=colors_map[label_num], linewidth=1.5)
        
        mean_errors = errors.mean(axis=0)
        std_errors = errors.std(axis=0)
        ax.plot(range(len(kpi_names)), mean_errors, color=colors_map[label_num], linewidth=3, label=f'Mean (μ={mean_errors.mean():.2f})', marker='o', markersize=5)
        ax.fill_between(range(len(kpi_names)), mean_errors - std_errors, mean_errors + std_errors, alpha=0.2, color=colors_map[label_num])
        
        ax.set_xticks(range(len(kpi_names)))
        ax.set_xticklabels(short_kpi_names, rotation=0, ha='center', fontsize=10)
        ax.set_ylabel('Reconstruction Error', fontsize=14)
        ax.set_title(f'{label_name}\n({len(errors)} samples)', fontsize=18, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.5, linestyle='--')
        ax.set_ylim(0, 1.6)
        ax.tick_params(axis='x', which='major', pad=5)
        ax.legend(fontsize=12)
        
        if label_num != 3:
            key_kpis_idx = np.argsort(mean_errors)[::-1][:3]
            key_kpis_text = "Top Affected KPIs:\n" + "\n".join([f"  - {short_kpi_names[i]}: {mean_errors[i]:.2f}" for i in key_kpis_idx])
            ax.text(0.98, 0.98, key_kpis_text, transform=ax.transAxes, fontsize=11, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

    fig.suptitle('Anomaly KPI Patterns', fontsize=28, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('kpi_patterns_beautiful.png', dpi=200, bbox_inches='tight')
    print(f"Saved final beautiful KPI pattern visualization to kpi_patterns_beautiful.png")
    plt.close()

def main():
    """Main function to run the entire pipeline."""
    print("="*80)
    print("DBSCAN CLUSTERING PIPELINE - COMPREHENSIVE ALGORITHM COMPARISON")
    print("="*80)
    print("\nInitializing pipeline...")

    kpi_map = load_4g_kpis_from_yaml()
    if not kpi_map:
        print("ERROR: Failed to load KPI configuration from YAML.")
        return

    kpi_names = list(kpi_map.keys())
    print(f"\n✓ Loaded {len(kpi_names)} KPIs from configuration file (kpi_config_dbscan.yaml)")
    print(f"  KPIs loaded: {', '.join(kpi_names[:5])}... (showing first 5)")

    # Verify that we have all expected KPIs
    expected_categories = {
        'Accessibility': ['rach_success_rate', 'rrc_success_rate', 'erab_success_rate'],
        'Retainability': ['erab_abnormal_drops', 'erab_normal_drops'],
        'Traffic': ['active_users_dl', 'data_volume_dl', 'prb_utilization_dl'],
        'Quality': ['ul_sinr_pusch_mean', 'dl_cqi_mean', 'interference_mean'],
        'Mobility': ['handover_success_rate']
    }

    print("\n✓ KPI Categories Present:")
    for category, sample_kpis in expected_categories.items():
        present = sum(1 for kpi in sample_kpis if kpi in kpi_names)
        print(f"  - {category}: {present}/{len(sample_kpis)} sample KPIs found")

    X_scaled, X_unscaled, y_true = generate_synthetic_data(kpi_names, n_points_per_cluster=300, n_noise=200)

    if X_scaled is None:
        print("ERROR: Data generation failed.")
        return

    print(f"\n✓ Generated dataset shape: {X_scaled.shape} (samples × KPIs)")

    results = run_clustering_analysis(X_scaled.values, y_true)
    visualize_results(X_scaled.values, y_true, results)
    visualize_kpi_patterns_beautifully(X_unscaled, y_true, kpi_names)

    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print("\nGenerated outputs:")
    print("  1. clustering_comparison_best_PCA.png - PCA-based cluster visualization")
    print("  2. clustering_comparison_best_t-SNE.png - t-SNE-based cluster visualization")
    print("  3. distance_metric_comparison.png - Euclidean vs Cosine comparison")
    print("  4. kpi_patterns_beautiful.png - KPI signature patterns per anomaly type")
    print("\nRecommendations based on results:")
    print("  - Check the 'Top 5 Overall' summary above for best algorithm")
    print("  - Compare Euclidean vs Cosine metrics in distance_metric_comparison.png")
    print("  - Examine cluster overlap in t-SNE plots to assess task difficulty")
    print("  - Review KPI patterns to understand anomaly signatures")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
