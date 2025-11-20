
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
    Generates a highly realistic synthetic dataset with fluctuating anomaly ceilings
    and sporadic noise.
    """
    if not kpi_list:
        print("KPI list is empty. Cannot generate data.")
        return None, None, None

    n_kpis = len(kpi_list)
    total_points = 3 * n_points_per_cluster + n_noise
    print(f"Generating final realistic data with {total_points} points and {n_kpis} KPIs.")

    # Create a baseline of very low reconstruction error
    data = np.random.uniform(0.0, 0.05, size=(total_points, n_kpis))
    labels = np.zeros(total_points, dtype=int)

    # Helper for sporadic noise
    def add_sporadic_noise(start, end, primary_kpis):
        for i in range(start, end):
            for kpi_idx in range(n_kpis):
                if kpi_list[kpi_idx] not in primary_kpis and np.random.rand() < 0.05:
                    data[i, kpi_idx] = np.random.uniform(0.4, 0.6)
    
    # Helper to generate varied anomalies
    def generate_cluster_anomalies(start, end, primary_kpis, base_level):
        for kpi_name in primary_kpis:
            if kpi_name in kpi_list:
                kpi_index = kpi_list.index(kpi_name)
                # Each KPI gets its own randomized ceiling
                anomaly_floor = base_level - np.random.uniform(0.1, 0.2)
                anomaly_ceiling = base_level + np.random.uniform(0.1, 0.3)
                data[start:end, kpi_index] = np.random.uniform(anomaly_floor, anomaly_ceiling, size=n_points_per_cluster)

    # --- Cluster A: Uplink Interference ---
    start, end = 0, n_points_per_cluster
    labels[start:end] = 0
    generate_cluster_anomalies(start, end, UPLINK_INTERFERENCE_KPIS, base_level=1.0)
    add_sporadic_noise(start, end, UPLINK_INTERFERENCE_KPIS)

    # --- Cluster B: Mass Event ---
    start, end = n_points_per_cluster, 2 * n_points_per_cluster
    labels[start:end] = 1
    generate_cluster_anomalies(start, end, MASS_EVENT_KPIS, base_level=0.8)
    add_sporadic_noise(start, end, MASS_EVENT_KPIS)

    # --- Cluster C: Sleeping Cell ---
    start, end = 2 * n_points_per_cluster, 3 * n_points_per_cluster
    labels[start:end] = 2
    generate_cluster_anomalies(start, end, SLEEPING_CELL_KPIS, base_level=1.05)
    add_sporadic_noise(start, end, SLEEPING_CELL_KPIS)

    # --- Noise Points: Lone Wolf Anomalies ---
    start, end = 3 * n_points_per_cluster, total_points
    labels[start:end] = 3
    data[start:end, :] = np.random.uniform(0.0, 1.3, size=(n_noise, n_kpis))

    data_unscaled = data.copy()
    np.clip(data_unscaled, 0, 1.5, out=data_unscaled) # Clip at a reasonable max

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_unscaled)
    df_scaled = pd.DataFrame(data_scaled, columns=kpi_list)
    return df_scaled, data_unscaled, np.array(labels)

def run_clustering_analysis(X, y_true):
    """
    Applies K-means, DBSCAN, and HDBSCAN, and prints a detailed analysis.
    """
    print("\n" + "="*80)
    print("Starting Clustering Model Application and Analysis")
    print("="*80 + "\n")
    results = {}

    print("\n--- 1. K-means Analysis ---\n")
    for k in [3, 4]:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        y_pred = kmeans.fit_predict(X)
        results[f'kmeans_{k}'] = y_pred
        ari = adjusted_rand_score(y_true, y_pred)
        silhouette = silhouette_score(X, y_pred)
        print(f"K-means (k={k}): ARI={ari:.4f}, Silhouette={silhouette:.4f}")

    print("\n--- 2. DBSCAN Analysis (Metric: Cosine) ---\n")
    distances = pairwise_distances(X, metric='cosine')
    nn_distances = np.sort(np.min(distances + np.eye(len(distances)), axis=1))
    eps_values = {'small': np.quantile(nn_distances, 0.3), 'medium': np.quantile(nn_distances, 0.6), 'large': np.quantile(nn_distances, 0.95)}
    print(f"Experimenting with 3 Epsilon values: Small (~{eps_values['small']:.3f}), Medium (~{eps_values['medium']:.3f}), Large (~{eps_values['large']:.3f})\n")
    for name, eps in eps_values.items():
        dbscan = DBSCAN(eps=eps, min_samples=10, metric='cosine')
        y_pred = dbscan.fit_predict(X)
        results[f'dbscan_{name}'] = y_pred
        n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
        n_noise = np.sum(y_pred == -1)
        print(f"DBSCAN (Epsilon={eps:.3f} - '{name}'): Clusters={n_clusters}, Noise={n_noise} ({n_noise/len(X):.1%}), ARI={adjusted_rand_score(y_true, y_pred):.4f}")

    print("\n--- 3. HDBSCAN Analysis (Metric: Cosine) ---\n")
    hdb = hdbscan.HDBSCAN(min_cluster_size=15, metric='cosine', algorithm='generic')
    y_pred = hdb.fit_predict(X)
    results['hdbscan'] = y_pred
    n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
    n_noise = np.sum(y_pred == -1)
    print(f"HDBSCAN: Clusters={n_clusters}, Noise={n_noise} ({n_noise/len(X):.1%}), ARI={adjusted_rand_score(y_true, y_pred):.4f}")

    return results

def visualize_results(X, y_true, clustering_results):
    """
    Generates and saves 2x4 plots comparing all clustering results using PCA and t-SNE.
    """
    print("\nGenerating scatter plot visualizations...")
    cluster_map_true = {0: 'Uplink Interference', 1: 'Mass Event', 2: 'Sleeping Cell', 3: 'Noise'}
    y_true_named = [cluster_map_true[l] for l in y_true]
    custom_palette = {name: color for name, color in zip(cluster_map_true.values(), sns.color_palette("viridis", 4))}
    custom_palette['Noise'] = (0.5, 0.5, 0.5)

    reducers = {'PCA': PCA(n_components=2, random_state=42), 't-SNE': TSNE(n_components=2, random_state=42, perplexity=50, max_iter=1000)}
    plot_titles = {'ground_truth': 'Ground Truth', 'kmeans_3': 'K-means (k=3)', 'kmeans_4': 'K-means (k=4)', 'dbscan_small': 'DBSCAN (Small ε)', 'dbscan_medium': 'DBSCAN (Medium ε)', 'dbscan_large': 'DBSCAN (Large ε)', 'hdbscan': 'HDBSCAN'}
    plot_keys = list(plot_titles.keys())

    for reducer_name, reducer in reducers.items():
        X_2d = reducer.fit_transform(X)
        fig, axes = plt.subplots(2, 4, figsize=(28, 14))
        fig.suptitle(f'Clustering Comparison with {reducer_name} Reduction', fontsize=22, y=0.97)
        ax_flat = axes.flatten()

        for i, key in enumerate(plot_keys):
            ax = ax_flat[i]
            ax.set_title(plot_titles[key], fontsize=16)
            if key == 'ground_truth':
                sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=y_true_named, palette=custom_palette, ax=ax, s=50, alpha=0.7)
            else:
                y_pred = clustering_results[key]
                n_clusters_pred = len(set(y_pred))
                algo_palette = sns.color_palette("deep", n_clusters_pred)
                if -1 in y_pred:
                    cluster_labels = sorted(list(set(y_pred)))
                    color_map = {label: algo_palette[i] for i, label in enumerate(cluster_labels) if label != -1}
                    color_map[-1] = (0.5, 0.5, 0.5)
                    final_palette = [color_map.get(l) for l in y_pred]
                else:
                    final_palette = algo_palette
                sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=y_pred, palette=final_palette, ax=ax, s=50, alpha=0.7, legend='full')
            
            ax.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
            if ax.get_legend(): ax.legend(loc='upper right', prop={'size': 10})
        
        ax_flat[7].axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filename = f"clustering_comparison_{reducer_name}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {filename}")
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
    kpi_map = load_4g_kpis_from_yaml()
    if not kpi_map:
        return
    
    kpi_names = list(kpi_map.keys())

    X_scaled, X_unscaled, y_true = generate_synthetic_data(kpi_names, n_points_per_cluster=300, n_noise=200)
    
    if X_scaled is None:
        return
        
    results = run_clustering_analysis(X_scaled, y_true)
    visualize_results(X_scaled.values, y_true, results)
    visualize_kpi_patterns_beautifully(X_unscaled, y_true, kpi_names)

if __name__ == "__main__":
    main()
