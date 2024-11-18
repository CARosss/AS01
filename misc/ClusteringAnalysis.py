"""
Advanced Clustering Analysis Module
Provides a comprehensive set of clustering algorithms with visualization and evaluation capabilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, MeanShift, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors


class ClusteringAnalysis:
    def __init__(self, random_state=42):
        """Initialize the clustering analysis with optional random state."""
        self.random_state = random_state
        self.scaler = StandardScaler()

    def prepare_data(self, data, selected_features):
        """
        Prepare and scale data for clustering.

        Parameters:
        -----------
        data : pandas.DataFrame or numpy.ndarray
            Input data
        selected_features : list
            List of feature names or indices

        Returns:
        --------
        tuple : (scaled_data, feature_names, raw_data)
        """
        if isinstance(data, pd.DataFrame):
            if isinstance(selected_features[0], str):
                X = data[selected_features].values
                feature_names = selected_features
            else:
                X = data.iloc[:, selected_features].values
                feature_names = data.columns[selected_features].tolist()
        else:
            X = data[:, selected_features]
            feature_names = [f"Feature {i}" for i in selected_features]

        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, feature_names, X

    def plot_clusters(self, X, labels, feature_names, color_values=None, title="Clustering Results", ax=None):
        """
        Plot clustering results with optional color gradient.

        Parameters:
        -----------
        X : numpy.ndarray
            Input data (first two features used for plotting)
        labels : numpy.ndarray
            Cluster labels
        feature_names : list
            Names of features
        color_values : numpy.ndarray, optional
            Values to use for color gradient
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        unique_labels = np.unique(labels)
        markers = ['o', '^', 's', 'D', 'v']  # Add more if needed

        for label, marker in zip(unique_labels, markers):
            mask = labels == label
            if color_values is not None:
                scatter = ax.scatter(X[mask, 0], X[mask, 1],
                                     c=color_values[mask],
                                     marker=marker,
                                     label=f'Cluster {label}',
                                     cmap='rainbow',
                                     alpha=0.7)
                plt.colorbar(scatter, label='Color Value')
            else:
                ax.scatter(X[mask, 0], X[mask, 1],
                           marker=marker,
                           label=f'Cluster {label}',
                           alpha=0.7)

        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_title(title)
        ax.legend()
        return ax

    def evaluate_clustering(self, X, labels, method_name, feature_names):
        """
        Evaluate clustering results using multiple metrics.

        Parameters:
        -----------
        X : numpy.ndarray
            Scaled input data
        labels : numpy.ndarray
            Cluster labels
        """
        if all(l == -1 for l in labels):
            return {
                'method': method_name,
                'error': 'All points classified as noise'
            }

        mask = labels != -1
        X_clean = X[mask]
        labels_clean = labels[mask]

        if len(np.unique(labels_clean)) > 1:
            metrics = {
                'method': method_name,
                'silhouette': silhouette_score(X_clean, labels_clean),
                'calinski_harabasz': calinski_harabasz_score(X_clean, labels_clean),
                'davies_bouldin': davies_bouldin_score(X_clean, labels_clean),
                'n_clusters': len(np.unique(labels_clean)),
                'cluster_sizes': dict(zip(np.unique(labels_clean),
                                          np.bincount(labels_clean)))
            }

            # Add cluster statistics
            cluster_stats = {}
            for cluster in np.unique(labels_clean):
                cluster_mask = labels_clean == cluster
                stats = {}
                for i, feat in enumerate(feature_names):
                    values = X_clean[cluster_mask, i]
                    stats[feat] = {
                        'mean': np.mean(values),
                        'std': np.std(values)
                    }
                cluster_stats[f'cluster_{cluster}'] = stats
            metrics['cluster_statistics'] = cluster_stats

            return metrics

    def run_clustering(self, data, selected_features, methods=None, color_values=None):
        """
        Run multiple clustering methods and compare results.

        Parameters:
        -----------
        data : pandas.DataFrame or numpy.ndarray
            Input data
        selected_features : list
            List of feature names or indices
        methods : dict, optional
            Dictionary of clustering methods and their parameters
        color_values : numpy.ndarray, optional
            Values to use for color gradient in plots
        """
        X_scaled, feature_names, X_raw = self.prepare_data(data, selected_features)

        if methods is None:
            methods = {
                'KMeans': {'model': KMeans(n_clusters=3, random_state=self.random_state)},
                'DBSCAN': {'model': DBSCAN(eps=0.5, min_samples=5)},
                'MeanShift': {'model': MeanShift(bandwidth=2)},
                'GMM': {'model': GaussianMixture(n_components=3, random_state=self.random_state)},
                'Hierarchical': {'model': AgglomerativeClustering(n_clusters=3)}
            }

        results = {}
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        for i, (method_name, method_dict) in enumerate(methods.items()):
            model = method_dict['model']
            labels = model.fit_predict(X_scaled)

            # Plot results
            if i < len(axes):
                self.plot_clusters(X_raw, labels, feature_names,
                                   color_values, f"{method_name} Clustering", axes[i])

            # Evaluate clustering
            results[method_name] = self.evaluate_clustering(X_scaled, labels,
                                                            method_name, feature_names)

        # Remove empty subplots if any
        for i in range(len(methods), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()

        return results


# Example usage:
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('../data/E-INSPIRE_I_master_catalogue.csv')

    # Initialize clustering analysis
    ca = ClusteringAnalysis()

    # Select features and run analysis
    selected_features = ['MgFe', 'age_mean_mass', '[M/H]_mean_mass', 'velDisp_ppxf_res']
    results = ca.run_clustering(df, selected_features, color_values=df['DoR'].values)

    # Print results
    for method_name, metrics in results.items():
        print(f"\n{method_name} Results:")
        for metric_name, value in metrics.items():
            if metric_name != 'cluster_statistics':
                print(f"{metric_name}: {value}")
        print("\nCluster Statistics:")
        for cluster, stats in metrics.get('cluster_statistics', {}).items():
            print(f"\n{cluster}:")
            for feat, values in stats.items():
                print(f"{feat}: mean = {values['mean']:.3f}, std = {values['std']:.3f}")