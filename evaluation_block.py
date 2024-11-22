from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score,
                             adjusted_mutual_info_score, v_measure_score,
                             homogeneity_score, completeness_score,
                             confusion_matrix)
import numpy as np


def create_dor_labels(dor_values):
    dor_labels = np.zeros_like(dor_values, dtype=int)
    dor_labels[(dor_values >= 0.3) & (dor_values < 0.6)] = 1
    dor_labels[dor_values >= 0.6] = 2
    return dor_labels


def align_labels(pred_labels, true_labels):
    # Create confusion matrix
    conf_mat = confusion_matrix(true_labels, pred_labels)

    # Find best matching pairs
    n_clusters = conf_mat.shape[1]
    aligned_labels = np.zeros_like(pred_labels)
    used_true = set()
    used_pred = set()

    # For each cluster, find the best match
    while len(used_true) < n_clusters:
        # Find maximum value in confusion matrix
        max_val = 0
        best_true = 0
        best_pred = 0

        for i in range(n_clusters):
            if i in used_true:
                continue
            for j in range(n_clusters):
                if j in used_pred:
                    continue
                if conf_mat[i, j] > max_val:
                    max_val = conf_mat[i, j]
                    best_true = i
                    best_pred = j

        # Assign new label
        aligned_labels[pred_labels == best_pred] = best_true
        used_true.add(best_true)
        used_pred.add(best_pred)

    return aligned_labels


def evaluate_clustering_simplified(X, labels, dor_values=None):
    results = {}

    # Basic cluster information
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    # Skip if only one cluster or all noise points
    if n_clusters < 2:
        return {"error": "Insufficient number of clusters for evaluation"}

    mask = labels != -1
    X_clean = X[mask]
    labels_clean = labels[mask]

    # Calculate internal metrics (these don't need alignment)
    results['internal_metrics'] = {
        'silhouette': silhouette_score(X_clean, labels_clean),
        'calinski_harabasz': calinski_harabasz_score(X_clean, labels_clean),
        'davies_bouldin': davies_bouldin_score(X_clean, labels_clean)
    }

    # Calculate external metrics with aligned labels
    if dor_values is not None:
        dor_labels = create_dor_labels(dor_values[mask])

        # Align the cluster labels with DoR groups
        aligned_labels = align_labels(labels_clean, dor_labels)

        # Calculate metrics using aligned labels
        results['external_metrics'] = {
            'adjusted_rand': adjusted_rand_score(dor_labels, aligned_labels),
            'adjusted_mutual_info': adjusted_mutual_info_score(dor_labels, aligned_labels),
            'v_measure': v_measure_score(dor_labels, aligned_labels),
            'homogeneity': homogeneity_score(dor_labels, aligned_labels),
            'completeness': completeness_score(dor_labels, aligned_labels)
        }

        # Add confusion matrix
        results['confusion_matrix'] = confusion_matrix(dor_labels, aligned_labels)

    return results


def print_metrics_comparison(all_results):
    # Print internal metrics comparison
    print("\n=== Internal Metrics Comparison ===")
    metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin']

    # Create header
    print(f"{'Method':<15}", end='')
    for metric in metrics:
        print(f"{metric:<20}", end='')
    print()
    print("-" * (15 + 20 * len(metrics)))

    # Print values for each method
    for method, results in all_results.items():
        if 'internal_metrics' in results:
            print(f"{method:<15}", end='')
            for metric in metrics:
                value = results['internal_metrics'][metric]
                print(f"{value:< 20.3f}", end='')
            print()

    if any('external_metrics' in results for results in all_results.values()):
        print("\n=== External Metrics Comparison (vs DoR groups) ===")
        metrics = ['adjusted_rand', 'adjusted_mutual_info', 'v_measure',
                   'homogeneity', 'completeness']

        print(f"{'Method':<15}", end='')
        for metric in metrics:
            print(f"{metric:<20}", end='')
        print()
        print("-" * (15 + 20 * len(metrics)))

        # Print values for each method
        for method, results in all_results.items():
            if 'external_metrics' in results:
                print(f"{method:<15}", end='')
                for metric in metrics:
                    value = results['external_metrics'][metric]
                    print(f"{value:< 20.3f}", end='')
                print()

        # Print confusion matrices
        print("\n=== Confusion Matrices (DoR groups vs Aligned Clusters) ===")
        for method, results in all_results.items():
            if 'confusion_matrix' in results:
                print(f"\n{method}:")
                print("Rows: DoR groups (0: 0-0.3, 1: 0.3-0.6, 2: 0.6-1.0)")
                print("Columns: Aligned cluster labels")
                print(results['confusion_matrix'])


def evaluate_all_methods_simplified(methods, X, dor_values):
    all_results = {}

    for method_name, labels in methods.items():
        results = evaluate_clustering_simplified(X, labels, dor_values)
        all_results[method_name] = results

    # Print comparison table
    print_metrics_comparison(all_results)

    return all_results


X, _ = prepare_data(df, selected_features)
X_scaled = StandardScaler().fit_transform(X)

results = evaluate_all_methods_simplified(methods, X_scaled, df['DoR'].values)