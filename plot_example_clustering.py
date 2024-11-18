import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles, make_classification
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

for n_clusters in range(2, 4):
    # Generate datasets
    n_samples = 1500
    random_state = 170
    X1, _ = make_circles(n_samples=n_samples, factor=.5, noise=.05)
    X2, _ = make_moons(n_samples=n_samples, noise=.05)
    X3, _ = make_blobs(n_samples=n_samples, random_state=8)
    X4, _ = make_blobs(n_samples=n_samples, centers=3, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)
    X5, _ = make_classification(n_samples=n_samples, n_features=2, n_informative=2, n_redundant=0,
                                n_clusters_per_class=1, random_state=random_state)

    datasets = [X1, X2, X3, X4, X5]
    clustering_algorithms = [
        ('KMeans', KMeans(n_clusters=n_clusters)),
        ('GaussianMixture', GaussianMixture(n_components=n_clusters)),
        ('AgglomerativeClustering', AgglomerativeClustering(n_clusters=n_clusters))
    ]

    fig, axes = plt.subplots(len(datasets), len(clustering_algorithms), figsize=(15, 10))

    for i, dataset in enumerate(datasets):
        for j, (name, algorithm) in enumerate(clustering_algorithms):
            # Fit the clustering algorithm
            y_pred = algorithm.fit_predict(dataset)

            # Plot the data
            axes[i, j].scatter(dataset[:, 0], dataset[:, 1], c=y_pred, s=10, cmap='viridis')
            if i == 0:
                axes[i, j].set_title(name)
            if j == 0:
                axes[i, j].set_ylabel(['Circles', 'Moons', 'Blobs 1', 'Blobs 2', 'Classification'][i])
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    print("\n\n\n")

    plt.tight_layout()
    plt.show()
