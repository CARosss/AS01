import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import re


def parse_sdss_id(filename):
    match = re.match(r'spec-(\d{4})-(\d{5})-(\d{4})\.fits', filename)
    if match:
        return map(int, match.groups())
    return None, None, None


def load_data():
    kmeans = pd.read_csv("cluster_results/k-means_clusters.csv")
    gmm = pd.read_csv("cluster_results/gmm_clusters.csv")
    hierarchical = pd.read_csv("cluster_results/hierarchical_clusters.csv")
    catalogue = pd.read_csv("data/E-INSPIRE_I_master_catalogue.csv")

    cluster_dfs = []
    for df in [kmeans, gmm, hierarchical]:
        df[['plate', 'mjd', 'fiberid']] = pd.DataFrame(
            [parse_sdss_id(x) for x in df['SDSS_ID']], index=df.index
        )
        merged_df = pd.merge(
            df,
            catalogue[
                ['plate', 'mjd', 'fiberid', 'DoR', 'age_mean_mass', '[M/H]_mean_mass', 'MgFe', 'velDisp_ppxf_res']],
            on=['plate', 'mjd', 'fiberid']
        )
        cluster_dfs.append(merged_df)

    return cluster_dfs[0], cluster_dfs[1], cluster_dfs[2], catalogue


def plot_gaussian_distributions(data, clusters, feature, method_name, ax):
    colors = ['red', 'blue', 'green']
    x = np.linspace(data[feature].min(), data[feature].max(), 100)

    for i in range(len(np.unique(clusters))):
        cluster_data = data[feature][clusters == i]
        mean = np.mean(cluster_data)
        std = np.std(cluster_data)

        gaussian = norm.pdf(x, mean, std)
        # gaussian = gaussian / np.max(gaussian)

        ax.plot(x, gaussian, color=colors[i], label=f'Cluster {i}\nμ={mean:.2f}, σ={std:.2f}')

    ax.set_title(f'{method_name}')
    ax.set_xlabel(feature)
    ax.set_ylabel('Normalized Density')
    ax.legend()
    ax.grid(True, alpha=0.3)


def main():
    kmeans, gmm, hierarchical, catalogue = load_data()

    features = ['DoR', 'age_mean_mass', '[M/H]_mean_mass', 'MgFe', 'velDisp_ppxf_res']
    for feature in features:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'{feature} Distribution Across Clustering Methods')

        plot_gaussian_distributions(kmeans, kmeans['Cluster'], feature, 'K-means', axes[0])
        plot_gaussian_distributions(gmm, gmm['Cluster'], feature, 'GMM', axes[1])
        plot_gaussian_distributions(hierarchical, hierarchical['Cluster'], feature, 'Hierarchical', axes[2])

        plt.show()


if __name__ == "__main__":
    #os.makedirs('cluster_plots', exist_ok=True)
    main()