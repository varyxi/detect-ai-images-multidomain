import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

class OODetector:
    def __init__(self, save_path=None, random_state=42, top_dists=2):
        self.save_path = save_path
        self.random_state = random_state
        self.top_dists = top_dists

    # def _distance(self, x, centroids, inv_covs):
    #     dists = []
    #     # for k in range(len(centroids)):
    #     #     diff = x - centroids[k]
    #     #     diff = diff.reshape((1, -1))
    #     #     d = np.sqrt(diff @ inv_covs[k].reshape((-1, 1))) # @ diff)
    #     #     dists.append(float(d[0][0]))
    #     # dists = [d for d in sorted(dists)[:self.top_dists] if not np.isnan(d)]
    #     # if not dists:
    #     #     return float(np.inf)
    #     #print(x.shape, centroids[0].shape, inv_covs[0].shape)
    #     for k in range(len(centroids)):
    #         diff = x - centroids[k]
    #         print(x.shape, diff.shape, centroids[k].shape, inv_covs[k].shape)
    #         d = np.sqrt(diff.reshape((1, -1)) @ inv_covs[k].reshape((-1, 1))) # @ diff.T)
    #         dists.append(d)
    #     #print(dists)
    #     score = sum([dist / (i + 1) for i, dist in enumerate(dists)])
    #     return score

    def _distance(self, x, centroids, inv_covs):
        dists = []
        for k in range(len(centroids)):
            diff = x - centroids[k]  # shape (n_features,)
            inv_cov = inv_covs[k]    # shape (n_features, n_features)
            
            # Reshape diff to column vector (n_features, 1)
            diff_col = diff.reshape(-1, 1)
            
            # Calculate Mahalanobis distance: sqrt((x-μ)ᵀ * Σ⁻¹ * (x-μ))
            d = np.sqrt(diff.T @ inv_cov @ diff)  # Proper vector-matrix-vector multiplication
            
            dists.append(float(d))
        
        dists = [d for d in sorted(dists)[:self.top_dists] if not np.isnan(d)]
        if not dists:
            return float(np.inf)
        return sum([dist / (i + 1) for i, dist in enumerate(dists)])

    def fit(self, embeddings_orig, save_path=None, n_components_pca=100, n_components_gmm=12):
        pca = PCA(n_components = n_components_pca)
        embeddings_pca = pca.fit_transform(np.asarray(embeddings_orig))
        gmm = GaussianMixture(n_components=n_components_gmm, covariance_type='full', max_iter=100, random_state=self.random_state)
        gmm.fit(embeddings_pca)
        if self.save_path is not None:
            joblib.dump((pca, gmm), save_path)

        inv_covs = []
        for k in range(gmm.n_components):
            inv_cov = np.linalg.inv(gmm.covariances_[k] + 1e-5 * np.eye(gmm.covariances_[k].shape[0]))
            inv_covs.append(inv_cov)

        centroids = gmm.means_
        all_train_distances = np.array([self._distance(x, centroids, inv_cov) for x in embeddings_pca])
        threshold = np.percentile(all_train_distances, 99)
        return pca, gmm, centroids, inv_covs, threshold

    def detect(self, x, pca, centroids, inv_covs, threshold):
        x_pca = pca.transform(x.reshape(1, -1))[0]
        score = self._distance(x_pca, centroids, inv_covs)
        return (score > threshold), score