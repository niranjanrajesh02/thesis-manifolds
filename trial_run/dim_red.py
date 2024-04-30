import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler

def load_activations():
    # Load the activations from the file
    activations = np.load('/home/niranjan.rajesh_asp24/thesis-manifolds/trial_run/results/activations.npy')
    return activations


def estimate_linear_dim(activations):
    # Estimate the linear dimension of the activations using PCA
    pca = PCA()
    pca.fit(activations)
    
    # Find the number of components required to explain 95% of the variance
    var = np.cumsum(pca.explained_variance_ratio_)
    print("Cumulative variance: ", var)
    linear_dim = np.argmax(var > 0.95) + 1
    print("Linear dimension: ", linear_dim)

    return linear_dim

def estimate_nonlinear_dim_kPCA(activations):
    # Estimate the nonlinear dimension of the activations using Kernel PCA
    kpca = KernelPCA(n_components=activations.shape[1], kernel='rbf', gamma=0.1)
    kpca.fit(activations)

    eigenvalues = kpca.eigenvalues_
    # sort
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    var = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    print("Cumulative variance: ", var)
    nonlinear_dim = np.argmax(var > 0.95) + 1
    print("Nonlinear dimension: ", nonlinear_dim)

    return nonlinear_dim


def estimate_nonlinear_dim_iso(activations):
    # activations = activations.T
    # Estimate the nonlinear dimension of the activations using Isomap

    scaler = StandardScaler()
    activations = scaler.fit_transform(activations)

    isomap = Isomap(n_components=2)
    isomap.fit(activations)

    dist_matrix = isomap.dist_matrix_ # geodesic dist matrix
    double_centered_dist_matrix = np.zeros(dist_matrix.shape)
    row_mean = np.mean(dist_matrix, axis=1)
    col_mean = np.mean(dist_matrix, axis=0)
    grand_mean = np.mean(dist_matrix)

    # dist_matrix = dist_matrix ** 2
    for i in range(dist_matrix.shape[0]):
        for j in range(dist_matrix.shape[1]):
            double_centered_dist_matrix[i, j] = dist_matrix[i, j] - row_mean[i] - col_mean[j] + grand_mean

    eigenvalues, _ = np.linalg.eig(double_centered_dist_matrix)
    eigenvalues = eigenvalues ** 2 #? remove neg eigenvalues 
    

    # print("Eigenvalues: ", eigenvalues)
    # Sort the eigenvalues in descending order
    eigenvalues = np.sort(eigenvalues)[::-1]
    # analogous to the explained variance ratio
    cumulative_eigenval_ratio = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    print("Cumulative variance: ", cumulative_eigenval_ratio)
    # Find the number of components required to explain 95% of the variance
    nonlinear_dim = np.argmax(cumulative_eigenval_ratio > 0.95) + 1
    print("Nonlinear dimension: ", nonlinear_dim)

    return nonlinear_dim




    

if __name__ == '__main__':
    # Load the activations
    activations = load_activations()
    # activations = activations.T
    print(f'Activations Shape:', activations.shape)
    # Estimate the linear dimension
    linear_dim = estimate_linear_dim(activations)
    # Estimate the nonlinear dimension
    nonlinear_dim = estimate_nonlinear_dim_iso(activations)
    
   