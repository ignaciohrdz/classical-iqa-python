import numpy as np


def zca_whitening(x):
    """
    Computes ZCA whitening matrix (aka Mahalanobis whitening).
    Source: https://stackoverflow.com/a/38590790
    :param x: [m x n] matrix (feature_length x n_samples)
    :returns zca_mat: [m x m] matrix
    """

    # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
    sigma = np.cov(x.T, rowvar=True)  # [M x M]

    # Singular Value Decomposition. X = U * np.diag(S) * V
    # U: eigenvectors
    # S: eigenvalues
    # TODO: Deal with the error: SVD did not converge
    U, S, _ = np.linalg.svd(sigma)

    epsilon = 1e-5  # prevents division by zero

    zca_mat = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))
    x_white = np.dot(zca_mat, x.T)

    # Float16 is more memory-efficient
    x_white = np.float16(x_white)

    return x_white.T
