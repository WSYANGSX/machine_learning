import numpy as np


def generate_multivariate_gaussian(mu, sigma, n_samples):
    """
    生成多维高斯分布的样本
    :param mu: 均值向量 (numpy array)
    :param sigma: 协方差矩阵 (numpy array)
    :param n_samples: 样本数量
    :return: 样本矩阵 (shape: (n_samples, len(mu)))
    """
    d = len(mu)
    L = np.linalg.cholesky(sigma)
    z = np.random.normal(0, 1, (n_samples, d))
    x = np.dot(z, L.T) + mu
    return x


# 示例
mu = np.array([0, 0])
sigma = np.array([[1, 0.5], [0.5, 1]])
n_samples = 1000
samples = generate_multivariate_gaussian(mu, sigma, n_samples)
print(samples)