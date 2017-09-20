'''
ML/DL SIG experiment 2
9/19
PCA for feature reduction
'''


import argparse
import numpy as np
import sklearn.datasets as ds
from sklearn.preprocessing import StandardScaler


class PCAOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--use_ratio', type=int, default=1, help="whether to perform reduction to using a ratio")
        self.parser.add_argument('--exp_var_ratio', type=float, default=.95, help="explained variance ratio for PCA")
        self.parser.add_argument('--num_features', type=int, default=5, help="number of features to keep")

    def parse_args(self):
        return self.parser.parse_args()


def pca(data, args):
    cov_mat = np.cov(data, rowvar=False)
    eigh_values, eigh_vectors = np.linalg.eig(cov_mat)
    idxs = np.argsort(eigh_values)[::-1]

    if args.use_ratio:
        sum_selected, sum_total, next_idx_to_select = 0, sum(eigh_values), 0

        while (sum_selected / sum_total) <= args.exp_var_ratio:
            sum_selected += eigh_values[idxs[next_idx_to_select]]
            next_idx_to_select += 1

        y = np.dot(data, eigh_vectors.T)

        return np.array([y[:, idxs[i]] for i in range(next_idx_to_select)]).T
    else:
        y = np.dot(data, eigh_vectors)
        
        return np.array([y[:, idxs[i]] for i in range(args.num_features)]).T


args = PCAOptions().parse_args()

z_scaler = StandardScaler()
data = ds.load_breast_cancer()['data'];
z_data = z_scaler.fit_transform(data)

transformed_data = pca(z_data, args)
