
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA

# look mommy I can do PCA!


def display(matrix):
    plt.scatter(matrix[:, 0], matrix[:, 1])
    plt.show()


data = pd.read_csv('HR_comma_sep.csv')
sample_rows = data.sample(500)
numbers = sample_rows.iloc[:, :-2]
matrix = numbers.values
mean = np.mean(matrix, axis=0)
normed_matrix = (matrix - mean)/np.std(matrix, axis=0)
cov_mat = np.cov(normed_matrix, rowvar=False, bias=True)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
result = np.dot(normed_matrix, eig_vecs[:, :2])
display(result)

alt_result = PCA(matrix)


