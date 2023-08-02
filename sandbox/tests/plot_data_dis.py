import pandas as pd
import pdb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sandbox.data_utils import read_csv_with_missing_val
from sandbox.impute import impute_fn

def rescale(matrix):
    scaler = StandardScaler()
    N, T, D = matrix.shape
    X = matrix.reshape(N * T, D)
    rescaled_matrix = scaler.fit_transform(X)
    rescaled_matrix = rescaled_matrix.reshape(N, T, D)
    return rescaled_matrix, scaler

dataset = read_csv_with_missing_val("sandbox/data/phds/training_1.csv")
dataset, train_scaler = rescale(dataset)
numerical_data_rescaled, saits = impute_fn(dataset)
pdb.set_trace()
numerical_data_rescaled_flat = numerical_data_rescaled.flatten()

# Plot the distribution of the rescaled training data
plt.hist(numerical_data_rescaled_flat, bins=50)
plt.title('Distribution of Rescaled Training Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
