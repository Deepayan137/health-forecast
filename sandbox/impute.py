import numpy as np

from pypots.data import load_specific_dataset, mcar, masked_fill
from pypots.imputation import SAITS
from pypots.utils.metrics import cal_mae

import pdb
def moving_average(data, window_size=5):
    """Compute the moving average of a time series."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def call_saits(X):
    n_features = X.shape[-1]
    X_intact, X, missing_mask, indicating_mask = mcar(X, 0.1)
    X = masked_fill(X, 1 - missing_mask, np.nan)
    dataset = {"X": X}
    # Model training. This is PyPOTS showtime.
    saits = SAITS(n_steps=169, n_features=n_features, n_layers=2, 
        d_model=256, d_inner=128, n_heads=4, 
        d_k=64, d_v=64, dropout=0.1, epochs=16)
    saits.fit(dataset)  # train the model. Here I use the whole dataset as the training set, because ground truth is not visible to the model.
    imputation = saits.impute(dataset)  # impute the originally-missing values and artificially-missing values
    # mae = cal_mae(imputation, X_intact, indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)
    return imputation, saits


def impute_missing_values(data, labels):
    # Extract data for V (assuming it's the first dimension)
    data_V = data[:, :, 0]
    
    # Separate data based on groups
    group_0_data = data_V[np.array(labels) == 0]
    group_1_data = data_V[np.array(labels) == 1]
    
    # Linear interpolation
    ffill_bfill_0, saits_0 = call_saits(group_0_data[:, :, None])
    ffill_bfill_1, saits_1 = call_saits(group_1_data[:, :, None])
    # ffill_bfill_0 = np.array([moving_average(patient, 5) for patient in ffill_bfill_0.squeeze()])
    # ffill_bfill_1 = np.array([moving_average(patient, 5) for patient in ffill_bfill_1.squeeze()])
    data = np.concatenate((ffill_bfill_0, ffill_bfill_1), axis=0)
    return data, saits_0, saits_1

def impute_test_data(data, labels, saits0, saits1):
    data_V = data[:, :, 0][:, :, None]
    # Separate data based on groups
    group_0_data = data_V[np.array(labels) == 0]
    dataset = {"X": group_0_data}
    ffill_bfill_0 = saits0.impute(dataset)
    
    group_1_data = data_V[np.array(labels) == 1]
    dataset = {"X": group_1_data}
    ffill_bfill_1 = saits1.impute(dataset)
    data = np.concatenate((ffill_bfill_0, ffill_bfill_1), axis=0)
    return data
