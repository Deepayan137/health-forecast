import numpy as np

from pypots.data import load_specific_dataset, mcar, masked_fill
from pypots.imputation import SAITS
from pypots.utils.metrics import cal_mae

import pdb

def impute_fn(X):
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

