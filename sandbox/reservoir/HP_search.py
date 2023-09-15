import time
import argparse
import datetime
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

import reservoirpy as rpy

from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.observables import nrmse, rsquare
from sklearn.model_selection import train_test_split
from reservoirpy.hyper import research

parser = argparse.ArgumentParser()
parser.add_argument('--nb_trials', type=int, required=True)
parser.add_argument('--study_name', type=str, required=True)
parser.add_argument("--save_dir", type=str, default="../new_saved/")
args = parser.parse_args()

# Data Preprocessing
imp_path = f"{args.save_dir}/imputed_data_1.pkl"
with open(imp_path, 'rb') as f:
    (dataset, saits) = pickle.load(f)

def format_data(dataset):
    N, T, D = dataset.shape
    X = dataset[:, :T-1]
    y = dataset[:, 1:T]
    return X, y

def min_max_scale(data):
    """
    Scale data using MinMaxScaler along the last dimension.

    Args:
    - data (np.ndarray): A 3D numpy array of shape (N, T, D).

    Returns:
    - np.ndarray: Scaled data of the same shape.
    """
    N, T, D = data.shape
    
    # Reshape data to be 2D
    data_reshaped = data.reshape(-1, D)
    
    # Initialize and fit scaler
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_reshaped)
    
    # Reshape back to original shape
    data_scaled = data_scaled.reshape(N, T, D)
    
    return data_scaled

train_dataset, val_dataset = train_test_split(dataset, shuffle=True, train_size=0.9)
train_dataset, val_dataset = min_max_scale(train_dataset), min_max_scale(val_dataset)
X_train, y_train = format_data(train_dataset)
X_test, y_test = format_data(val_dataset)

dataset = ((X_train, y_train), (X_test, y_test))

# Trial Fixed hyper-parameters

def objective(dataset, config, *, iss, N, sr, lr, ridge, seed):
    train_data, validation_data = dataset
    X_train, y_train = train_data
    X_val, y_val = validation_data

    # Record objective values for each trial
    rpy.verbosity(0)
    losses = []

    for seed in range(config["instances_per_trial"]):
        reservoir = Reservoir(N,
                              sr=sr,
                              lr=lr,
                              input_scaling=iss,
                              seed=seed)
        
        readout = Ridge(ridge=ridge)
        model = reservoir >> readout

        # Train and test your model
        predictions = model.fit(X_train, y_train).run(X_test)

        # Compute the desired metric(s)
        loss = nrmse(y_test, predictions, norm_value=np.ptp(X_train))

        losses.append(loss)

    return {'loss':np.mean(losses)}


# Define study parameters
hyperopt_config = {
    "exp": f"{args.study_name}", # the experimentation name
    "hp_max_evals": args.nb_trials,             # the number of differents sets of parameters hyperopt has to try
    "hp_method": "random",           # the method used by hyperopt to chose those sets (see below)
    "seed": 42,                      # the random state seed, to ensure reproducibility
    "instances_per_trial": 3,        # how many random ESN will be tried with each sets of parameters
    "hp_space": {                    # what are the ranges of parameters explored
        "N": ["choice", 500],             # the number of neurons is fixed to 300
        "sr": ["loguniform", 1e-2, 10],   # the spectral radius is log-uniformly distributed between 1e-6 and 10
        "lr": ["loguniform", 1e-3, 1],  # idem with the leaking rate, from 1e-3 to 1
        "iss": ["choice", 0.9],           # the input scaling is fixed
        "ridge": ["loguniform", 1e-9, 1],        # and so is the regularization parameter.
        "seed": ["choice", 1234]          # an other random seed for the ESN initialization
    }
}



# Create study
# we precautionously save the configuration in a JSON file
# each file will begin with a number corresponding to the current experimentation run number.
with open(f"{hyperopt_config['exp']}.config.json", "w+") as f:
    json.dump(hyperopt_config, f)


# Launch the optimization for this specific job
start = time.time()
best = research(objective, dataset, f"{hyperopt_config['exp']}.config.json", ".")
end = time.time()

print(f"Optimization done with {args.nb_trials} trials in {str(datetime.timedelta(seconds=end-start))}")