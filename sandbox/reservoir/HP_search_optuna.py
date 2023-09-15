import os
import time
import pdb
import argparse
import datetime
import numpy as np
import pandas as pd
import pickle
import joblib
import reservoirpy as rpy
import optuna

from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.observables import nrmse, rsquare
from sklearn.model_selection import train_test_split
from optuna.storages import JournalStorage, JournalFileStorage

from sandbox.data_utils import read_csv_with_missing_val,\
 get_groundtruth, format_data, CustomScaler, read_csv
from sandbox.reservoir.sk_node import SklearnNode
from sandbox.metrics import Metrics
optuna.logging.set_verbosity(optuna.logging.ERROR)


parser = argparse.ArgumentParser()
parser.add_argument('--nb_trials', type=int, required=True)
parser.add_argument('--study_name', type=str, required=True)
parser.add_argument("--save_dir", type=str, default="sandbox/data")
parser.add_argument("--root", type=str, default="sandbox/data/phds")
parser.add_argument("--use_ode", action='store_true')

args = parser.parse_args()

ROOT = 'sandbox/data/phds'
num=40
if args.use_ode:
    csv_path = os.path.join(ROOT, f'pred_training_1.csv')
    dataset = read_csv(csv_path)
else:
    # Data Preprocessing
    imp_path = f"{args.save_dir}/imputed_data_{num}.pkl"
    with open(imp_path, 'rb') as f:
        (dataset, saits) = pickle.load(f)

# Scale the dataset
scaler = CustomScaler()
dataset = scaler.transform(dataset)

train_dataset, val_dataset = train_test_split(dataset, shuffle=True, train_size=0.9)
X_train, y_train = format_data(train_dataset)
X_test, y_test = format_data(val_dataset)

dataset = ((X_train, y_train), (X_test, y_test))

gt_file = os.path.join(ROOT, f'truesetpoint_{num}.csv')
y_true = get_groundtruth(gt_file)
if args.use_ode:
    test_csv_path = os.path.join(ROOT, f'pred_validation_1.csv')
    all_test_dataset = read_csv(test_csv_path)
else:
    test_csv_path = os.path.join(ROOT, f'validation_{num}.csv')
    test_dataset = read_csv_with_missing_val(test_csv_path)
    dataset = {"X": test_dataset}
    all_test_dataset = saits.impute(dataset)

test_scaler = CustomScaler()
all_test_dataset = test_scaler.transform(all_test_dataset)
seed_timesteps = 21
warming_inputs = all_test_dataset[:, :seed_timesteps, :]
# Trial Fixed hyper-parameters
nb_seeds = 3
N = 500
iss = 0.1
sr = 0.1
lr = 0.4
ridge = 0.001
def objective(trial):
    # Record objective values for each trial
    rpy.verbosity(0)
    losses = []

    # Trial generated parameters (with log scale)
    # iss = trial.suggest_float("sr", 0.7, 3.0, log=True)
    sr = trial.suggest_float("sr", 1e-2, 5, log=True)
    lr = trial.suggest_float("lr", 1e-3, 2, log=True)
    # N = trial.suggest_int('N', 100, 500, 100)
    ridge = trial.suggest_float("ridge", 1e-6, 1, log=True)
    for seed in range(nb_seeds):
        reservoir = Reservoir(N,
                              sr=sr,
                              lr=lr,
                              input_scaling=iss,
                              seed=seed)
        
        # readout = Ridge(ridge=ridge)
        readout = SklearnNode(method="Ridge", alpha=ridge)
        model = reservoir >> readout

        # Train and test your model
        predictions = model.fit(X_train, y_train).run(X_test)

        # Compute the desired metric(s)
        loss = nrmse(y_test, predictions, norm_value=np.ptp(X_train))
        # losses.append(loss)
        warming_out = model.run(warming_inputs, reset=True)
        nb_generations = 148
        X_gen = np.zeros((all_test_dataset.shape[0], nb_generations, 1))
        y = np.array(warming_out)
        for t in range(nb_generations):  # generation
            y = np.array(model.run(y))
            y_last = y[:, -1, :]
            X_gen[:, t, :] = y_last
            y = np.concatenate((y[:, 1:, :], y_last[:, None, :]), axis=1)
        
        X_gen = np.concatenate((warming_inputs, X_gen), axis=1)
        y_pred = test_scaler.inverse_transform(X_gen)
        metric_calculator = Metrics(y_true, y_pred[:, -1, 0])
        metrics = metric_calculator()
        print(metrics)
        losses.append(metrics['rmse'])
    return np.mean(losses)


# Define study parameters
sampler = optuna.samplers.RandomSampler() 
log_name = f"hp_logs/optuna-journal_{args.study_name}.log"
storage = JournalStorage(JournalFileStorage(log_name))
def optimize_study(n_trials):
    study = optuna.create_study(
        study_name=args.study_name,
        direction='minimize',
        storage=storage,
        sampler=sampler,
        load_if_exists=True
    )

    for i in range(n_trials):
        trial = study.ask()
        study.tell(trial, objective(trial))


# Launch the optimization for this specific job
start = time.time()
n_process = 5
n_trials_per_process = args.nb_trials // n_process
args_list = [n_trials_per_process for i in range(n_process)]
# study.optimize(objective, n_trials=args.nb_trials)
joblib.Parallel(n_jobs=n_process)(joblib.delayed(optimize_study)(args) for args in args_list)

end = time.time()

print(f"Optimization done with {args.nb_trials} trials in {str(datetime.timedelta(seconds=end-start))}")