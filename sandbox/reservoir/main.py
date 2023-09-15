import os
import pickle
import numpy as np
import json
import pdb
import argparse

import reservoirpy as rpy
from sklearn.model_selection import train_test_split
from reservoirpy.nodes import Reservoir, Ridge, FORCE
from reservoirpy.observables import nrmse

from sandbox.data_utils import read_csv_with_missing_val, get_groundtruth, read_csv
from sandbox.reservoir.data_utils import format_data, CustomScaler
from sandbox.metrics import Metrics
from sandbox.utils import plot_data, plot_pred
from sandbox.reservoir.sk_node import SklearnNode
rpy.verbosity(0)

def parse_option():
    parser = argparse.ArgumentParser("argument for training")
    parser.add_argument("--root", type=str, default="sandbox/data/phds")
    parser.add_argument("--save_dir", type=str, default="sandbox/new_saved")
    parser.add_argument("--force", action='store_true')
    parser.add_argument("--use_ode", action='store_true')
    opt = parser.parse_args()
    return opt

def get_reservoir(opt):
    nb_seeds = 3
    N = 500
    iss = 0.1
    ridge = 0.003
    sr = 0.16
    lr = 0.46
    seed=0
    reservoir = Reservoir(N,
                      sr=sr,
                      lr=lr,
                      input_scaling=iss,
                      seed=seed)
    # readout = Ridge(ridge=ridge)
    if opt.force:
        readout = FORCE(2)
    else:
        readout = SklearnNode(method="Ridge", alpha=ridge)
    model = reservoir >> readout
    return model

ROOT = 'sandbox/data/phds'
def main(opt):
    # Set up logger
    trial = 1
    if opt.use_ode:
        csv_path = os.path.join(opt.root, f'pred_training_1.csv')
        dataset = read_csv(csv_path)
    else:
        imp_path = f"{opt.save_dir}/imputed_data_{trial}.pkl"
        with open(imp_path, 'rb') as f:
            (dataset, saits) = pickle.load(f)
        # Scale the dataset
    scaler = CustomScaler()
    dataset = scaler.transform(dataset)
    train_dataset, val_dataset = train_test_split(dataset, shuffle=True, train_size=0.9)
    X_train, y_train = format_data(train_dataset)
    X_val, y_val = format_data(val_dataset)
    model = get_reservoir(opt)
    if opt.force:
        model.train(X_train, y_train)
        predictions = model.run(X_val)
    else:
        predictions = model.fit(X_train, y_train).run(X_val)
    loss = nrmse(y_val, predictions, norm_value=np.ptp(X_train))
    print(loss)
    gt_file = os.path.join(ROOT, f'truesetpoint_{trial}.csv')
    y_true = get_groundtruth(gt_file)
    if opt.use_ode:
        test_csv_path = os.path.join(opt.root, f'pred_validation_1.csv')
        all_test_dataset = read_csv(test_csv_path)
    else:
        test_csv_path = os.path.join(opt.root, f'validation_{trial}.csv')
        test_dataset = read_csv_with_missing_val(test_csv_path)
        dataset = {"X": test_dataset}
        all_test_dataset = saits.impute(dataset)
    test_scaler = CustomScaler()
    all_test_dataset = test_scaler.transform(all_test_dataset)
    seed_timesteps = 21
    warming_inputs = all_test_dataset[:, :seed_timesteps, :]
    warming_out = model.run(warming_inputs, reset=True)
    nb_generations = 148
    X_gen = np.zeros((all_test_dataset.shape[0], nb_generations, 2))
    y = np.array(warming_out)
    for t in range(nb_generations):  # generation
        y = np.array(model.run(y))
        y_last = y[:, -1, :]
        X_gen[:, t, :] = y_last
        y = np.concatenate((y[:, 1:, :], y_last[:, None, :]), axis=1)
    # Inverse scaling on generated output
    X_gen = np.concatenate((warming_inputs, X_gen), axis=1)
    y_pred = test_scaler.inverse_transform(X_gen)
    plot_pred(y_pred, y_true, f'{opt.save_dir}/plot_pred_{trial}.png')
    metric_calculator = Metrics(y_true[:50], y_pred[:, -1, 0])
    metrics = metric_calculator()
    print(f'Forecast metrics:{metrics}')
    pdb.set_trace()
if __name__ == "__main__":
    opt = parse_option()
    main(opt)

