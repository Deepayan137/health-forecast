import os
import pandas as pd
import pickle
import numpy as np
import json
import pdb
import argparse

import reservoirpy as rpy
from sklearn.model_selection import train_test_split
from reservoirpy.nodes import Reservoir, Ridge, FORCE
from reservoirpy.observables import nrmse

from sandbox.data_utils import read_csv_with_missing_val, get_groundtruth,\
 read_csv, get_groundtruth_modified
from sandbox.reservoir.data_utils import format_data, CustomScaler
from sandbox.metrics import Metrics
from sandbox.utils import plot_data, plot_pred
from sandbox.reservoir.sk_node import SklearnNode
from sandbox.impute import impute_missing_values, impute_test_data
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
    ridge = 0.01
    sr = 0.5
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
        dataset, labels = read_csv(csv_path)
    else:
        csv_path = os.path.join(ROOT, f'training_{trial}.csv')
        dataset, labels = read_csv_with_missing_val(csv_path)
        imp_path = f"{opt.save_dir}/imputed_data_{trial}.pkl"
        with open(imp_path, 'rb') as f:
            (dataset, saits0, saits1) = pickle.load(f)
        # Scale the dataset
    scaler = CustomScaler()
    dataset = scaler.transform(dataset)
    ###
    labels = np.array(labels) # Convert list to numpy array
    expanded_labels = np.expand_dims(labels, axis=1).repeat(dataset.shape[1], axis=1)
    expanded_labels = np.expand_dims(expanded_labels, axis=2)
    dataset = np.concatenate((dataset, expanded_labels), axis=2)
    ###
    train_dataset, val_dataset = train_test_split(dataset, shuffle=True, train_size=0.9)
    X_train, y_train = format_data(train_dataset)
    X_val, y_val = format_data(val_dataset)
    model = get_reservoir(opt)
    if opt.force:
        model.train(X_train, y_train)
        predictions = model.run(X_val)
    else:
        predictions = model.fit(X_train, y_train).run(X_val)
    print("Model trained")
    predictions = np.array(predictions)
    loss = nrmse(y_val, predictions, norm_value=np.ptp(X_train))
    ###
    gt_file = os.path.join(ROOT, f'true_validation{trial}.csv')
    y_true = get_groundtruth_modified(gt_file)
    ###
    if opt.use_ode:
        test_csv_path = os.path.join(opt.root, f'pred_validation_1.csv')
        all_test_dataset, test_labels = read_csv(test_csv_path)
    else:
        test_csv_path = os.path.join(opt.root, f'validation_{trial}.csv')
        test_dataset, test_labels = read_csv_with_missing_val(test_csv_path)
        all_test_dataset = impute_test_data(test_dataset, test_labels, saits0, saits1)
        
    all_test_dataset = scaler.transform(all_test_dataset)
    seed_timesteps = 21
    warming_inputs = all_test_dataset[:, :seed_timesteps, :]
    ####
    test_labels = np.array(test_labels) # Convert list to numpy array
    expanded_test_labels = np.expand_dims(test_labels, axis=1).repeat(warming_inputs.shape[1], axis=1)
    expanded_test_labels = np.expand_dims(expanded_test_labels, axis=2)
    warming_inputs = np.concatenate((warming_inputs, expanded_test_labels), axis=2)
    ####
    warming_out = model.run(warming_inputs, reset=True)
    nb_generations = 148
    X_gen = np.zeros((all_test_dataset.shape[0], nb_generations, 2))
    y = np.array(warming_out)
    print("Auto regressive phase")
    for t in range(nb_generations):  # generation
        y = np.array(model.run(y))
        y_last = y[:, -1, :]
        X_gen[:, t, :] = y_last
        y = np.concatenate((y[:, 1:, :], y_last[:, None, :]), axis=1)
    # Inverse scaling on generated output
    X_gen = np.concatenate((warming_inputs, X_gen), axis=1)
    X_gen = X_gen[:, :, 0]
    y_pred = scaler.inverse_transform(X_gen)
    np.save('final/reservoir_predictions_ode.npy', y_pred)
    # plot_pred(y_pred, y_true, f'{opt.save_dir}/plot_pred_{trial}.png')
    selected_days = [30, 60, 90, 120, 150, 168]
    metric_calculator = Metrics(y_true, y_pred[:, selected_days, 0])
    metrics = metric_calculator()
    df = pd.DataFrame(metrics)
    df.to_csv('final/reservoir_metrics_every_month_ode.csv')
    print(f'Forecast metrics:{metrics}')
if __name__ == "__main__":
    opt = parse_option()
    main(opt)

