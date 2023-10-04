import os
import pandas as pd
import pdb
import pickle
import numpy as np
import json
import argparse
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from src.data_utils import read_csv_with_missing_val, get_groundtruth,\
 read_csv, CustomScaler
from src.metrics import Metrics
from src.impute import impute_missing_values, impute_test_data

np.random.seed(0)

def parse_option():
    parser = argparse.ArgumentParser("argument for training")
    parser.add_argument("--root", type=str, default="data/phds")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--input_dim", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--output_dim", type=int, default=4)
    parser.add_argument("--use_ode", action='store_true')
    opt = parser.parse_args()
    return opt

def main(opt):
    # Set up logger
    trial = 1
    if opt.use_ode:
        csv_path = os.path.join(opt.root, f'pred_training_{trial}.csv')
        dataset, labels = read_csv(csv_path)
    else:
        csv_path = os.path.join(opt.root, f'training_{trial}.csv')
        dataset, labels = read_csv_with_missing_val(csv_path)
        imp_path = f"{opt.root}/imputed_data_{trial}.pkl"
        if not os.path.exists(imp_path):
            dataset, saits0, saits1 = impute_missing_values(dataset, labels, smooth=True)
            print(f"Writing imputed data to {imp_path}")
            with open(imp_path, 'wb') as f:
                pickle.dump((dataset, saits0, saits1), f)
        else:
            print(f"Loading imputed data from {imp_path}")
            with open(imp_path, 'rb') as f:
                (dataset, saits0, saits1) = pickle.load(f)
    
    labels = np.array(labels) # Convert list to numpy array
    expanded_labels = np.expand_dims(labels, axis=1).repeat(dataset.shape[1], axis=1)
    expanded_labels = np.expand_dims(expanded_labels, axis=2)
    dataset = np.concatenate((dataset, expanded_labels), axis=2)

    gt_file = os.path.join(opt.root, f'true_validation{trial}.csv')
    y_true = get_groundtruth(gt_file)

    if opt.use_ode:
        test_csv_path = os.path.join(opt.root, f'pred_validation_{trial}.csv')
        all_test_dataset, test_labels = read_csv(test_csv_path)
    else:
        test_csv_path = os.path.join(opt.root, f'validation_{trial}.csv')
        test_dataset, test_labels = read_csv_with_missing_val(test_csv_path)
        all_test_dataset = impute_test_data(test_dataset, test_labels, saits0, saits1)
    
    seed_timesteps = 21
    warming_inputs = all_test_dataset[:, :seed_timesteps, :]
    ####
    test_labels = np.array(test_labels) # Convert list to numpy array
    expanded_test_labels = np.expand_dims(test_labels, axis=1).repeat(warming_inputs.shape[1], axis=1)
    expanded_test_labels = np.expand_dims(expanded_test_labels, axis=2)
    X_test = np.concatenate((warming_inputs, expanded_test_labels), axis=2)
    ####
    np.random.shuffle(dataset)
    selected_days = [30, 60, 90, 120, 150, 164]
    X_train, y_train = dataset[:, :21, :], dataset[:, selected_days, 0]
    X_train = X_train.reshape(X_train.shape[0], -1)
    regr = MLPRegressor(random_state=1, max_iter=1000, hidden_layer_sizes=(512)).fit(X_train, y_train)
    y_pred = regr.predict(X_test.reshape(X_test.shape[0], -1))
    metric_calculator = Metrics(y_true, y_pred)
    metrics = metric_calculator()
    df = pd.DataFrame(metrics)
    if opt.use_ode:
        df.to_csv(f'{opt.save_dir}/metrics/mlp_metrics_every_month_ode_{trial}.csv')
    else:
        df.to_csv(f'{opt.save_dir}/metrics/mlp_metrics_every_month_raw_{trial}.csv')
    print(f'Forecast metrics:{metrics}')

if __name__ == "__main__":
    opt = parse_option()
    main(opt)
