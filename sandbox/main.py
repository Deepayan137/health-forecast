import os
import pdb
import pandas as pd
import numpy as np
import pickle
import json
import argparse
import logging
from datetime import datetime
import sys
from copy import deepcopy
from tqdm import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from sandbox.data_utils import read_csv_with_missing_val, \
get_groundtruth, create_sub_sequences, rescale, inverse_transform_data
from sandbox.utils import plot_pred, setup_logger
from sandbox.model import LSTMModel, TransAm
from sandbox.trainer import Trainer
from sandbox.impute import impute_fn
from sandbox.metrics import Metrics

torch.manual_seed(42)  #

def parse_option():
    parser = argparse.ArgumentParser("argument for training")
    parser.add_argument("--root", type=str, default="sandbox/data/phds")
    parser.add_argument("--imp_method", type=str, default="saits")
    parser.add_argument("--window_size", type=int, default=21)
    parser.add_argument("--forecast", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--out_dim", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--do_jitter", action='store_true')
    parser.add_argument("--do_mixup", action='store_true')
    opt = parser.parse_args()
    return opt

def main(opt):
    # Set up logger
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%d-%m-%Y_%H:%M:%S")
    imp_method = opt.imp_method
    exp_name = f"{formatted_datetime}"
    log_path = os.path.join("sandbox", "experiments", exp_name)
    os.makedirs(log_path)
    args_dict = vars(opt)
    # Save the dictionary to a json file
    with open(os.path.join(log_path, 'config.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)
    logging = setup_logger(log_path)
    logging.info('Started')
    trials = range(1, 21)
    metric_list = []
    for trial in trials:
        if trial > 5:
            break
        csv_path = os.path.join(opt.root, f'training_{trial}.csv')
        imp_path = f"sandbox/new_saved/imputed_data_{trial}.pkl"
        if not os.path.exists(imp_path):
            dataset = read_csv_with_missing_val(csv_path)
            dataset, train_scaler = rescale(dataset)
            dataset, saits = impute_fn(dataset)
            print(f"Writing imputed data to {imp_path}")
            with open(imp_path, 'wb') as f:
                pickle.dump((dataset, saits), f)
        else:
            print(f"Loading imputed data from {imp_path}")
            with open(imp_path, 'rb') as f:
                (dataset, saits) = pickle.load(f)

        train_dataset, val_dataset = train_test_split(dataset, shuffle=False, train_size=0.9)
        test_csv_path = os.path.join(opt.root, f'validation_{trial}.csv')
        test_dataset = read_csv_with_missing_val(test_csv_path)
        test_dataset, test_scaler = rescale(test_dataset)
        dataset = {"X": test_dataset}
        all_test_dataset = saits.impute(dataset)
        test_dataset = all_test_dataset[:, :opt.window_size, :]
        test_dataset = torch.tensor(test_dataset).float()
        gt_file = os.path.join(opt.root, f'truesetpoint_{trial}.csv')
        y_true = get_groundtruth(gt_file)
        X_train, y_train = create_sub_sequences(train_dataset, window_size=opt.window_size, to_tensor=True)
        X_val, y_val = create_sub_sequences(val_dataset, window_size=opt.window_size, to_tensor=True)
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
        # Define dimensions
        input_dim = train_dataset[0][0].shape[-1]
        hidden_dim = opt.hidden_dim
        output_dim = opt.out_dim

        logging.info('Creating LSTM model...')
        model = LSTMModel(input_dim, hidden_dim, output_dim)
        # model = TransAm()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        trainer = Trainer(model, optimizer, criterion, trial)
        logging.info('Training model...')
        print(f"Jitter is {opt.do_jitter} and Mixup is {opt.do_mixup}")
        loss_list = []
        early_stop_count = 0
        print_every = 5
        best_val_loss = np.inf
        best_bias = np.inf
        for epoch in range(1, opt.epochs + 1):
            train_loss = trainer.train_one_epoch(train_loader, do_jitter=opt.do_jitter, 
                do_mixup=opt.do_mixup)
            val_loss = trainer.evaluate(val_loader)
            loss_list.append({"train_loss":train_loss, "val_loss":val_loss})
            if val_loss < best_val_loss:
                print(f"Better model found at epoch {epoch} with validation loss {val_loss:.4f}. Saving model...")
                best_val_loss = val_loss
                trainer.save(f"sandbox/new_saved/best_model_{trial}.pth")
                early_stop_count = 0
            else:
                early_stop_count += 1  # Increase count if validation loss didn't decrease
                print(f"Validation curve not improved for {early_stop_count} epochs")
                if early_stop_count == 15:
                    # print(f'Validation loss has not improved for {self.early_stop_count} epochs, stopping training. Talk about high standards!')
                    print("Breaking")
                    break
        df = pd.DataFrame(loss_list)
        print("Logging losses")
        df.to_csv(f"sandbox/new_saved/loss_log_{trial}.csv")
        trainer.plot_losses(df)
        model = trainer.load(f"sandbox/new_saved/best_model_{trial}.pth")
        X_gen = trainer.generate_predictions(model, test_dataset, 169-opt.window_size)
        X_gen = torch.cat([test_dataset, X_gen], dim=1)
        y_pred = inverse_transform_data(X_gen.numpy(), test_scaler)
        # y_pred = X_gen.numpy()
        plot_pred(y_pred, y_true, f'sandbox/new_saved/plot_pred_{trial}.png')
        np.save(f'sandbox/new_saved/pred_{trial}.npy', y_pred[:, -1, 0])
        metric_calculator = Metrics(y_true, all_test_dataset[:, 168, 0])
        metrics = metric_calculator()
        print(f'Imputation metrcis:{metrics}')
        metric_calculator = Metrics(y_true, y_pred[:, 168, 0])
        metrics = metric_calculator()
        print(f'Forecast metrics:{metrics}')
        
        metric_list.append(metrics)
    df = pd.DataFrame(metric_list)
    df.to_csv('all_trials.csv', index=False)
    mean_bias = np.mean(metrics["bias"])
    mean_rbias = np.mean(metrics["relative_bias"])
    mean_rmse = np.mean(metrics["rmse"])
    print(f"mean bias:{mean_bias}\nmean_relative_bias:{mean_rbias}\nmean_rmse:{mean_rmse}")
if __name__ == '__main__':
    opt = parse_option()
    main(opt)