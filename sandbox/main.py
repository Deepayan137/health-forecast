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
from sandbox.utils import plot_data, plot_pred, setup_logger
from sandbox.models import LSTMModel, Seq2Seq, Seq2SeqAttn
from sandbox.modules import LSTMTrainer, Seq2SeqTrainer
from sandbox.impute import impute_fn
from sandbox.loss import MaskedMSELoss
from sandbox.metrics import Metrics
from sandbox.dataset import TimeSeriesDataset

torch.manual_seed(42)  #

def parse_option():
    parser = argparse.ArgumentParser("argument for training")
    parser.add_argument("--root", type=str, default="sandbox/data/phds")
    parser.add_argument("--save_dir", type=str, default="sandbox/new_saved")
    parser.add_argument("--exp_name", type=str, default="trial1")
    parser.add_argument("--model_name", type=str, default="lstm")
    parser.add_argument("--seed_ts", type=int, default=21)
    parser.add_argument("--forecast", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--out_dim", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--do_jitter", action='store_true')
    parser.add_argument("--do_mixup", action='store_true')
    parser.add_argument("--no_impute", action='store_true')
    parser.add_argument("--tf_ratio", type=float, default=1.0)
    parser.add_argument("--tf_decay", type=float, default=0.999)
    opt = parser.parse_args()
    return opt

def main(opt):
    # Set up logger
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%d-%m-%Y_%H:%M:%S")
    exp_name = f"{opt.exp_name}"
    # log_path = os.path.join("sandbox", "experiments", exp_name)
    # os.makedirs(log_path)
    args_dict = vars(opt)
    # Save the dictionary to a json file
    # with open(os.path.join(log_path, 'config.json'), 'w') as f:
    #     json.dump(args_dict, f, indent=4)
    # logging = setup_logger(log_path)
    # logging.info('Started')
    trials = range(1, 90)
    metric_list = []
    for trial in trials:
        csv_path = os.path.join(opt.root, f'training_{trial}.csv')
        imp_path = f"{opt.save_dir}/imputed_data_{trial}.pkl"
        dataset = read_csv_with_missing_val(csv_path)
        if not opt.no_impute:
            if not os.path.exists(imp_path):
                # dataset, train_scaler = rescale(dataset)
                dataset, saits = impute_fn(dataset)
                print(f"Writing imputed data to {imp_path}")
                with open(imp_path, 'wb') as f:
                    pickle.dump((dataset, saits), f)
            else:
                print(f"Loading imputed data from {imp_path}")
                with open(imp_path, 'rb') as f:
                    (dataset, saits) = pickle.load(f)
        else:
            dataset = np.nan_to_num(dataset)
        train_dataset, val_dataset = train_test_split(dataset, shuffle=True, train_size=0.9)
        plot_data(val_dataset, f'{opt.save_dir}')
        test_csv_path = os.path.join(opt.root, f'validation_{trial}.csv')
        test_dataset = read_csv_with_missing_val(test_csv_path)
        # test_dataset, test_scaler = rescale(test_dataset)
        if not opt.no_impute:
            dataset = {"X": test_dataset}
            all_test_dataset = saits.impute(dataset)
            seed_timesteps = 21
            test_dataset = all_test_dataset[:, :seed_timesteps, :]
        else:
            test_dataset = np.nan_to_num(test_dataset)
        
        test_dataset = torch.tensor(test_dataset).float()
        gt_file = os.path.join(opt.root, f'truesetpoint_{trial}.csv')
        y_true = get_groundtruth(gt_file)
        
        train_dataset = TimeSeriesDataset(train_dataset, model_name=opt.model_name)
        val_dataset = TimeSeriesDataset(val_dataset, model_name=opt.model_name)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # Define dimensions
        input_dim = 2
        hidden_dim = opt.hidden_dim
        output_dim = opt.out_dim

        logging.info('Creating LSTM model...')
        if opt.model_name == 'lstm':
            model = LSTMModel(input_dim, hidden_dim, output_dim)
        elif opt.model_name == "seq2seq":
            model = Seq2Seq(input_dim, hidden_dim, output_dim)
        elif opt.model_name == "seq2seq_attn":
            model = Seq2SeqAttn(input_dim, hidden_dim, output_dim)
        # model = TransAm()
        if not opt.no_impute:
            criterion = nn.MSELoss()
        else:
            criterion = MaskedMSELoss()
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
        if opt.model_name == 'lstm':
            trainer = LSTMTrainer(model, optimizer, criterion, opt.seed_ts, trial)
        else:
            trainer = Seq2SeqTrainer(model, optimizer, criterion, opt.seed_ts, trial,
                opt.tf_ratio, opt.tf_decay)
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
                trainer.save(f"{opt.save_dir}/best_{opt.model_name}_{trial}.pth")
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
        df.to_csv(f"{opt.save_dir}/loss_log_{trial}.csv")
        trainer.plot_losses(df, opt.save_dir)
        model = trainer.load(f"{opt.save_dir}/best_{opt.model_name}_{trial}.pth")
        X_gen = trainer.generate_predictions(test_dataset, model, 169-opt.seed_ts)
        X_gen = torch.cat([test_dataset, X_gen.cpu().detach()], dim=1)
        # y_pred = inverse_transform_data(X_gen.numpy(), test_scaler)
        y_pred = X_gen.numpy()
        plot_pred(y_pred, y_true, f'{opt.save_dir}/plot_pred_{trial}.png')
        np.save(f'{opt.save_dir}/pred_{trial}.npy', y_pred[:, -1, 0])
        if not opt.no_impute:
            metric_calculator = Metrics(y_true, all_test_dataset[:, 168, 0])
            metrics = metric_calculator()
            print(f'Imputation metrcis:{metrics}')
        metric_calculator = Metrics(y_true, y_pred[:, 168, 0])
        metrics = metric_calculator()
        print(f'Forecast metrics:{metrics}')
        
        metric_list.append(metrics)
        df = pd.DataFrame(metric_list)
        df.to_csv(f'{exp_name}.csv', index=False)
    mean_bias = np.mean(df["bias"])
    mean_rbias = np.mean(df["relative_bias"])
    mean_rmse = np.mean(df["rmse"])
    print(f"mean bias:{mean_bias}\nmean_relative_bias:{mean_rbias}\nmean_rmse:{mean_rmse}")
if __name__ == '__main__':
    opt = parse_option()
    main(opt)