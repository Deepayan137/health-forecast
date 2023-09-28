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

from sandbox.data_utils import read_csv_with_missing_val, read_csv,\
 get_groundtruth, create_sub_sequences, rescale, inverse_transform_data, CustomScaler, \
 get_groundtruth_modified
from sandbox.utils import plot_data, plot_pred, setup_logger, plot_imputed_data
from sandbox.models import LSTMModel, LSTMClassifier, Seq2Seq, Seq2SeqAttn, EnhancedLSTMModel
from sandbox.modules import LSTMTrainer, Seq2SeqTrainer, BiasedLSTMTrainer
from sandbox.impute import impute_missing_values, impute_test_data
from sandbox.loss import MaskedMSELoss
from sandbox.metrics import Metrics, Metrics_single
from sandbox.dataset import TimeSeriesDataset

torch.manual_seed(43)  #

def parse_option():
    parser = argparse.ArgumentParser("argument for training")
    parser.add_argument("--root", type=str, default="sandbox/data/phds")
    parser.add_argument("--save_dir", type=str, default="sandbox/new_saved")
    parser.add_argument("--log_dir", type=str, default="sandbox/logs")
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--log_all", action='store_true')
    parser.add_argument("--model_name", type=str, default="lstm")
    parser.add_argument("--seed_ts", type=int, default=21)
    parser.add_argument("--forecast", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--out_dim", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--do_jitter", action='store_true')
    parser.add_argument("--do_mixup", action='store_true')
    parser.add_argument("--use_ode", action='store_true')
    parser.add_argument("--pretrain", action='store_true')
    parser.add_argument("--pretrain_epochs", type=int, default=20)
    parser.add_argument("--tf_ratio", type=float, default=1.0)
    parser.add_argument("--tf_decay", type=float, default=0.999)
    opt = parser.parse_args()
    return opt

def train(opt, trainer, train_loader, val_loader, trial, pretrain=False):
    logging.info('Training model...')
    print(f"Jitter is {opt.do_jitter} and Mixup is {opt.do_mixup}")
    loss_list = []
    early_stop_count = 0
    print_every = 5
    best_val_loss = np.inf
    best_bias = np.inf
    for epoch in range(1, opt.epochs + 1):
        train_loss = trainer.train_one_epoch(train_loader, do_jitter=opt.do_jitter, 
            do_mixup=opt.do_mixup, pretrain=pretrain)
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
            if early_stop_count == 9:
                print("Breaking")
                break
    print("Finished")
    if opt.log_all:
        df = pd.DataFrame(loss_list)
        print("Logging losses")
        if pretrain:
            df.to_csv(f"{opt.save_dir}/pretrain_loss_log_{trial}.csv")
        else:
            df.to_csv(f"{opt.save_dir}/loss_log_{trial}.csv")
        trainer.plot_losses(df, opt.save_dir)


def main(opt):
    # Set up logger
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%d-%m-%Y_%H:%M:%S")
    exp_name = f"{opt.exp_name}"
    logging.info('Started')
    trials = range(1, 2)
    metric_list = []
    for trial in trials:
        if opt.use_ode:
            csv_path = os.path.join(opt.root, f'pred_training_1.csv')
            dataset, labels = read_csv(csv_path)
        else:
            csv_path = os.path.join(opt.root, f'training_{trial}.csv')
            dataset, labels = read_csv_with_missing_val(csv_path)
            imp_path = f"{opt.save_dir}/imputed_data_{trial}.pkl"
            if not os.path.exists(imp_path):
                dataset, saits0, saits1 = impute_missing_values(dataset, labels, smooth=True)
                print(f"Writing imputed data to {imp_path}")
                with open(imp_path, 'wb') as f:
                    pickle.dump((dataset, saits0, saits1), f)
            else:
                print(f"Loading imputed data from {imp_path}")
                with open(imp_path, 'rb') as f:
                    (dataset, saits0, saits1) = pickle.load(f)

        scaler = CustomScaler()
        dataset = scaler.transform(dataset)
        train_dataset, val_dataset, train_labels, val_labels = train_test_split(dataset, labels, shuffle=True, train_size=0.9, stratify=labels)
        seed_timesteps = 25
        if opt.use_ode:
            test_csv_path = os.path.join(opt.root, f'pred_validation_1.csv')
            test_dataset, test_labels = read_csv(test_csv_path)
            test_dataset = scaler.transform(test_dataset)
            test_dataset = test_dataset[:, :seed_timesteps, :]
        else:
            test_csv_path = os.path.join(opt.root, f'validation_{trial}.csv')
            test_dataset, test_labels = read_csv_with_missing_val(test_csv_path)
            test_dataset = impute_test_data(test_dataset, test_labels, saits0, saits1, smooth=False)
            test_dataset = scaler.transform(test_dataset)
            test_dataset = test_dataset[:, :seed_timesteps, :]
        test_dataset, test_labels = torch.tensor(test_dataset).float(),\
        torch.tensor(test_labels).float()
        # gt_file = os.path.join(opt.root, f'truesetpoint_{trial}.csv')
        gt_file = os.path.join(opt.root, f'true_validation{trial}.csv')
        y_true = get_groundtruth_modified(gt_file)
        if opt.pretrain:
            synth_path = os.path.join(opt.save_dir, f"synth_data_{trial}.pkl")
            print(f"Loading imputed data from {synth_path}")
            with open(synth_path, 'rb') as f:
                pretrain_dataset = pickle.load(f)
            print(f"Pretrain dataset size:{pretrain_dataset.shape}")
            pretrain_dataset = TimeSeriesDataset(pretrain_dataset, model_name=opt.model_name)
            pretrain_loader = DataLoader(pretrain_dataset, batch_size=32, shuffle=True)
        
        train_dataset = TimeSeriesDataset(train_dataset, train_labels)
        val_dataset = TimeSeriesDataset(val_dataset, val_labels)
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
        cls_model = LSTMClassifier(input_dim, hidden_dim)
        forecast_criterion = nn.MSELoss()
        cls_criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
        if opt.model_name == 'lstm':
            trainer = BiasedLSTMTrainer(opt, model, cls_model, optimizer, forecast_criterion, 
                cls_criterion, opt.seed_ts, trial)
        else:
            trainer = Seq2SeqTrainer(opt, model, optimizer, criterion, opt.seed_ts, trial,
                opt.tf_ratio, opt.tf_decay)
        if opt.pretrain:
            train(opt, trainer, pretrain_loader, val_loader, trial, pretrain=True)
        train(opt, trainer, train_loader, val_loader, trial)
        model = trainer.load(f"{opt.save_dir}/best_{opt.model_name}_{trial}.pth")
        X_gen = trainer.generate_test_predictions(test_dataset, test_labels, model, 169)
        y_pred = X_gen.detach().cpu().numpy()
        y_pred = scaler.inverse_transform(y_pred)
        if opt.use_ode:
            np.save('final/lstm_predictions_ode.npy', y_pred)
        else:
            np.save('final/lstm_predictions_raw.npy', y_pred)
        # select only required days
        selected_days = [30, 60, 90, 120, 150, 168]
        metric_calculator = Metrics(y_true, y_pred[:, selected_days, 0])
        metrics = metric_calculator()
        df = pd.DataFrame(metrics)
        if opt.use_ode:
            df.to_csv('final/lstm_metrics_every_month_ode.csv')
        else:
            df.to_csv('final/lstm_metrics_every_month_raw.csv')
        print(f'Forecast metrics:{metrics}')
    # mean_bias = np.mean(df["bias"])
    # mean_rbias = np.mean(df["relative_bias"])
    # mean_rmse = np.mean(df["rmse"])

if __name__ == '__main__':
    opt = parse_option()
    main(opt)