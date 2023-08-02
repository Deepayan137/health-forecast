import pdb
import numpy as np
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


from sandbox.data_utils import *
from sandbox.utils import *
from sandbox.opts import parse_option
from sandbox.model import LSTMModel, TransAm
from sandbox.trainer import Trainer
from sandbox.impute import impute_fn

def parse_option():
    parser = argparse.ArgumentParser("argument for training")
    parser.add_argument("--csv", type=str, default="sandbox/data/simulated_data_4reservoirpy.csv")
    parser.add_argument("--err", type=int, default=1)
    parser.add_argument("--num_trials", type=int, default=2)
    parser.add_argument('--reduce_n', action='store_true', help='Reduce N (remove patient records)')
    parser.add_argument('--replace_t', action='store_true', help='Replace T with NA (remove data on specific days and replace with NA)')
    parser.add_argument("--rr_n", type=float, default=0.9)
    parser.add_argument("--rr_t", type=float, default=0.1)
    parser.add_argument("--record_every", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="simulated")
    parser.add_argument("--imp_method", type=str, default="saits")
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--forecast", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--out_dim", type=int, default=5)
    parser.add_argument("--lstm_epochs", type=int, default=10)
    parser.add_argument("--seed_timesteps", type=int, default=10)
    opt = parser.parse_args()
    return opt

def train_model(model, criterion, optimizer, train_loader, num_epochs=100):
    # criterion = nn.MSELoss()  # or nn.L1Loss() for MAE
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in trange(num_epochs):
        for batch in train_loader:
            X_train_batch, y_train_batch = batch
            
            optimizer.zero_grad()
            output = model(X_train_batch)
            loss = criterion(output, y_train_batch[:, -1, :].squeeze())
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch: {epoch+1}, Loss: {loss.item()}')

    return model


def generate_predictions(model, seed_input, num_generations):
    model.eval()
    current_input = seed_input
    predictions = []

    with torch.no_grad():
        for _ in range(num_generations):
            output = model(current_input)
            predictions.append(output)
            current_input = torch.cat((current_input[:, 1:, :], output.unsqueeze(1)), dim=1)

    return torch.stack(predictions, dim=1)

def main(opt):
    # Set up logger
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%d-%m-%Y_%H:%M:%S")
    imp_method = opt.imp_method
    dataset = opt.dataset
    exp_name = f"{imp_method}_{dataset}_{formatted_datetime}"
    log_path = os.path.join("sandbox", "new_experiments", exp_name)
    os.makedirs(log_path)
    args_dict = vars(opt)
    # Save the dictionary to a json file
    with open(os.path.join(log_path, 'config.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)
    logging = setup_logger(log_path)
    logging.info('Started')
    mse_list, mae_list = [], []
    seeds = [0]
    trials = range(opt.num_trials)
    seeds = seeds[:len(trials)]
    for seed, trial in zip(seeds, trials):
        logging.info(f'Seed...{seed}')
        logging.info('Creating dataset...')
        csv_path = opt.csv
        X = create_dataset(csv_path, opt)

        logging.info('Splitting data...')
        X_train, X_test = split_train_test_data(X)
        data_processor = DataProcessor()
        if opt.reduce_n:
            logging.info('Reducing N...')
            logging.info(f'Reduction Ratio...{opt.rr_n}')
            X_train = data_processor.reduce_N(X_train, reduction_ratio=opt.rr_n, seed=seed)
            logging.info(f'Number of patients left...{X_train.shape[0]}')
            X_orig = deepcopy(X_train)
        if opt.replace_t:
            logging.info('Replacing T...')
            logging.info(f'Reduction Ratio...{opt.rr_t}')
            X_train, indices = data_processor.replace_T_with_na(X_train, record_every=opt.record_every, reduction_ratio=opt.rr_t, seed=seed)
            logging.info(f'Time steps remaining...{len(indices)}')
            X_missing = deepcopy(X_train)
        logging.info('Rescaling data...')
        X_train, train_scaler = rescale(X_train)
        X_test, test_scaler = rescale(X_test)

        # Perform data imputation
        if opt.replace_t:
            X_train, saits = impute_fn(X_train)
            # X_filled = deepcopy(inverse_transform_data(X_train, train_scaler))
        # plot_data(X_orig, X_missing, X_filled, log_path)
        window_size = opt.window_size
        forecast = opt.forecast
        X_train, y_train = create_sub_sequences(X_train, window_size=window_size)
        X_test, y_test = to_forecasting(X_test, forecast=forecast)

        # Cast data to PyTorch tensors
        # X_train = np.nan_to_num(X_train)
        X_train = torch.tensor(X_train).float()
        # y_train = np.nan_to_num(y_train)
        y_train = torch.tensor(y_train).float()
        # mask = torch.tensor(mask).bool()

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

        # Define dimensions
        input_dim = X_train.shape[-1]
        hidden_dim = opt.hidden_dim
        output_dim = opt.out_dim

        logging.info('Creating LSTM model...')
        model = LSTMModel(input_dim, hidden_dim, output_dim)
        # criterion = MaskedMSELoss()  # or nn.L1Loss() for MAE
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        trainer = Trainer(model, optimizer, criterion, trial)
        # lstm = TransAm()
        logging.info('Training model...')
        # model = train_model(model, criterion, optimizer, train_loader, num_epochs=opt.lstm_epochs)
        trainer.train_epochs(train_loader, train_loader, 20)
        logging.info('Generating predictions...')
        seed_timesteps = opt.seed_timesteps
        num_generations = y_test.shape[1] - seed_timesteps

        warming_inputs = X_test[:, :seed_timesteps, :]
        #############
        if opt.replace_t:
            warming_inputs, indices_to_keep = data_processor.replace_T_with_na(warming_inputs, record_every=3)
            logging.info(f"Time instances recorded in Test ... {len(indices_to_keep)}")
            # Initialize X_temp with nan values
            X_temp = np.full((warming_inputs.shape[0], 169, 2), np.nan)
            # Create a mask of False values
            for idx in indices_to_keep:
                X_temp[:, idx] = warming_inputs[:, idx]

            dataset = {"X": X_temp}
            
            warming_inputs = saits.impute(dataset)
            warming_inputs = warming_inputs[:, :seed_timesteps, :]
        ####################
        # warming_inputs = np.nan_to_num(warming_inputs)n
        warming_inputs = torch.tensor(warming_inputs).float()
        # model = trainer.load(f'sandbox/saved/best_model_{trial}.pth')
        X_gen = trainer.generate_predictions(warming_inputs, num_generations)

        y_test_orig = inverse_transform_data(y_test[:, (seed_timesteps):, :], test_scaler)
        y_pred = inverse_transform_data(X_gen, test_scaler)

        mse = calculate_error_metric(y_test_orig, y_pred, metric='mse')
        mae = calculate_error_metric(y_test_orig, y_pred, metric='mae')
        write_predictions(y_test_orig, y_pred, log_path, trial=trial)
        logging.info('MSE: {}'.format(mse))
        logging.info('MAE: {}'.format(mae))
        plot_test_pred_data(y_test_orig, y_pred, log_path, trial=trial)
        mse_list.append(mse)
        mae_list.append(mae)
    logging.info(f'Mean MSE: {np.mean(mse_list)}')
    logging.info(f'Mean MAE: {np.mean(mae_list)}')

if __name__ == "__main__":
    opt = parse_option()
    main(opt)


# python -m sandbox.main --err 0 --reduce_n --num_trials 5 --seed_timesteps 10 --record_every 7 --rr_n 0.9 --out_dim 2 --replace_t

