import os
import argparse
import joblib
import pickle
import numpy as np
import optuna
import reservoirpy as rpy
from optuna.storages import JournalStorage, JournalFileStorage
import time

# Assuming all required imports and data loading functions are here
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.observables import nrmse, rsquare
from sklearn.model_selection import train_test_split
from optuna.storages import JournalStorage, JournalFileStorage

from sandbox.data_utils import read_csv_with_missing_val,\
 get_groundtruth, format_data, CustomScaler, read_csv
from sandbox.reservoir.sk_node import SklearnNode
from sandbox.metrics import Metrics
optuna.logging.set_verbosity(optuna.logging.ERROR)

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="sandbox/data")
    parser.add_argument("--root", type=str, default="sandbox/data/phds")
    parser.add_argument("--log_dir", type=str, default="hp_logs/")
    parser.add_argument('--nb_trials', type=int, required=True)
    parser.add_argument('--study_name', type=str, required=True)
    parser.add_argument('--N', type=int, default=500)
    parser.add_argument('--iss', type=float, default=0.1)
    parser.add_argument('--sr', type=float, default=2.0)
    parser.add_argument('--lr', type=float, default=0.375)
    parser.add_argument('--ridge', type=float, default=2.1e-4)
    parser.add_argument('--nb_seeds', type=int, default=3)
    parser.add_argument('--n_process', type=int, default=10)
    parser.add_argument("--use_ode", action='store_true')
    args = parser.parse_args()
    return args

class TimeSeriesHyperParameterTuner:
    
    def __init__(self, opt, dataset_trainval, warming_inputs, y_true,
            n_trials, study_name, log_name, n_process):
        self.opt = opt
        self.n_trials = n_trials
        self.study_name = study_name
        self.log_name = log_name
        self.n_process = n_process
        ((self.X_train, self.y_train), (self.X_test, self.y_test)) = dataset_trainval
        self.warming_inputs = warming_inputs
        self.y_true = y_true
        self.sampler = optuna.samplers.RandomSampler()
        self.storage = JournalStorage(JournalFileStorage(os.path.join(self.opt.log_dir, 
            self.log_name)))
        
    def _objective(self, trial):
        # Record objective values for each trial
        rpy.verbosity(0)
        losses = []
        # Trial generated parameters (with log scale)
        self.opt.iss = trial.suggest_float("iss", 0.05, 0.20, log=True)
        # self.opt.sr = trial.suggest_float("sr", 2.2e-3, 1, log=True)
        # self.opt.lr = trial.suggest_float("lr", 0.3, 1, log=True)
        # self.opt.ridge = trial.suggest_float("ridge", 1e-7, 4e-3, log=True)
        # N = trial.suggest_int('N', 100, 500, 100)
        # ridge = trial.suggest_float("ridge", 1e-3, 1, log=True)
        for seed in range(self.opt.nb_seeds):
            reservoir = Reservoir(self.opt.N,
                                  sr=self.opt.sr,
                                  lr=self.opt.lr,
                                  input_scaling=self.opt.iss,
                                  seed=seed)
            
            # readout = Ridge(ridge=ridge)
            readout = SklearnNode(method="Ridge", alpha=self.opt.ridge)
            model = reservoir >> readout

            # Train and test your model
            predictions = model.fit(self.X_train, self.y_train).run(self.X_test)

            # Compute the desired metric(s)
            # loss = nrmse(self.y_test, predictions, norm_value=np.ptp(self.X_train))
            # losses.append(loss)
            warming_out = model.run(self.warming_inputs, reset=True)
            nb_generations = 148
            X_gen = np.zeros((all_test_dataset.shape[0], nb_generations, 1))
            y = np.array(warming_out)
            for t in range(nb_generations):  # generation
                y = np.array(model.run(y))
                y_last = y[:, -1, :]
                X_gen[:, t, :] = y_last
                y = np.concatenate((y[:, 1:, :], y_last[:, None, :]), axis=1)
            X_gen = np.concatenate((self.warming_inputs, X_gen), axis=1)
            y_pred = test_scaler.inverse_transform(X_gen)
            metric_calculator = Metrics(self.y_true, y_pred[:, -1, 0])
            metrics = metric_calculator()
            print(metrics)
            losses.append(metrics['rrmse'])
        return np.mean(losses)
    
    def _optimize_study(self, n_trials, process_idx):
        study = optuna.create_study(
            study_name=self.study_name,
            direction='minimize',
            storage=JournalStorage(JournalFileStorage(
                os.path.join(self.opt.log_dir, f"{self.log_name}_{process_idx}.log"))),
            sampler=self.sampler,
            load_if_exists=True
        )

        for i in range(n_trials):
            trial = study.ask()
            study.tell(trial, self._objective(trial))
    
    def run_search(self):
        n_trials_per_process = self.n_trials // self.n_process
        args_list = [(n_trials_per_process, idx) for idx in range(self.n_process)]

        # Parallelize the hyperparameter search across processes
        joblib.Parallel(n_jobs=self.n_process)(
            joblib.delayed(self._optimize_study)(*args) for args in args_list
        )
        
    @staticmethod
    def consolidate_logs(log_name, n_process, study_name, log_dir):
        # consolidated_storage = optuna.storages.RDBStorage("sqlite:///consolidated.db")
        consolidated_storage = JournalStorage(JournalFileStorage(os.path.join(log_dir, 
            f"consolidated_{study_name}.log")))
        consolidated_study = optuna.create_study(
            study_name=f"{study_name}_consolidated",
            storage=consolidated_storage,
            direction='minimize',
            load_if_exists=True
        )
        for i in range(n_process):
            current_log_name = f"{log_name}_{i}.log"
            storage = JournalStorage(JournalFileStorage(os.path.join(log_dir, current_log_name)))
            study = optuna.load_study(study_name=study_name, storage=storage)
            for trial in study.trials:
                consolidated_study.add_trial(trial)
        return consolidated_study


if __name__ == "__main__":
    # Data Preprocessing
    args = parse_option()
    num = 1
    if args.use_ode:
        csv_path = os.path.join(args.root, f'pred_training_1.csv')
        dataset = read_csv(csv_path)
    else:
        imp_path = f"{args.save_dir}/imputed_data_{num}.pkl"
        with open(imp_path, 'rb') as f:
            (dataset, saits) = pickle.load(f)

    # Scale the dataset
    scaler = CustomScaler()
    dataset = scaler.transform(dataset)

    train_dataset, val_dataset = train_test_split(dataset, shuffle=True, train_size=0.9)
    X_train, y_train = format_data(train_dataset)
    X_test, y_test = format_data(val_dataset)
    dataset_trainval = ((X_train, y_train), (X_test, y_test))

    gt_file = os.path.join(args.root, f'truesetpoint_{num}.csv')
    y_true = get_groundtruth(gt_file)
    if args.use_ode:
        test_csv_path = os.path.join(args.root, f'pred_validation_1.csv')
        all_test_dataset = read_csv(test_csv_path)
    else:
        test_csv_path = os.path.join(args.root, f'validation_{num}.csv')
        test_dataset = read_csv_with_missing_val(test_csv_path)
        dataset = {"X": test_dataset}
        all_test_dataset = saits.impute(dataset)
    # test_scaler = CustomScaler()
    all_test_dataset = scaler.transform(all_test_dataset)
    seed_timesteps = 21
    warming_inputs = all_test_dataset[:, :seed_timesteps, :]

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # Set the parameters for your tuner
    tuner = TimeSeriesHyperParameterTuner(
        args,
        dataset_trainval,
        warming_inputs,
        y_true,
        n_trials=args.nb_trials,
        study_name=f"{args.study_name}",
        log_name=f"optuna-journal_{args.study_name}.log",
        n_process=args.n_process
    )
    start_time = time.time()
    tuner.run_search()
    end_time = time.time()
    print(f"Optimization finished. Time taken: {end_time - start_time   } seconds.")
    consolidated_study = TimeSeriesHyperParameterTuner.consolidate_logs(f"optuna-journal_{args.study_name}.log", 
        args.n_process, args.study_name, args.log_dir)

