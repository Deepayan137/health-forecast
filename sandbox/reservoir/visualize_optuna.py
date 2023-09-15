import argparse

import optuna
from optuna.storages import JournalStorage, JournalFileStorage
from optuna.visualization import plot_slice
from optuna.visualization import plot_contour

parser = argparse.ArgumentParser()
parser.add_argument('--study_name', type=str, required=True)
args = parser.parse_args()

log_name = f"optuna-journal_{args.study_name}.log"
storage = JournalStorage(JournalFileStorage(log_name))
study = optuna.load_study(
    study_name = f'{args.study_name}',
    storage = storage
)

plot_slice(study)