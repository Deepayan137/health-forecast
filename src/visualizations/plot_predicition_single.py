import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pickle
import pdb
from sandbox.data_utils import read_csv
# Load the true trajectories

def return_mean(path, delimiter=';'):
	if path.endswith('.csv'):
		data, _ = read_csv(path, delimiter=delimiter)
	elif path.endswith('.npy'):
		data = np.load(path)
	data1, data2 = data[:50, :, 0], data[50:, :, 0]
	mean1, mean2 = np.mean(data1, axis=0), np.mean(data2, axis=0)
	return mean1, mean2

true_path = 'sandbox/data/phds/true_validation1.csv'
true1, true2 = return_mean(true_path, delimiter=';')
# Load Ode prediction trajectory
ode_predicted_path = 'sandbox/data/phds/pred_validation_1.csv'
ode1_mean, ode2_mean = return_mean(ode_predicted_path, ',')

# Load Reservoir predictions
lstm_predicted_path_raw = 'results/predictions/lstm_predictions_raw_1.npy'
lstm_raw1_mean, lstm_raw2_mean = return_mean(lstm_predicted_path_raw)

# Load LSTM Trajectories
rsvr_predicted_path_raw = 'results/predictions/reservoir_predictions_raw_1.npy'
rsvr_raw1_mean, rsvr_raw2_mean = return_mean(rsvr_predicted_path_raw)

# Load Reservoir predictions
# lstm_predicted_path_ode = 'results/predictions/lstm_predictions_ode_1.npy'
# lstm_ode1_mean, lstm_ode2_mean = return_mean(lstm_predicted_path_ode)

# # Load LSTM Trajectories
# rsvr_predicted_path_ode = 'results/predictions/reservoir_predictions_ode_1.npy'
# rsvr_ode1_mean, rsvr_ode2_mean = return_mean(rsvr_predicted_path_ode)

plt.figure(figsize=(10,6))
plt.plot(true1, color='blue', alpha=1.0, label='True Trajectory Group 1')
plt.plot(true2, color='red', alpha=1.0, label='True Trajectory Group 2')

plt.plot(ode1_mean, color='green', alpha=1.0, label='ODE Trajectory Group 1')
plt.plot(ode2_mean, color='green', alpha=1.0, label='ODE Trajectory Group 2', linestyle='dashed')

plt.plot(rsvr_raw1_mean, color='orange', alpha=1.0, label='Reservoir Trajectory Group 1 with Raw Input')
plt.plot(rsvr_raw2_mean, color='orange', alpha=1.0, label='Reservoir Trajectory Group 2 with raw input', linestyle='dashed')

plt.plot(lstm_raw1_mean, color='purple', alpha=1.0, label='LSTM Trajectory Group 1 with Raw Input')
plt.plot(lstm_raw2_mean, color='purple', alpha=1.0, label='LSTM Trajectory Group 2 with Raw Input', linestyle='dashed')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='lower right', bbox_to_anchor=(1, 0))
plt.xlabel('Time (Days)')
plt.ylabel('log10VL')
plt.show()
plt.savefig('Predictions.png')
