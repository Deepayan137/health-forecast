import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pickle
# Correcting file paths and loading the true trajectories and the imputed training data
true_csv_path = 'sandbox/data/phds/true_training1.csv'
imputed_training_path = 'sandbox/new_saved/imputed_data_1.pkl'

# Load the true trajectories
true_df = pd.read_csv(true_csv_path, delimiter=';')

# Extract the log10VL values for each patient
true_trajectories = []
for id in true_df['Id'].unique():
    patient_data = true_df[true_df['Id'] == id]['log10VL'].values
    true_trajectories.append(patient_data)

# Convert list to numpy array
true_trajectories = np.array(true_trajectories)

# Load the imputed training data

with open(imputed_training_path, 'rb') as f:
    imputed_training, _, _ = pickle.load(f)

# Plotting the true and imputed trajectories
plt.figure(figsize=(10, 6))

# Plotting true trajectories
for patient_data in true_trajectories:
    plt.plot(patient_data, color='blue', alpha=0.3, label='True Trajectory')

# Plotting imputed training data
for i in range(imputed_training.shape[0]):
    plt.plot(imputed_training[i, :, 0], color='red', alpha=0.3, label='Imputed Training Data')

# Adding legends
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.title('True Trajectories vs Imputed Training Data')
plt.xlabel('Time (Days)')
plt.ylabel('log10VL')
plt.show()
plt.savefig('imputed_vs_training.png')
