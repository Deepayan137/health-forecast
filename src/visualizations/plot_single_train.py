import pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
# Load the true trajectories
true_path = 'sandbox/data/phds/true_training1.csv'
true_df = pd.read_csv(true_path, delimiter=';')

# Load simulated data
simulated_path = 'sandbox/data/phds/training_1.csv'
sim_df = pd.read_csv(simulated_path, delimiter=';')

# Load imputed data
imputed_training_path = 'sandbox/data/phds/imputed_data_1.pkl'
with open(imputed_training_path, 'rb') as f:
    imputed_training, _, _ = pickle.load(f)

# Load ODE
ode_predicted_path = 'sandbox/data/phds/pred_training_1.csv'
ode_df = pd.read_csv(ode_predicted_path, delimiter=',')

# Plotting the true and predicted trajectories
plt.figure(figsize=(10,6))

group0_id = 0
true_data_group0 = true_df[true_df['Id'] == 'Group1.1']['log10VL'].values
true_data_time = np.arange(true_data_group0.shape[0])
true_data_group0_tup = list(zip(true_data_time, true_data_group0))

sim_data_group0 = sim_df[sim_df['Id'] == 1]['log10VL'].values
sim_data_time = sim_df[sim_df['Id'] == 1]['Time'].values
sim_data_tup = list(zip(sim_data_time, sim_data_group0))

ode_pred_group0 = ode_df[ode_df['Id'] == 1]['V'].values
ode_pred_group0_tup = list(zip(true_data_time, ode_pred_group0))

imp_data0 = imputed_training[0, :, 0]
imp_time = np.arange(imp_data0.shape[0])

plt.plot(true_data_group0, color='blue', alpha=1.0, label='True Trajectory Group 1')
plt.scatter(*zip(*sim_data_tup), color='green', alpha=1.0, label='Simulated Data Group 1')
plt.plot(ode_pred_group0, color='orange', alpha=1.0, label='ODE Imputed Data Group 1')
plt.plot(imp_data0, color='red', alpha=1.0, label='SAITS Imputed Data Group 1')


true_data_group1 = true_df[true_df['Id'] == 'Group2.1']['log10VL'].values

sim_data_group1 = sim_df[sim_df['Id'] == 51]['log10VL'].values
sim_data_time = sim_df[sim_df['Id'] == 51]['Time'].values
sim_data_tup1 = list(zip(sim_data_time, sim_data_group1))
ode_pred_group1 = ode_df[ode_df['Id'] == 51]['V'].values
imp_data1 = imputed_training[51, :, 0]


plt.plot(true_data_group1, color='blue', alpha=1.0, label='True Trajectory Group 2', linestyle='dashed')
plt.scatter(*zip(*sim_data_tup1), color='green', alpha=1.0, label='Simulated Data Group 2', marker='x' )
plt.plot(ode_pred_group1, color='orange', alpha=1.0, label='ODE Imputed Data Group 2', linestyle='dashed')
plt.plot(imp_data1, color='red', alpha=1.0, label='SAITS Imputed Data Group 2', linestyle='dashed')


handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='lower right', bbox_to_anchor=(1, 0))


# plt.title('True Trajectories vs Imputed Training Data')
plt.xlabel('Time (Days)')
plt.ylabel('log10VL')
plt.show()
plt.savefig('new_imputed_vs_training.png')

