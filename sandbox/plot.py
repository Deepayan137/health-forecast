import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the true trajectories
true_path = 'TrueTrajectories/true_validation1.csv'
true_df = pd.read_csv(true_path, delimiter=';')

# Load the predicted trajectories
predicted_path = 'final/lstm_predictions_raw.npy'
predicted_trajectories = np.load(predicted_path)

# Plotting the true and predicted trajectories
plt.figure(figsize=(10,6))

# Loop over each patient
for i in range(100):
    # Extract true trajectory for the patient
    patient_id = true_df['Id'].unique()[i]
    patient_data = true_df[true_df['Id'] == patient_id]['log10VL'].values
    
    # Define color and linestyle for the true trajectory
    if 'Group1' in patient_id:
        color = 'blue'  # Group 1 true trajectory
        label_true = 'Group 1 True' if i == 0 else ""
    else:
        color = 'red'  # Group 2 true trajectory
        label_true = 'Group 2 True' if i == 50 else ""
    
    plt.plot(patient_data, color=color, alpha=0.3, label=label_true)
    
    # Extract and plot predicted trajectory for the patient
    predicted_data = predicted_trajectories[i, :, 0]
    if i < 50:
        predicted_color = 'green'  # Group 1 predicted trajectory
        label_pred = 'Group 1 Predicted' if i == 0 else ""
    else:
        predicted_color = 'cyan'  # Group 2 predicted trajectory
        label_pred = 'Group 2 Predicted' if i == 50 else ""
    
    plt.plot(predicted_data, linestyle='dashed', color=predicted_color, alpha=0.3,label=label_pred)

plt.title('True and Predicted Trajectories')
plt.xlabel('Time (Days)')
plt.ylabel('log10VL')
plt.legend(loc='lower right')
plt.show()
plt.savefig('final/LSTM_Predictions_Raw.png')

