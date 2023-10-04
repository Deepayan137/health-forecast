import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_true_vs_pred(true_path, predicted_path, name=None):
    # Loop over each patient
    ode_trajectories = np.load(true_path)
    predicted_trajectories = np.load(predicted_path)
    plt.figure(figsize=(10,6))
    for i in range(100):
        # Extract true trajectory for the patient
        patient_data = ode_trajectories[i, :, 0]
        
        # Define color and linestyle for the true trajectory
        if i < 50:
            color = 'blue'  # Group 1 true trajectory
            label_true = 'Group 1 ODE' if i == 0 else ""
        else:
            color = 'red'  # Group 2 true trajectory
            label_true = 'Group 2 ODE' if i == 50 else ""
        
        plt.plot(patient_data, color=color, alpha=0.2, label=label_true)
        
        # Extract and plot predicted trajectory for the patient
        predicted_data = predicted_trajectories[i, :, 0]
        if i < 50:
            predicted_color = 'purple'  # Group 1 predicted trajectory
            label_pred = 'Group 1 Predicted' if i == 0 else ""
        else:
            predicted_color = 'cyan'  # Group 2 predicted trajectory
            label_pred = 'Group 2 Predicted' if i == 50 else ""
        
        plt.plot(predicted_data, color=predicted_color, alpha=1.0,label=label_pred)

    # plt.title('True and Predicted Trajectories')
    plt.xlabel('Time (Days)')
    plt.ylabel('log10VL')
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig(f'final/{name}')


if __name__ == "__main__":
    true_path = 'results/predictions/ode_predictions2.npy'
    predicted_path = 'results/predictions/lstm_predictions_ode_2.npy'
    name = 'ode_original_vs_lstm_pred_seed.png'
    plot_true_vs_pred(true_path, predicted_path, name)
