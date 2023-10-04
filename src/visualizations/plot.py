import os
import pdb
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_true_vs_pred(true_path, predicted_path, opt):
    # Loop over each patient
    true_df = pd.read_csv(true_path, delimiter=';')
    predicted_trajectories = np.load(predicted_path)
    plt.figure(figsize=(10,6))
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
    plt.savefig(f'{opt.save_dir}/{opt.name}')


def parse_option():
    parser = argparse.ArgumentParser("argument for training")
    parser.add_argument("--root", type=str, default="data/phds")
    parser.add_argument("--save_dir", type=str, default="results/plots")
    parser.add_argument("--name", type=str)
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_option()
    true_path = f'{opt.root}/true_validation1.csv'
    predict_dir = os.path.dirname(opt.save_dir)
    predicted_path = f'{predict_dir}/predictions/ode_predictions1.npy'
    name = opt.name
    plot_true_vs_pred(true_path, predicted_path, opt)
