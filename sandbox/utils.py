import os
import logging
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd



def plot_data(X_orig, X_missing, X_filled, log_path):
    # Define the number of time instances
    num_time_instances = X_orig.shape[1]

    # Extract only the last column (-1) of the first data entry (0) for each array
    X_orig, X_missing, X_filled = X_orig[0, :, -1],\
     X_missing[0, :, -1], X_filled[0, :, -1]

    # Replace nan values with a constant
    # nan_replacement_value = -1  # Can be any other number or computed statistic
    # X_missing = np.nan_to_num(X_missing, nan=nan_replacement_value)
    # X_filled = np.nan_to_num(X_filled, nan=nan_replacement_value)

    # Create the plot
    plt.plot(range(num_time_instances), X_orig, color='blue', linewidth=2, label='X orig')
    plt.plot(range(num_time_instances), X_missing, color='red', linewidth=2, label='X missing')
    plt.plot(range(num_time_instances), X_filled, color='green', linewidth=2, label='X filled')

    # Set plot labels and title
    plt.xlabel('Time')
    plt.ylabel('Variable')
    plt.title('Original, Missing and Imputated Data')

    # Add legend
    plt.legend()

    # Save the plot to a file
    plt.savefig(os.path.join(log_path, "data_plot.png"))
    plt.close()  # Close the plot

def plot_pred(X_pred, gt_points, log_path):
    num_individuals, num_time_instances, num_variables = X_pred.shape
    
    # Create a color palette for test and pred plots
    pred_color_palette = plt.cm.get_cmap('Reds', 10+1)
    
    # Plot individual test curves
    for i in range(5):
        pred_individual_data = X_pred[i, :, 0]  # Select the last dimension for plotting
        plt.plot(range(num_time_instances), pred_individual_data, color=pred_color_palette(i), linewidth=0.5)
        plt.plot(num_time_instances, gt_points[i], color='blue')
    # Plot the mean test curve in bold blue
    mean_pred_data = np.mean(X_pred[:, :, 0], axis=0)
    plt.plot(range(num_time_instances), mean_pred_data, color='red', linewidth=2)
    # Set plot labels and title
    plt.xlabel('Time')
    plt.ylabel('Variable T')
    plt.title('Test and Pred Data')
    
    # Add legend
    plt.legend()
    
    # Show the plot
    # plt.show()
    plt.savefig(os.path.join(log_path))
    plt.close()

def plot_test_pred_data(X_test, X_pred, log_path, trial=0):
    # Get the number of individuals and time instances
    num_individuals, num_time_instances, num_variables = X_test.shape
    
    # Create a color palette for test and pred plots
    test_color_palette = plt.cm.get_cmap('Blues', 10+1)
    pred_color_palette = plt.cm.get_cmap('Reds', 10+1)
    
    # Plot individual test curves
    for i in range(5):
        test_individual_data = X_test[i, :, -1]  # Select the last dimension for plotting
        plt.plot(range(num_time_instances), test_individual_data, color=test_color_palette(i), linewidth=0.5)
    
    # Plot the mean test curve in bold blue
    mean_test_data = np.mean(X_test[:, :, -1], axis=0)
    plt.plot(range(num_time_instances), mean_test_data, color='blue', linewidth=2)
     
    # Plot individual pred curves
    for i in range(10):
        pred_individual_data = X_pred[i, :, -1]  # Select the last dimension for plotting
        plt.plot(range(num_time_instances), pred_individual_data, color=pred_color_palette(i), linewidth=0.5)
    
    # Plot the mean pred curve in bold red
    mean_pred_data = np.mean(X_pred[:, :, -1], axis=0)
    plt.plot(range(num_time_instances), mean_pred_data, color='red', linewidth=2, label=f'Pred Mean trial {trial}')
    # Set plot labels and title
    plt.xlabel('Time')
    plt.ylabel('Variable T')
    plt.title('Test and Pred Data')
    
    # Add legend
    plt.legend()
    
    # Show the plot
    # plt.show()
    plt.savefig(os.path.join(log_path, f"plot_{trial}.png"))
    plt.close()

def setup_logger(log_dir):
    # Create a custom logger
    logger = logging.getLogger(__name__)

    # Set the threshold for this logger to DEBUG
    logger.setLevel(logging.DEBUG)

    # Create handlers
    console_handler = logging.StreamHandler()  # Console handler
    file_handler = logging.FileHandler(os.path.join(log_dir, 'logfile.log'))  # File handler

    # Set level of handlers to DEBUG
    console_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.DEBUG)

    # Create formatters and add it to handlers
    format_str = '%(asctime)s - %(levelname)s - %(message)s'  # You can customize this line
    formatter = logging.Formatter(format_str)
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

def calculate_error_metric(y_true, y_pred, metric='mse'):
    """
    Function to calculate specified error metric.

    Parameters:
    y_true: Ground truth values.
    y_pred: Predicted values.
    metric: The error metric to calculate. One of 'mae', 'mse', 'rmse', 'r2'.

    Returns:
    The calculated error metric.
    """

    # Flatten the arrays for calculation
    y_true, y_pred = y_true[:, :, -1], y_pred[:, :, -1]
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    if metric.lower() == 'mae':
        return mean_absolute_error(y_true, y_pred)
    elif metric.lower() == 'mse':
        return mean_squared_error(y_true, y_pred)
    elif metric.lower() == 'rmse':
        return np.sqrt(mean_squared_error(y_true, y_pred))
    elif metric.lower() == 'r2':
        return r2_score(y_true, y_pred)
    else:
        raise ValueError(f"Unknown error metric: {metric}")


def write_predictions(y_true, y_pred, log_path, trial=0):
    y_true, y_pred = y_true[:, :, -1], y_pred[:, :, -1]
    y_true_mean = np.mean(y_true, axis=0)
    y_pred_mean = np.mean(y_pred, axis=0)
    # Create a dataframe
    df = pd.DataFrame({
        'y_true_mean': y_true_mean,
        'y_pred_mean': y_pred_mean
    })

    # Write to csv
    df.to_csv(os.path.join(log_path, f'output_{trial}.csv'), index=False)

