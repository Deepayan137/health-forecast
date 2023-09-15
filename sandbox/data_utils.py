import os
import argparse
import random
import torch
import numpy as np
import pandas as pd

import pdb
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class CustomScaler:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.original_shape = None

    def transform(self, data):
        """
        Scale data using MinMaxScaler along the last dimension.

        Args:
        - data (np.ndarray): A 3D numpy array of shape (N, T, D).

        Returns:
        - np.ndarray: Scaled data of the same shape.
        """
        N, T, D = data.shape
        self.original_shape = (N, T, D)
        
        # Reshape data to be 2D
        data_reshaped = data.reshape(-1, D)
        
        # Fit and transform using the scaler
        data_scaled = self.scaler.fit_transform(data_reshaped)
        
        # Reshape back to original shape
        data_scaled = data_scaled.reshape(N, T, D)
        
        return data_scaled

    def inverse_transform(self, scaled_data):
        """
        Convert the scaled data back to its original scale.

        Args:
        - scaled_data (np.ndarray): Scaled data of shape (N, T, D).

        Returns:
        - np.ndarray: Data reverted to its original scale.
        """
        _, T, D = self.original_shape
        
        # Reshape scaled data to be 2D
        scaled_data_reshaped = scaled_data.reshape(-1, D)
        
        # Use the scaler's inverse_transform method
        data_original = self.scaler.inverse_transform(scaled_data_reshaped)
        
        # Reshape back to original shape
        data_original = data_original.reshape(-1, T, D)
        
        return data_original

def format_data(dataset):
    N, T, D = dataset.shape
    X = dataset[:, :T-1]
    y = dataset[:, 1:T]
    return X, y

def create_input_sequences(df, opt):
    # Extract the T, I, P, E, and V columns from the DataFrame
    if opt.err == 0:
        # T = df['T'].values
        # I = df['I'].values
        # P = df['P'].values
        E = df['E'].values
        V = df['V'].values
    else:
        # T = df['Terr'].values
        # I = df['Ierr'].values
        # P = df['Perr'].values
        E = df['Eerr'].values
        V = df['Verr'].values
    input_sequence = np.vstack((E, V))
    # input_sequence = np.vstack((E, V))
    return input_sequence.transpose()

def create_dataset(csv, opt):
    df = pd.read_csv(csv, delimiter=';')
    # get the unique ids
    unique_ids = df['Id'].unique()
    unique_groups = df["Group"].unique()
    # loop over the unique ids and create a csv file for each
    X = []
    for i, uid in enumerate(unique_ids):
        sub_df = df[df['Id'] == uid]
        x = create_input_sequences(sub_df, opt)
        X.append(x)
    X = np.array(X)
    return X

def read_csv(csv_path):
    df = pd.read_csv(csv_path, delimiter=',')
    df['Time'] = df['Time'].astype(int)
    data = np.zeros((100, 169, 1))
    labels = []
    for i, id in enumerate(df['Id'].unique()):
        # This is where the magic happens, darling! Filter the rows for the current patient id
        patient_data = df[df['Id'] == id]
        if i < 50:
            labels.append(0)
        else:
            labels.append(1)
        # Put that data in the right place
        data[i, patient_data['Time'].values] = patient_data[['V']].values

    # Voila! We're done. Now, go out there and own it!
    return data, labels


def read_csv_with_missing_val(csv_path):
    df = pd.read_csv(csv_path, delimiter=';')
    # Work it! Convert "Time" column to int, 'cause fashionably late is not our style today.
    df['Time'] = df['Time'].astype(int)
    # Don't mesh things up, initialize a 3D array filled with NaNs
    # if 'training' in csv_path:
    data = np.full((100, 169, 2), np.nan)
    # elif 'validation' in csv_path:
    #     data = np.full((100, 21, 2), np.nan)
    # Now let's strut our stuff on the runway, I mean, for each unique id in the dataset
    labels = []
    for i, id in enumerate(df['Id'].unique()):
        # This is where the magic happens, darling! Filter the rows for the current patient id
        patient_data = df[df['Id'] == id]
        group_id = np.unique(patient_data['Group'].values).item()
        if group_id == 'Group1':
            labels.append(0)
        else:
            labels.append(1)
        # Put that data in the right place
            data[i, patient_data['Time'].values] = patient_data[['V']].values

    # Voila! We're done. Now, go out there and own it!
    return data, labels

def get_groundtruth(csv_path):
    df = pd.read_csv(csv_path, delimiter=';')
    data = np.zeros((100), dtype=np.float32)
    for i, id in enumerate(df['Id'].unique()):
        data[i] = df[df['Id'] == id]['V'].values
    return data

def split_train_test_data(X):
    # Get the number of patients and time instances
    np.random.seed(42)
    num_patients, num_time_instances, num_variables = X.shape
    group1_indices = np.arange(0, 500)
    group2_indices = np.arange(500, 1000)
    test_group1_indices = np.random.choice(group1_indices, size=100, replace=False)
    test_group2_indices = np.random.choice(group2_indices, size=100, replace=False)
    train_group1_indices = np.setdiff1d(group1_indices, test_group1_indices)
    train_group2_indices = np.setdiff1d(group2_indices, test_group2_indices)
    train_indices = np.concatenate([train_group1_indices, train_group2_indices])
    test_indices = np.concatenate([test_group1_indices, test_group2_indices])
    X_train, X_test = X[train_indices], X[test_indices]
    return X_train, X_test

def rescale(matrix):
    # Create a MinMaxScaler object
    scaler = StandardScaler()
    N, T, D = matrix.shape
    # Reshape the matrix to 2D (assuming the dimensions are N, T, D)
    X = matrix.reshape(N * T, D)
    
    rescaled_matrix = scaler.fit_transform(X)
    rescaled_matrix = rescaled_matrix.reshape(N, T, D)
    return rescaled_matrix, scaler

def inverse_transform_data(normalized_matrix, scaler):
    # Reshape the matrix to 2D (N*T, D) for inverse transformation
    N, T, D = normalized_matrix.shape
    reshaped_matrix = np.reshape(normalized_matrix, (N * T, D))

    # Perform the inverse transformation using the provided scaler
    inverse_transformed_matrix = scaler.inverse_transform(reshaped_matrix)

    # Reshape the inverse transformed matrix back to the original shape
    inverse_transformed_matrix = np.reshape(inverse_transformed_matrix, (N, T, D))

    return inverse_transformed_matrix

def create_sub_sequences(X, window_size=20, to_tensor=True):
    """
    Create input sequences and corresponding targets for a given dataset and window size.

    Parameters:
    X: Input dataset of shape (N, T, D), where N is number of data points, 
       T is the number of time steps, and D is the feature dimension.
    window_size: Size of the window to use when creating the sequence.

    Returns:
    X_seq: Input sequences for the model, of shape (N, T - window_size, window_size, D).
    y_seq: Corresponding targets for each input sequence, of shape (N, T - window_size, D).
    """
    N, T, D = X.shape
    X_seq = np.empty((N, T - window_size, window_size, D))
    y_seq = np.empty((N, T - window_size, D))
    
    for i in range(T - window_size):
        X_seq[:, i, :, :] = X[:, i:i+window_size, :]
        y_seq[:, i, :] = X[:, i+window_size, :]
    X_seq, y_seq = X_seq.reshape(-1, window_size, D), y_seq.reshape(-1, 1, D)
    if to_tensor:
        X_seq, y_seq = torch.tensor(X_seq).float(), torch.tensor(y_seq).float()
    return X_seq, y_seq

def to_forecasting(X, forecast):
    """
    Generate input (x) and target (y) time series for forecasting.

    Args:
        X (ndarray): The input data of shape (num_samples, seq_length, num_features).
        forecast (int): The number of time steps to forecast ahead.

    Returns:
        x (ndarray): The input time series of shape (num_samples, seq_length - forecast, num_features).
        y (ndarray): The target forecasting time series of shape (num_samples, forecast, num_features).
    """
    num_samples, seq_length, num_features = X.shape

    # Determine the dimensions of the input (x) and target (y) time series
    input_length = seq_length - forecast
    # Generate input (x) and target (y) time series
    x = X[:, :input_length, :]
    y = X[:, forecast:seq_length, :]

    return x, y

class DataProcessor():
    def reduce_N(self, data, reduction_ratio=0.1, seed=0):
        np.random.seed(seed)
        num_patients = data.shape[0]
        num_patients_to_keep = int(num_patients * (1 - reduction_ratio))
        indices_to_keep = np.random.choice(num_patients, size=num_patients_to_keep, replace=False)
        return data[indices_to_keep]
    
    def replace_T_with_na(self, data, record_every, reduction_ratio=0.0, seed=0):
        np.random.seed(seed)
        num_time_steps = data.shape[1]
        indices_to_keep = np.arange(0, num_time_steps, record_every)
        if reduction_ratio > 0.:
            num_recording_days = indices_to_keep.shape[0]
            days_to_keep = int(num_recording_days * (1 - reduction_ratio))
            indices_to_keep = np.random.choice(indices_to_keep, size=days_to_keep, replace=False)
        # Calculate the non-kept indices
        non_kept_indices = np.setdiff1d(np.arange(num_time_steps), indices_to_keep)
        # Replace the data on non-kept days with NA
        data[:, non_kept_indices] = np.nan
        return data, indices_to_keep

class CustomScaler:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.original_shape = None

    def transform(self, data):
        """
        Scale data using MinMaxScaler along the last dimension.

        Args:
        - data (np.ndarray): A 3D numpy array of shape (N, T, D).

        Returns:
        - np.ndarray: Scaled data of the same shape.
        """
        N, T, D = data.shape
        self.original_shape = (N, T, D)
        
        # Reshape data to be 2D
        data_reshaped = data.reshape(-1, D)
        
        # Fit and transform using the scaler
        data_scaled = self.scaler.fit_transform(data_reshaped)
        
        # Reshape back to original shape
        data_scaled = data_scaled.reshape(N, T, D)
        
        return data_scaled

    def inverse_transform(self, scaled_data):
        """
        Convert the scaled data back to its original scale.

        Args:
        - scaled_data (np.ndarray): Scaled data of shape (N, T, D).

        Returns:
        - np.ndarray: Data reverted to its original scale.
        """
        _, T, D = self.original_shape
        
        # Reshape scaled data to be 2D
        scaled_data_reshaped = scaled_data.reshape(-1, D)
        
        # Use the scaler's inverse_transform method
        data_original = self.scaler.inverse_transform(scaled_data_reshaped)
        
        # Reshape back to original shape
        data_original = data_original.reshape(-1, T, D)
        
        return data_original
 