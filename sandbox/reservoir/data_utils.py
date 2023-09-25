import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def format_data(dataset):
    N, T, D = dataset.shape
    X = dataset[:, :T-1]
    y = dataset[:, 1:T]
    return X, y

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

def get_groundtruth_modified(csv_path):
    df = pd.read_csv(csv_path, delimiter=';')
    days_to_extract = [30, 60, 90, 120, 150, 168]
    num_patients = len(df['Id'].unique())
    data = np.zeros((num_patients, len(days_to_extract)), dtype=np.float32)
    
    for i, id in enumerate(df['Id'].unique()):
        patient_data = df[df['Id'] == id]
        for j, day in enumerate(days_to_extract):
            value_for_day = patient_data[patient_data['Time'] == day]['log10VL'].values
            if value_for_day.size > 0:
                data[i, j] = value_for_day[0]
    
    return data

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


if __name__ == "__main__":
    # Example usage
    data = np.random.randn(10, 5, 3)  # Sample data for 10 patients, 5 time steps, and 3 features

    scaler = CustomScaler()
    scaled_data = scaler.transform(data)
    original_data = scaler.inverse_transform(scaled_data)
