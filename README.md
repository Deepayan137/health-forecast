# RRI-PHDS-ODE-Neural-Net-Study

Comparative study of neural network models versus mechanistic ODE models for forecasting with limited and sparse data.

## Overview

This research aims to compare the performance of three distinct neural network architectures: 

- Multi-layer Perceptron (MLP)
- Echo State Network (ESN) with a reservoir
- Long Short-Term Memory (LSTM)

against a mechanistic ODE model for a simple data forecasting task. The dataset comprises simulated patient data, which has been enriched with random missing values to replicate real-world scenarios. Given the sparse nature of this data, we've applied various preprocessing techniques such as imputation (using SAITS) and data smoothing (using moving averages).

## Requirements

- pytorch 1.12
- pypots
- reservoirpy
- scikit-learn

To install all dependencies, use:

```bash
pip install -r requirements.txt
```

## Setup and Usage

1. **Clone the Repository**
   
   ```bash
   git clone https://github.com/Deepayan137/forecast-health-data.git
   cd forecast-health-data
   ```

2. **Data Preparation**
   
   Download the data files from this [Google Drive link](https://drive.google.com/file/d/1dU0Jud300RWbRgFxK-u1zIOB84QW12b3/view?usp=drive_link). After downloading, untar the compressed file and place it in the `data` folder.

3. **Model Training**

   - **LSTM Model**:
     
     Standard training:
     ```bash
     python -m src.train_lstm --out_dim 1 --epochs 600 --exp_name --trial 1
     ```

     For ODE-based inputs:
     ```bash
     python -m src.train_lstm --out_dim 1 --epochs 600 --exp_name --trial 1 --use_ode
     ```

   - **Reservoir**:

     1. Hyperparameter Optimization:
        
        ```bash
        python -m src.reservoir.hp_search_optuna --nb_trial 40 --study_name hp_search --trial 1
        ```
        The `--nb_trial` parameter specifies the number of trials for the hyperparameter optimization. Visualization of the parameters can be done using the notebook located at `src/reservoir/visualize.ipynb`. For more details on hyperparameter optimization, see this [documentation](https://reservoirpy.readthedocs.io/en/latest/user_guide/hyper.html).

     2. Model Training:

        ```bash
        python -m src.reservoir.main --trial 1
        ```
        To train using the outputs of the mechanistic model, append the `--use_ode` flag, similar to the LSTM training command.

   - **MLP Model**:

     Standard training:
     ```bash
     python -m src.train_mlp --trial 1
     ```

     For ODE-based inputs:
     ```bash
     python -m src.train_mlp --trial 1 --use_ode
     ```

4. **Results**:

   All models will display the forecast results for specified days in the terminal. Additionally, metrics such as bias and mean squared error (MSE) will be saved in CSV format under `results/metrics/`.

## Performance Insights

![Performance Results](static/mse_results.png)

The table above presents the RMSE values of our models. The Mechanistic ODE model, boasting the lowest RMSE, highlights its capability to deliver precise forecasts, even when working with sparse data. Both the Reservoir and LSTM models yield competitive RMSE values, with the LSTM model achieving a commendable RMSE of 0.165 on the 180th day using raw input.
