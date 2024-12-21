import os
import tensorflow as tf
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import pickle
from keras.models import model_from_json
import keras
import seaborn as sns
from sklearn.model_selection import train_test_split
import xarray as xr
from datetime import datetime

def get_phi(h, s):
    g = 9769603575225600.0
    ρ_I = 9.207917118369125e-19
    ρ_W = 1.0282341471330407e-18

    p_W = ρ_W * g * np.maximum(0, h - s)
    p_I = ρ_I * g * h

    # Avoid division by zero with a safe divide
    phi = np.where(p_I > 0, np.maximum((1 - p_W / p_I), 0), 0)
    return phi

def process_csv(filename):
    df = pd.read_csv(filename, index_col=0)
    df = df.sample(frac=1).reset_index(drop=True)
    df['vel_mag'] = np.sqrt(df['x_velocity']**2 + df['y_velocity']**2)
    df['driving_stress'] = df['h'] * 9.8 * df['mag_s']
    df['phi'] = get_phi(df['h'].to_numpy(), df['s'].to_numpy())
    return df

def get_model(inputs, outputs):
    # Assuming you want to predict a single continuous output
    input_dim = inputs.shape[1]
    number_of_layers = 10
    neurons = 200
    a_fcn = 'silu'  # Activation function
    
    # Create a Sequential model
    model = Sequential()

    # Add the input layer and the first hidden layer
    model.add(Dense(units=neurons, input_dim=input_dim, activation=None))
    model.add(BatchNormalization())  # Normalize inputs to the activation function
    model.add(Activation(a_fcn))    # Apply activation function separately

    # Add the remaining hidden layers with BatchNormalization
    for i in range(number_of_layers - 1):
        model.add(Dense(units=neurons, activation=None))
        model.add(BatchNormalization())
        model.add(Activation(a_fcn))

    # Add the output layer for regression
    model.add(Dense(units=outputs.shape[1]))  # Linear activation for regression

    # Compile the model with Mean Squared Error loss for regression
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Return the model
    return model

def train_ensemble_mlp_model(select_dataset=1, epochs=1, variable='C', number_of_models=10, 
                             columns=['s', 'b', 'h', 'mag_h', 'mag_s', 'mag_b', 'driving_stress'], 
                             bad_r2_score=0.5, bad_mse_score=1e-2, start_number=0, 
                             variable_type='static', folder_name='mlp_ensemble'):
    df_pig = process_csv('regularized_const_01C_C_only_englacial_temp_pig_r005_geo_12.csv')
    df_thwaites = process_csv('regularized_const_01C_C_only_englacial_temp_thwaites_r005_geo_12.csv')
    df_dotson = process_csv('regularized_const_01C_C_only_englacial_temp_dotson_r005_geo_12.csv')

    # Define datasets and weights based on selection
    datasets = []
    weights = []
    if select_dataset == 0:
        datasets = [df_dotson, df_thwaites]
        weights = [2, 1]
        print("dataset selected: [df_dotson (weight=2), df_thwaites (weight=1)]")
        model_name = 'dotson2_thwaites1_r01_geo'
    elif select_dataset == 1:
        datasets = [df_pig, df_thwaites]
        weights = [2, 1]
        print("dataset selected: [df_pig (weight=2), df_thwaites (weight=1)]")
        model_name = 'pig2_thwaites1_r01_geo'
    elif select_dataset == 2:
        datasets = [df_dotson, df_pig]
        weights = [1, 1]
        print("dataset selected: [df_dotson (weight=1), df_pig (weight=1)]")
        model_name = 'dotson_pig_r01_geo'
    elif select_dataset == 3:
        datasets = [df_dotson]
        weights = [1]
        print("dataset selected: [df_dotson (weight=1)]")
        model_name = 'dotson_r01_geo'
    elif select_dataset == 4:
        datasets = [df_pig]
        weights = [1]
        print("dataset selected: [df_pig (weight=1)]")
        model_name = 'pig_r01_geo'
    elif select_dataset == 5:
        datasets = [df_thwaites]
        weights = [1]
        print("dataset selected: [df_thwaites (weight=1)]")
        model_name = 'thwaites_r01_geo'

    # Combine datasets with weights
    df = pd.concat(datasets, ignore_index=True)
    value = 0.1  # Replace this with your desired threshold value
    df = df[df['phi'] > value].reset_index(drop=True)  # Filter rows where 'phi' is greater than the threshold
    sample_weights = np.concatenate([
        np.ones(len(d[d['phi'] > value])) * w for d, w in zip(datasets, weights)
    ])

    if len(df) != len(sample_weights):
        raise ValueError("Mismatch between filtered data and sample weights after filtering by 'phi'.")

    history_list = []
    model_list = []
    bad_models = 0
    good_models = 0
    r2_score_list = []
    r2_adjusted_list = []
    mse_list = []

    for i in range(number_of_models):
        # Shuffle data and weights together
        shuffled_indices = np.random.permutation(len(df))
        df_shuffled = df.iloc[shuffled_indices].reset_index(drop=True)
        sample_weights_shuffled = sample_weights[shuffled_indices]

        # Prepare inputs and outputs
        predict_variable = [variable]
        input_columns = columns
        inputs = df_shuffled[input_columns].to_numpy()
        outputs = df_shuffled[predict_variable].to_numpy()

        if np.isnan(inputs).any() or np.isnan(outputs).any():
            raise ValueError("There are NaNs in the inputs or outputs.")
        if np.isinf(inputs).any() or np.isinf(outputs).any():
            raise ValueError("There are Infs in the inputs or outputs.")

        # Scale inputs and outputs
        input_scaler = MinMaxScaler()
        output_scaler = MinMaxScaler()
        inputs_scaled = input_scaler.fit_transform(inputs)
        outputs_scaled = output_scaler.fit_transform(outputs)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
            inputs_scaled, outputs_scaled, sample_weights_shuffled, test_size=0.1, random_state=42
        )

        # Create and train the model
        model = get_model(inputs, outputs)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=100, min_lr=0.00001)
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=150, restore_best_weights=True)

        history = model.fit(
            X_train, y_train, sample_weight=sw_train, epochs=epochs, batch_size=128,
            validation_split=0.1, callbacks=[reduce_lr, early_stopping], shuffle=True
        )

        # Evaluate the model
        y_test_predictions = model.predict(X_test)
        r2 = r2_score(y_test_predictions[:, 0], y_test[:, 0])
        r2_adjusted = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - len(X_test[0]) - 1)
        mse = mean_squared_error(y_test_predictions[:, 0], y_test[:, 0])

        # Save or discard the model based on performance
        if r2 < bad_r2_score or mse > bad_mse_score:
            print('Model', i, 'has a bad R2/mse test score of R2:', r2, 'MSE', mse, ' and will not be saved')
            bad_models += 1
        else:
            history_list.append(history)
            model_list.append(model)
            save_mlp_model(input_columns, model, input_scaler, output_scaler, history, 
                           start_number + good_models, r2, r2_adjusted, mse, variable_type, 
                           variable, folder_name=folder_name, model_name=model_name)
            good_models += 1

    # Print summary statistics
    print('Number of bad models:', bad_models)
    r2_score_stats = pd.DataFrame(r2_score_list).describe()
    r2_adjusted_stats = pd.DataFrame(r2_adjusted_list).describe()
    mse_stats = pd.DataFrame(mse_list).describe()
    return model_list, input_scaler, output_scaler, history_list, r2_score_list, r2_adjusted_list, mse_list, r2_score_stats, r2_adjusted_stats, mse_stats



def save_mlp_model(input_columns, model, input_scaler, output_scaler, history, save_number, r2, r2_adjusted, mse, variable_type, variable = 'C', folder_name = 'mlp_ensemble', model_name = 'dotson2_thwaites1_r1_geo'):
    name = 'model_' + str(len(input_columns)) + '_' + model_name + '_' + variable_type + '_' + variable

    # Bundle all components into a dictionary
    model_bundle = {
        'model_architecture': model.to_json(),
        'model_weights': model.get_weights(),
        'input_scaler': input_scaler,
        'output_scaler': output_scaler,
        'input_columns': input_columns,
        'output_columns': variable,
        'r2_test': r2,
        'r2_adjusted_test': r2_adjusted,
        'mse_test': mse,
        'history_list': history,
    }

    # Save the bundle to a single file
    #name = 'model_' + str(len(input_columns)) + '_' + 'split2+'+ '_' +predict_variable[0]
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    name_pkl = name + '_' + str(save_number) + '.pkl'
    name_h5 = name + '_' + str(save_number) + '.h5'

    with open(os.path.join(folder_name, name_pkl), "wb") as f:
        pickle.dump(model_bundle, f)
    model.save(os.path.join(folder_name, name_h5))

def loop_train_ensemble_mlp_model(list_columns , select_dataset = 1, epochs = 1, variable = 'C', number_of_models = 10, bad_r2_score = -500, start_number = 0, variable_type = 'mixed',base_folder_name = 'mlp_ensemble'):
    base_folder_name = base_folder_name + '_' + str(select_dataset)
    model_list = []
    input_scaler_list = []
    output_scaler_list = []
    history_list = []
    r2_stats_list = []
    r2_adjusted_stats_list = []
    mse_stats_list = []
    for i, columns in enumerate(list_columns):
        folder_name = os.path.join(base_folder_name, str(hash(tuple(columns))))
        model_ensemble, input_scaler, output_scaler, history_ensemble, r2_score_list, r2_adjusted_list, mse_list, r2_score_stats, r2_adjusted_stats, mse_stats = train_ensemble_mlp_model(select_dataset = select_dataset, epochs = epochs, variable = variable, number_of_models = number_of_models, columns = columns, bad_r2_score = bad_r2_score, start_number = start_number, variable_type = variable_type, folder_name = folder_name)
        model_list.append(model_ensemble)
        input_scaler_list.append(input_scaler)
        output_scaler_list.append(output_scaler)
        history_list.append(history_ensemble)
        r2_stats_list.append(r2_score_stats)
        r2_adjusted_stats_list.append(r2_adjusted_stats)
        mse_stats_list.append(mse_stats)

    with open(os.path.join(base_folder_name, 'r2_stats_list.pkl'), "wb") as f:
        pickle.dump(r2_stats_list, f)

    with open(os.path.join(base_folder_name, 'r2_adjusted_stats_list.pkl'), "wb") as f:
        pickle.dump(r2_adjusted_stats_list, f)
    
    with open(os.path.join(base_folder_name, 'mse_stats_list.pkl'), "wb") as f:
        pickle.dump(mse_stats_list, f)

    df_summary = pd.DataFrame(columns=['input_columns', 'r2_mean', 'r2_std', 'r2_median', 'r2_adjusted_mean', 'r2_adjusted_std', 'r2_adjusted_median', 'mse_mean', 'mse_std', 'mse_median'])
    for i, columns in enumerate(list_columns):
        df_summary.loc[i] = [columns, r2_stats_list[i].loc['mean'].values[0], r2_stats_list[i].loc['std'].values[0], r2_stats_list[i].loc['50%'].values[0], r2_adjusted_stats_list[i].loc['mean'].values[0], r2_adjusted_stats_list[i].loc['std'].values[0], r2_adjusted_stats_list[i].loc['50%'].values[0], mse_stats_list[i].loc['mean'].values[0], mse_stats_list[i].loc['std'].values[0], mse_stats_list[i].loc['50%'].values[0]]

    summary_name = 'summary_' + str(random.randint(0, 9999)) + '.csv'
    df_summary.to_csv(os.path.join(base_folder_name, summary_name))

    return model_list, input_scaler_list, output_scaler_list, history_list, r2_stats_list, r2_adjusted_stats_list, mse_stats_list, df_summary

def get_model_summary(base_folder='mlp_ensemble'):
    summary_list = []
    
    for folder in os.listdir(base_folder):
        try:
            folder_num = int(folder)
        except ValueError:
            continue  # Skip non-numeric folder names
        
        if os.path.isdir(os.path.join(base_folder, folder)):
            print('Processing folder:', folder)
            path = os.path.join(base_folder, folder)
            files = [f for f in os.listdir(path) if f.endswith('.pkl')]
            
            if not files:
                print(f"No .pkl files found in folder {folder}. Skipping.")
                continue
            
            r2_list, r2_adjusted_list, mse_list = [], [], []
            columns = None
            number_of_models = 0 
            for file in files:
                
                try:
                    with open(os.path.join(path, file), "rb") as f:
                        model_bundle = pickle.load(f)
                        r2_list.append(model_bundle['r2_test'])
                        r2_adjusted_list.append(model_bundle['r2_adjusted_test'])
                        mse_list.append(model_bundle['mse_test'])
                        columns = model_bundle.get('input_columns', columns)
                        number_of_models = number_of_models + 1
                except Exception as e:
                    print(f"Error processing file {file} in folder {folder}: {e}")
                    continue

            if r2_list and r2_adjusted_list and mse_list:
                r2_stats = pd.DataFrame(r2_list).describe()
                r2_adjusted_stats = pd.DataFrame(r2_adjusted_list).describe()
                mse_stats = pd.DataFrame(mse_list).describe()
                
                summary_list.append({
                    'folder_name':folder_num,
                    'input_columns': columns,
                    'r2_mean': r2_stats.loc['mean'].values[0],
                    'r2_std': r2_stats.loc['std'].values[0],
                    'r2_median': r2_stats.loc['50%'].values[0],
                    'r2_adjusted_mean': r2_adjusted_stats.loc['mean'].values[0],
                    'r2_adjusted_std': r2_adjusted_stats.loc['std'].values[0],
                    'r2_adjusted_median': r2_adjusted_stats.loc['50%'].values[0],
                    'mse_mean': mse_stats.loc['mean'].values[0],
                    'mse_std': mse_stats.loc['std'].values[0],
                    'mse_median': mse_stats.loc['50%'].values[0],
                    'number_of_models':number_of_models,
                })
                
    df_summary = pd.DataFrame(summary_list)
    if not df_summary.empty:
        summary_name = f'summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df_summary.to_csv(os.path.join(base_folder, summary_name), index=False)
        print(f'Summary saved as {summary_name}')
    else:
        print('No data to summarize.')
    
    return df_summary
    
# Not updated do not use
def load_ensemble_mlp_model(input_columns, number_of_models = 10, variable = 'C', starting_number = 0):
    name = 'model_' + str(len(input_columns)) + '_' + 'dotson_thwaites_r1_geo'+ '_' +variable
    model_list = []
    for i in range(number_of_models):
        with open('mlp_ensemble/'+name + '_' + str(i+starting_number) + '.pkl', "rb") as f:
            model_bundle = pickle.load(f)
        model = keras.models.load_model('mlp_ensemble/'+name+ '_' + str(i+starting_number) + '.h5')
        model_list.append(model)
    return model_list, model_bundle['input_scaler'], model_bundle['output_scaler'], model_bundle['input_columns'], model_bundle['output_columns']
        






    