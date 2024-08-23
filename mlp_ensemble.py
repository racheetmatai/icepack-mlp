import os
import tensorflow as tf
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
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

def process_csv(filename):
    df = pd.read_csv(filename, index_col=0)
    df = df.sample(frac=1).reset_index(drop=True)
    df['vel_mag'] = np.sqrt(df['x_velocity']**2 + df['y_velocity']**2)
    df['driving_stress'] = df['h'] * 9.8 * df['mag_s']
    return df

def get_model(inputs, outputs):
    # Assuming you want to predict a single continuous output
    input_dim = inputs.shape[1]
    number_of_layers = 10
    neurons =200
    a_fcn = 'silu'
    # Create a Sequential model
    model = Sequential()

    # Add the input layer and the first hidden layer
    model.add(Dense(units=neurons, activation=a_fcn, input_dim=input_dim))

    for i in range(number_of_layers-1):
        model.add(Dense(units=neurons, activation=a_fcn))
        #model.add(Dropout(0.2))

    # Add the output layer for regression
    model.add(Dense(units=outputs.shape[1])) #, activation='None'))

    # Compile the model with Mean Squared Error loss for regression
    #opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer='adam', loss='mean_squared_error') #, weighted_metrics=['mean_squared_error']) # check this weighted_metrics

    # Print a summary of the model architecture
    return model

def train_ensemble_mlp_model(epochs = 1, variable = 'C', number_of_models = 10, columns = ['s', 'b', 'h', 'mag_h', 'mag_s', 'mag_b', 'driving_stress'], bad_r2_score = 0.5, start_number = 0, variable_type = 'static'):
    df_pig = process_csv('regularized_const_01C_simultaneous_pig_r1_geo.csv')
    df_thwaites = process_csv('regularized_const_01C_simultaneous_thwaites_r1_geo.csv')
    df_dotson = process_csv('regularized_const_01C_simultaneous_dotson_r1_geo.csv')

    df = pd.concat([df_dotson,df_dotson,df_thwaites], ignore_index=True)

    predict_variable = [variable]
    input_columns =  columns 
    inputs = df[input_columns].to_numpy()
    outputs = df[predict_variable].to_numpy()#.reshape(-1,1)

    if np.isnan(inputs).any():
        raise ValueError("There are NaNs in the inputs")
    if np.isinf(inputs).any():
        raise ValueError("There are Infs in the inputs")
    if np.isnan(outputs).any():
        raise ValueError("There are NaNs in the outputs")
    if np.isinf(outputs).any():
        raise ValueError("There are Infs in the outputs")
    
    input_scaler = MinMaxScaler()
    output_scaler = MinMaxScaler() 

    inputs_scaled = input_scaler.fit_transform(inputs)
    outputs_scaled = output_scaler.fit_transform(outputs)

    history_list = []
    model_list = []
    bad_models = 0
    good_models = 0
    for i in range(number_of_models):
        X_train, X_test, y_train, y_test = train_test_split(inputs_scaled, outputs_scaled, test_size=0.1, random_state=42)
        model = get_model(inputs, outputs)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                patience=40, min_lr=0.00001) # 40
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=100, #100
            restore_best_weights=True)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=128, validation_split=0.1, callbacks=[reduce_lr, early_stopping], shuffle=True)
        y_test_predictions = model.predict(X_test)
        r2 = r2_score(y_test_predictions[:,0], y_test[:,0])
        mse = mean_squared_error(y_test_predictions[:,0], y_test[:,0])
        if r2 < bad_r2_score:
            print('Model', i, 'has a bad R2 test score of', r2, ' and will not be saved')
            bad_models += 1
        else:
            history_list.append(history) #, sample_weight = condition_array)
            model_list.append(model)
            save_mlp_model(input_columns, model, input_scaler, output_scaler, history, start_number + good_models, r2, mse, variable_type, variable)
            good_models = good_models + 1
        
    print('Number of bad models:', bad_models)
    return model_list, input_scaler, output_scaler, history_list

def save_mlp_model(input_columns, model, input_scaler, output_scaler, history, save_number, r2, mse, variable_type, variable = 'C'):
    name = 'model_' + str(len(input_columns)) + '_' + 'dotson2_thwaites1_r1_geo' + '_' + variable_type + '_' + variable

    # Bundle all components into a dictionary
    model_bundle = {
        'model_architecture': model.to_json(),
        'model_weights': model.get_weights(),
        'input_scaler': input_scaler,
        'output_scaler': output_scaler,
        'input_columns': input_columns,
        'output_columns': variable,
        'r2_test': r2,
        'mse_test': mse,
        'history_list': history
    }

    # Save the bundle to a single file
    #name = 'model_' + str(len(input_columns)) + '_' + 'split2+'+ '_' +predict_variable[0]
    with open('mlp_ensemble/'+name + '_' + str(save_number) + '.pkl', "wb") as f:
        pickle.dump(model_bundle, f)
    model.save('mlp_ensemble/'+name+ '_' + str(save_number) + '.h5')

def load_ensemble_mlp_model(input_columns, number_of_models = 10, variable = 'C', starting_number = 0):
    name = 'model_' + str(len(input_columns)) + '_' + 'dotson_thwaites_r1_geo'+ '_' +variable
    model_list = []
    for i in range(number_of_models):
        with open('mlp_ensemble/'+name + '_' + str(i+starting_number) + '.pkl', "rb") as f:
            model_bundle = pickle.load(f)
        model = keras.models.load_model('mlp_ensemble/'+name+ '_' + str(i+starting_number) + '.h5')
        model_list.append(model)
    return model_list, model_bundle['input_scaler'], model_bundle['output_scaler'], model_bundle['input_columns'], model_bundle['output_columns']
        






    