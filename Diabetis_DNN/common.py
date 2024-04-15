# Load packages we need
import sys
import os
import time

import numpy as np
import pandas as pd
import sklearn
import seaborn as sns

import tensorflow as tf
import tensorflow.keras as keras

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 14})


def var_exists(var_name):
    return (var_name in globals() or var_name in locals())


def load_data(path='diabetes_012_health_indicators_BRFSS2015.csv'):
    diabetes_data = pd.read_csv(path)
    print(f'Data loaded successfully data shape:', diabetes_data.shape )
    return diabetes_data


def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f'Seed set to {seed}')


def split_and_scale_data(data_df, random_state=42, target='Diabetes_012', test_val_prop=0.1, val_prop=0.5, scaler='MinMax', oneHotEncode=False):
   
    # Split the data into features and labels
    x_all = data.drop('Diabetes_012', axis=1)
    y_all = data['Diabetes_012']

    if onehot:
        num_classes = 3
        y_all = keras.utils.to_categorical(y_all, num_classes)

    # Split the data into training, validation, and test sets
    train_x, temp_x, train_y, temp_y = train_test_split(x_all, y_all, test_size=test_val_prop, random_state=seed)
    val_x, test_x, val_y, test_y = train_test_split(temp_x, temp_y, test_size=val_prop, random_state=seed)

    if scaler == 'MinMax':
        scaler = MinMaxScaler()
    elif scaler == 'Standard':
        scaler = StandardScaler()
    else:
        raise ValueError('Invalid scaler')

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test 