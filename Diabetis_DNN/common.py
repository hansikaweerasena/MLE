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