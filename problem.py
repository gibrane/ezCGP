# TODO edit problem towards mnist usage

import numpy as np
#from scipy.stats import weibull_min
import scipy.stats as scst
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import tensorflow as tf
import os
import six
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

import operators
import arguments
import mutate_methods as mut
import mate_methods as mate

# constants
generation_limit = 19
score_min = 0.00 # terminate immediately when 100% accuracy is achieved

# Helper to load in Housing Dataset
"""Returns X_train, y_train, X_test, y_test except the features have been scalled and the target values have been binned"""
def split_and_normalize(X_raw, y_raw):
    # 75% train, 15% test, 15% val
    X_train, X_test, y_train, y_test = \
        train_test_split(X_raw, y_raw, test_size=0.30)

    # reshaping because there's only one feature (price)
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_norm = scaler.transform(X_train)
    X_test_norm = scaler.transform(X_test)

    scaler2 = StandardScaler()
    scaler2.fit(y_train)

    y_train_norm = scaler2.transform(y_train)
    y_test_norm = scaler2.transform(y_test)

    return X_train_norm, y_train_norm, X_test_norm, y_test_norm

def load_housing():
    #Split the data into features, X,  and predictions, y
    raw_data = pd.read_csv('housing_dataset/kc_house_data.csv')
    raw_data.dropna(inplace = True) #drops rows with Nans
    y = raw_data['price']
    drop_features = ["id","date", "price", "zipcode", "yr_renovated"]
    X_raw = raw_data.drop(drop_features, axis = 1)
    X = X_raw.values
    y = y.values # prices

    X_train, y_train, X_test, y_test = split_and_normalize(X, y)

    validation_index = int(len(X_test)/2)

    # Subsample the testing data (split) to validation and test
    X_val = X_test[:validation_index]
    y_val = y_test[:validation_index]
    X_test = X_test[:validation_index]
    y_test = y_test[:validation_index]

    features = X_raw.columns
    return X_train, y_train, X_val, y_val, X_test, y_test

# total entries is 21613 in housing dataset
x_train, y_train, x_val, y_val, x_test, y_test = load_housing()
print("Training shapes: ", x_train.shape, y_train.shape)
print("Validation shapes: ", x_val.shape, y_val.shape)
print("Test shapes: ", x_test.shape, y_test.shape)
print("Total amount of data preprocessed: ", \
     x_train.shape[0] + x_val.shape[0] + x_test.shape[0])


x_train = np.array([x_train]) #this is so blocks.py does not break
x_test = np.array([x_test])

# Regression Score Function
def scoreFunction(predict, actual):
    try:
        # TODO Average Pecentage Change (priority in tuple is first) - might be domain specific
        # regression flag here and in evaluate() in blocks.py (softmax vs. dense)
        # think about outliers or data error, so try to avoid min/max, etc.
        # be conscious of the dataset
        mae = mean_absolute_error(actual, predict)
        mse = mean_squared_error(actual, predict)
        return mae, mse # to minimize
    except ValueError:
        print('Malformed predictions passed in. Setting worst fitness')

        # not just 1,1 because some mse/mae can be > 1
        return math.inf, math.inf # infinite error for truly terrible individuals

#print('Train: X: {} y: {}'.format(x_train[0].shape, y_train.shape))
#print('Validation: X: {} y: {}'.format(x_val.shape, y_val.shape))
#print('Test: X: {} y: {}'.format(x_test[0].shape, y_test.shape))

# NOTE: a lot of this is hastily developed and I do hope to improve the 'initialization'
#structure of the genome; please note your own ideas and we'll make that a 'project' on github soon

skeleton_block = { #this skeleton defines a SINGLE BLOCK of a genome
    'tensorblock_flag': True,
    'batch_size': 128,
    'nickname': 'tensor_mnist_block',
    'setup_dict_ftn': {
        #declare which primitives are available to the genome,
        #and assign a 'prob' so that you can control how likely a primitive will be used;
        #prob: float btwn 0 and 1 -> assigns that prob to that primitive...the sum can't be more than 1
        #prob: 1 -> equally distribute the remaining probability amoung all those remaining (hard to explain, sorry)
        #operators.input_layer: {'prob': 1},
        #operators.add_tensors: {'prob': 1},
        #operators.sub_tensors: {'prob': 1},
        #operators.mult_tensors: {'prob': 1},
        operators.dense_layer: {'prob': 1},
        #operators.conv_layer: {'prob': 1},
        #operators.max_pool_layer: {'prob': 1},
        #operators.avg_pool_layer: {'prob': 1},
        #operators.concat_func: {'prob': 1},
        # operators.sum_func: {'prob': 1},
        #operators.conv_block: {'prob': 1},
        #operators.res_block: {'prob': 1},
        #operators.sqeeze_excitation_block: {'prob': 1},
        #operators.identity_block: {'prob': 1},
    },
    'setup_dict_arg': {
        #if you have an 'arguments genome', declare which argument-datatypes should fill the argument genome
        #not used for now...arguments genome still needs to be tested
        arguments.argInt: {'prob': 1}}, #arg count set to 0 though
    'setup_dict_mate': {
        #declare which mating methods are available to genomes
        mate.Mate.dont_mate: {'prob': 1, 'args': []}},
    'setup_dict_mut': {
        #declare which mutation methods are available to the genomes
        mut.Mutate.mutate_singleInput: {'prob': 1, 'args': []},
        #mut.Mutate.mutate_singleArg: {'prob': 1, 'args': []},
        # mut.Mutate.mutate_singleFtn: {'prob': 1, 'args': []},
    },
    'operator_dict': operators.operDict, #further defines what datatypes what arguments are required for each primitive
    'block_input_dtypes': [tf.Tensor], #placeholder datatypes so that the genome can be built off datatypes instead of real data
    'block_outputs_dtypes': [tf.Tensor],
    'block_main_count': 23, #10 genes
    'block_arg_count': 2, #not used...no primitives require arguments
    'block_mut_prob': 1, #mutate genome with probability 1...always
    'block_mate_prob': 0 #mate with probability 0...never
}

skeleton_genome = { # this defines the WHOLE GENOME
    'input': [np.ndarray], # we don't pass in the labels since the labels are only used at evaluation and scoring time
    'output': [np.ndarray],
    1: skeleton_block
}
