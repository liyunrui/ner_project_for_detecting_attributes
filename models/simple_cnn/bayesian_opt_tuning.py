# ! /usr/bin/env python3
"""
Created on Oct 11 2018

@author: Ray

Reference: 
    - tutorial : https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/19_Hyper-Parameters.ipynb
    - library: https://scikit-optimize.github.io/

The priority of hyper-parameters:
    #-----------
    Most important
    #-----------
        - learning_rate
    #-----------
    Second importance
    #-----------
        - batch size
        - optimizer
        - #hidden units
    #-----------
    Third importance
    #-----------
        - #layers
        -regularization_constant
        -dropout
    #-----------
    # final 
    #-----------
    -batch normalization(True, False)
    -exponentially weitedd average(True, False)
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append('../models/')
# sys.path.append('../models/rnn_session/')

from data_frame import DataFrame # for reading data
from tf_base_model import TFBaseModel # for building our customized
# helper funtion
from tf_utils import time_distributed_dense_layer
from tf_utils import lstm_layer
from tf_utils import wavenet
from tf_utils import sequence_rmse
from tf_utils import shape_of_tensor
import pprint as pp
# model needed to be tuned
from simple_cnn import DataReader
from simple_cnn import simple_cnn
# para tuning library
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
# laoding data
from data_utils import get_glove_vectors
from data_utils import load_vocab_and_return_word_to_id_dict

# setting
STEP = 'FIRST PHASE'
data_dir = 'data/' 
base_dir = 'tuning/' # for saving model and prediction
log_dir = 'logs_tuning/'

filename_words_voc = "../data/wordvec/words_vocab.txt"
nwords = len(load_vocab_and_return_word_to_id_dict(filename_words_voc))
embeddings = (get_glove_vectors(filename_words_vec) if USE_PRETRAINED else None)
embeddings = embeddings.astype(np.float32)
USE_CHARS = True
if USE_CHARS:
    filename_chars_voc = "../data/wordvec/chars_vocab.txt"
    hidden_size_char = 100
    nchars = len(load_vocab_and_return_word_to_id_dict(filename_chars_voc))

# reading data
s = time.time()
dr = DataReader(data_dir = data_dir,TRACE_CODE = TRACE_CODE)
e = time.time()
print ('reading data takes {} mins'.format( (e-s)/60.0) )

if STEP == 'FIRST PHASE':
    ########################
    # Step1: We first need to define the valid search-ranges or search-dimensions for each of these parameters.
    ########################
    dim_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform', name='learning_rate')

    search_space = [
    dim_learning_rate,
    ]
    ########################
    # Step2: choosing starting point
    ########################
    default_parameters = [1e-3]

elif STEP == 'SECOND PHASE':
    ########################
    # Step1: We first need to define the valid search-ranges or search-dimensions for each of these parameters.
    ########################
    dim_num_dense_nodes_in_cnn = Integer(low=100, high=500, name='num_dense_nodes_in_cnn')
    dim_optimizer = Categorical(categories=['adam', 'adagrad','rms'], name='optimizer')
    dim_batch_size = Integer(low= 32, high=256, name='batch_size') 

    search_space = [
    dim_num_dense_nodes_in_cnn,
    dim_optimizer,
    dim_batch_size,
    ]
    ########################
    # Step2: choosing starting point
    ########################
    default_parameters = [300,'adam',128]

elif STEP == 'THIRD PHASE':
    ########################
    # Step1: We first need to define the valid search-ranges or search-dimensions for each of these parameters.
    ########################
    dim_num_layers = Integer(low=1, high=5, name='num_layers') # Why choose the range number is refer to my refrence 2
    dim_keep_prob = Real(low=0.0, high=1.0, name='keep_prob')
    dim_regularization_constant = Real(low=0.0, high=1e-4, name='regularization_constant', prior='log-uniform')

    search_space = [
    dim_num_layers,
    dim_keep_prob,
    dim_regularization_constant,
    ]
    ########################
    # Step2: choosing starting point
    ########################
    default_parameters = [2, 0.5, 0.0]
else:
    assert False, 'Now, We only have third phase.'

# for counting
c = 0

@use_named_args(search_space)
def fitness(learning_rate, batch_size = None,
            num_dense_nodes_in_cnn = None, optimizer = None,
            num_layers = None, keep_prob = None, regularization_constant = None,
           ):
 
    global c

    # saving hyper-parameters.
    if os.path.exists('tuning_logging.txt'):
        f = open('tuning_logging.txt', 'a')
        f.write(',')
    else:
        f = open('tuning_logging.txt', 'w')

    f.write('learning_rate {}'.format(learning_rate))
    f.write(',')

    f.write('batch_size {}'.format(batch_size))
    f.write(',')

    f.write('num_dense_nodes_in_cnn {}'.format(num_dense_nodes_in_cnn))
    f.write(',')

    f.write('optimizer {}'.format(optimizer))
    f.write(',')

    f.write('num_layers {}'.format(num_layers))
    f.write(',')

    f.write('keep_prob {}'.format(keep_prob))
    f.write(',')

    f.write('regularization_constant {}'.format(regularization_constant))
    f.write(',')

    #-------------------------
    # Customized model: Create the neural network with these hyper-parameters.
    #-------------------------
    if STEP == 'FIRST PHASE':
        # model
        nn = simple_cnn(
            reader=dr,
            log_dir=os.path.join(base_dir, 'logs'),
            checkpoint_dir=os.path.join(base_dir, 'checkpoints'),
            prediction_dir=os.path.join(base_dir, 'predictions'),
            optimizer='adam',
            max_seq_len = 36,
            learning_rate = learning_rate,
            hidden_size_cnn = 300,
            filter_widths= 3,
            num_hidden_layers = 2,
            ntags = 3,
            batch_size = 128,
            dim_word = 300,
            nwords = nwords,
            trainable_embedding = False,
            metric = 'f1',
            use_chars = USE_CHARS,
            nchars = nchars,
            hidden_size_char = 100,
            max_word_length = 54,
            dim_char = 100,
            num_training_steps = 10000,
            early_stopping_steps = 50,
            loss_averaging_window = 100,
            use_evaluation_metric_as_early_stopping = True,
            warm_start_init_step = 0, # for some case, we don't want to train the model from the beginning
            regularization_constant = 0.0,
            keep_prob = 1.0,
            enable_parameter_averaging = False,
            num_restarts = 0,
            min_steps_to_checkpoint = 1000, # The value of this need to larger than best_validation_tstep. Otherwise, we won't save our best model
            log_interval = 20,
            num_validation_batches = 5,
        )

    #-------------------------
    # model training 
    #-------------------------
    nn.fit()
    #-------------------------
    # restoring our session into the best model
    #-------------------------
    nn.restore()
    #-------------------------
    # Get the loss on the whole validation-set
    #-------------------------
    best_score_on_validating_set = nn.evaluate()
    f.write('best_score_on_validating_set {}'.format(best_score_on_validating_set))
    f.flush()
    # Delete the tf model with these hyper-parameters from memory.
    del nn
    c += 1
    return 1 - best_score_on_validating_set

search_result = gp_minimize(func = fitness,
                            dimensions = search_space,
                            acq_func = 'EI', # Expected Improvement.
                            n_calls = 40, # The total number of evaluations
                            x0 = default_parameters, # default_parameters as the starting point we have found by hand-tuning(recommended)
                            verbose = True 
                           )

result_list = sorted(zip(search_result.func_vals, search_result.x_iters))
print ('evaluation loss: {}'.format(pp.pformat(result_list)))
