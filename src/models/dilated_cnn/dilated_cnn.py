"""
Created on Oct 31 2018

@author: Ray

"""

import numpy as np
import pandas as pd
import tensorflow as tf
import gc
import time
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from data_frame import DataFrame
from tf_base_model import TFBaseModel # for building our customized
# model 
from tf_utils import TemporalConvNet
from tf_utils import time_distributed_dense_layer
from tf_utils import sequence_softmax_loss
from tf_utils import sequence_evaluation_metric
# laoding data
from data_utils import get_glove_vectors
from data_utils import load_vocab_and_return_word_to_id_dict


config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth=True

class DataReader(object):
    '''for reading data'''
    
    def __init__(self, data_dir):
        data_cols = [
            'item_id',
            'word_id',
            'history_length',
            'char_id',
            'word_length',
            'label'
        ]
        #-----------------
        # loading data
        #-----------------
        data_train = [np.load(os.path.join(data_dir, 'train/{}.npy'.format(i)), mmap_mode='r') for i in data_cols]
        data_val = [np.load(os.path.join(data_dir, 'val/{}.npy'.format(i)), mmap_mode='r') for i in data_cols]
        data_test = [np.load(os.path.join(data_dir, 'test/{}.npy'.format(i)), mmap_mode='r') for i in data_cols]

        #------------------
        # For Testing-phase
        #------------------
        self.test_df = DataFrame(columns=data_cols, data=data_test)
        print ('loaded data')
        #------------------
        # For Training-phase
        #------------------
        self.train_df = DataFrame(columns=data_cols, data=data_train)
        self.val_df = DataFrame(columns=data_cols, data=data_val)

        print ('shape of whole data : {}'.format(self.test_df.shapes()))
        print ('number of training example: {}'.format(len(self.train_df)))
        print ('number of validating example: {}'.format(len(self.val_df)))
        print ('number of testing example: {}'.format(len(self.test_df)))
        
    def train_batch_generator(self, batch_size, num_epochs=100000, shuffle = True, is_test = False):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.train_df,
            shuffle=shuffle,
            num_epochs=num_epochs,
            is_test=is_test
        )

    def val_batch_generator(self, batch_size, num_epochs=100000, shuffle = True, is_test = False):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.val_df,
            shuffle=shuffle,
            num_epochs=num_epochs,
            is_test=is_test
        )

    def test_batch_generator(self, batch_size):
        '''All row in our dataframe need to predicted as input of second-level model'''
        return self.batch_generator(
            batch_size=batch_size,
            df=self.test_df,
            shuffle=False,
            num_epochs=1,
            is_test=True
        )
    
    def batch_generator(self, batch_size, df, shuffle=True, num_epochs=10000, is_test=False):
        '''
        df: customized DataFrame object,
        '''
        # call our customized DataFrame object method batch_generator
        batch_gen = df.batch_generator(batch_size, shuffle = shuffle, num_epochs=num_epochs, allow_smaller_final_batch=is_test)
        # batch_gen is a generator
        for batch in batch_gen:
            # what batch_gen yield is also a customized Dataframe object.
            if not is_test:
                pass
            yield batch

class dilated_cnn(TFBaseModel):
    
    def __init__(self,max_seq_len,filter_widths,hidden_size_cnn,num_hidden_layers,
                 ntags,trainable_embedding,dim_word,nwords,metric,use_chars,
                 dim_char,max_word_length,nchars,hidden_size_char, dilation_rate,**kwargs):
        self.max_seq_len = max_seq_len
        self.filter_widths = filter_widths
        self.hidden_size_cnn = hidden_size_cnn
        self.num_hidden_layers = num_hidden_layers
        self.ntags = ntags # n_class
        self.dim_word = dim_word
        self.nwords = nwords
        self.trainable_embedding = trainable_embedding
        self.metric = metric
        self.USE_CHARS = use_chars
        self.dilation_rate = dilation_rate # list of dilation factor
        if self.USE_CHARS:
            try:
                self.dim_char = dim_char
                self.max_word_length = max_word_length
                self.nchars = nchars
                self.hidden_size_char = hidden_size_char
            except:
                assert False, 'Please assing dim_char, max_word_length, and nchars as arguments'
        # self.super call  _init_ function of parent class 
        super(dilated_cnn, self).__init__(**kwargs)

    def get_input_sequences(self):
        ####################################
        # Step1: get input_sequences 
        ####################################
        #------------
        # 1-D  
        #------------
        self.item_id = tf.placeholder(tf.int32, [None])
        self.history_length = tf.placeholder(tf.int32, [None]) # It's for arg of lstm model: sequence_length, == len(is_ordered_history)
        #------------   
        # 2-D  
        #------------
        self.word_id = tf.placeholder(tf.int32, [None, self.max_seq_len]) 
        self.label = tf.placeholder(tf.int32, [None, self.max_seq_len]) # [batch_size, num_class]
        if self.USE_CHARS:
            self.word_length = tf.placeholder(tf.int32, shape=[None, self.max_seq_len])
        #------------   
        # 3-D  
        #------------
        if self.USE_CHARS:
            self.char_id = tf.placeholder(tf.int32, shape=[None, self.max_seq_len, self.max_word_length]) # [batch_size, max_seq_length, max_word_length]


        #------------
        # boolean parameter
        #------------
        self.keep_prob = tf.placeholder(tf.float32)
        #------------
        # word_embedding: get embeddings matrix
        #------------
        if embeddings is None:
            logging.info('WARNING: randomly initializing word vectors')
            word_embeddings = tf.get_variable(
            shape = [nwords, dim_word],
            name = 'word_embeddings',
            dtype = tf.float32,
            )
        else:
            word_embeddings = tf.get_variable(
            initializer = embeddings, # it will hold the embedding
            #shape = [word2vec.shape[0], word2vec.shape[1]], # [num_vocabulary, embeddings_dim]
            trainable = trainable_embedding,
            name = 'word_embeddings',
            dtype = tf.float32
            )
        word_representation = tf.nn.embedding_lookup(params = word_embeddings, ids = self.word_id)
        #------------
        # char_embedding: get char embeddings matrix
        #------------
        if self.USE_CHARS:
            # get char embeddings matrix
            char_embeddings = tf.get_variable(
                    shape=[self.nchars, self.dim_char],
                    name="char_embeddings",
                    trainable = True,
                    dtype=tf.float32,
            )
            # get char_representation, 4-D, [batch_size, max_seq_length, max_word_length, dim_char]
            char_representation = tf.nn.embedding_lookup(params = char_embeddings, ids = self.char_id) 
            # convert 4-D into 3-D: put the timestep on axis=1 and should be charater-level axis
            s = tf.shape(char_representation) # 1-D tensor, (batch_size, max_seq_length, max_word_length, dim_char)
            char_representation = tf.reshape(char_representation, shape=[ s[0]*s[1], s[-2], self.dim_char]) # [batch_size * max_seq_length, max_word_length, dim_char]
            # for computing bi lstm on chars
            word_lengths = tf.reshape(self.word_length, shape=[s[0]*s[1]]) # 1-D tensor
            # bi lstm on chars
            cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size_char,
                    state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size_char,
                    state_is_tuple=True)
            _output = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, 
                    cell_bw, 
                    inputs = char_representation,
                    sequence_length = word_lengths, 
                    dtype=tf.float32) 
            """
            Return A tuple (outputs, output_states) 
              -outputs: A tuple (output_fw, output_bw) containing the forward and the backward rnn output
                   1.output_fw with shape of [batch_szie*max_word_lenght, max_word_length, hidden_size_char].For example, (72, 54, 100)
                   2.output_bw with shape of [batch_szie*max_word_lenght, max_word_length, hidden_size_char]
              -output_states: A tuple (output_state_fw, output_state_bw) containing the forward and the backward final states of bidirectional rnn.
                   1.final_state_output_fw with shape of [batch_szie*max_word_lenght, hidden_size_char]. For instance, (72, 100)
                   2.final_state_output_bw with shape of [batch_szie*max_word_lenght, hidden_size_char]

            """
            # get word level representation from characters embeddings
            _, ((_, output_fw_final_state), (_, output_bw_final_state)) = _output
            output = tf.concat([output_fw_final_state, output_bw_final_state], axis=-1)# concat on char_embedding level, [batch_szie*max_word_lenght, 2*hidden_size_char]
            # reshape to word level representation
            word_representation_extracted_from_char = tf.reshape(output, shape=[s[0], s[1], 2* hidden_size_char]) # [batch_size, max_seq_length, 2*hidden_size_char]
        
        if self.USE_CHARS:
            x = tf.concat([
            word_representation,
            word_representation_extracted_from_char
                ], axis = 2) # (?, 36, 500 == 300+200)  
        else:
            x = tf.concat([
            word_representation,
                ], axis=2) # (?, 36, 300)
        #print ('Original features : {}'.format(x.shape))
        return x
    
    def calculate_outputs(self, x):
        ####################################
        # Step2: calculate_outputs 
        ####################################
        #-------------------------
        # Model-Simple CNN
        #-------------------------
        for i, dilation in zip(range(self.num_hidden_layers), self.dilation_rate):
            if i == 0:
                self.conv = temporal_convolution_layer(x, 
                                           output_units = self.hidden_size_cnn,
                                           convolution_width = self.filter_widths,
                                           dilated = True,
                                           dilation_rate = [dilation],
                                           causal = True,
                                           bias=True,
                                           activation=None, 
                                           dropout=None,
                                           scope='cnn-{}'.format(i),
                                           reuse = False,
                                          )
            else:
                self.conv = temporal_convolution_layer(self.conv, 
                                           output_units = self.hidden_size_cnn,
                                           convolution_width = self.filter_widths,
                                           dilated = True,
                                           dilation_rate = [dilation],
                                           causal = True,
                                           bias=True,
                                           activation=None, 
                                           dropout=None,
                                           scope='cnn-{}'.format(i),
                                           reuse = False,
                                          )

            #print ('CNN-{} layer : {}'.format(i, conv.shape))
        # output layer (linear)
        y_hat = time_distributed_dense_layer(self.conv, self.ntags, activation=None, scope='output-layer') # (?, 122, 3)
        #--------------
        # for second-level model: 
        #--------------
        # method1: using final_temporal_idx, it's for only get the final step. -->That's not here task I want
        # method2: Only saving history_length rather tahn saving max_seq_lenght --> By far, don't know how to do that.
        self.prediction_tensors = {
            'item_id':self.item_id, # (?, )
            'word_id':self.word_id, # (?, max_seq_length)
            'history_length':self.history_length,
            'final_states':self.conv, # (?, max_seq_length, num_hidden_units)
            'final_predictions':y_hat, # (?, max_seq_length, ntags)
        }
        return y_hat
    
    def calculate_loss(self):
        x = self.get_input_sequences()
        self.preds = self.calculate_outputs(x)
        loss = sequence_softmax_loss(y = self.label, 
                                     y_hat = self.preds, 
                                     sequence_lengths = self.history_length, 
                                     max_sequence_length = self.max_seq_len)
        return loss
    
    def calculate_evaluation_metric(self):
        labels_pred = tf.cast(tf.argmax(self.preds, axis= 2),tf.int32) # (?, max_seq_length)
        score = sequence_evaluation_metric(y = self.label,
                                           y_hat = labels_pred, 
                                           sequence_lengths = self.history_length,
                                           max_sequence_length = self.max_seq_len
                                          )[self.metric]
        return score

if __name__ == '__main__':
    #--------------------------
    # setting
    #--------------------------
    dim_word = 300
    trainable_embedding = False
    USE_PRETRAINED = True
    filename_words_vec = "../data/wordvec/word2vec.npz".format(dim_word)
    filename_words_voc = "../data/wordvec/words_vocab.txt"
    filename_chars_voc = "../data/wordvec/chars_vocab.txt"
    base_dir = ''

    nwords = len(load_vocab_and_return_word_to_id_dict(filename_words_voc))
    embeddings = (get_glove_vectors(filename_words_vec) if USE_PRETRAINED else None)
    embeddings = embeddings.astype(np.float32)
    enable_parameter_averaging = False
    num_hidden_layers = 4
    dilation_rate = [2**i for i in range(num_hidden_layers)]
    USE_CHARS = True
    if USE_CHARS:
        hidden_size_char = 100
        nchars = len(load_vocab_and_return_word_to_id_dict(filename_chars_voc))

    # read data
    dr = DataReader(data_dir ='../data/')
    # simple_cnn model
    nn = dilated_cnn(
        reader=dr,
        log_dir=os.path.join(base_dir, 'logs'),
        checkpoint_dir=os.path.join(base_dir, 'checkpoints_w_causal'),
        prediction_dir=os.path.join(base_dir, 'predictions_w_causal'),
        optimizer='adam',
        max_seq_len = 36,
        learning_rate =0.001,
        hidden_size_cnn = 300,
        filter_widths= 3,
        num_hidden_layers = num_hidden_layers,
        ntags = 3,
        batch_size = 128,
        dim_word = 300,
        nwords = nwords,
        trainable_embedding = False,
        metric = 'f1',
        use_chars = USE_CHARS,
        nchars = nchars,
        hidden_size_char = hidden_size_char,
        dilation_rate = dilation_rate,
        max_word_length = 54,
        dim_char = 100,
        num_training_steps = 15000,
        early_stopping_steps = 3000,
        loss_averaging_window = 100,
        use_evaluation_metric_as_early_stopping = True,
        warm_start_init_step = 0, # for some case, we don't want to train the model from the beginning
        regularization_constant = 0.0,
        keep_prob = 1.0,
        enable_parameter_averaging = False,
        num_restarts = 0,
        min_steps_to_checkpoint = 500, # The value of this need to larger than best_validation_tstep. Otherwise, we won't save our best model
        log_interval = 1,
        num_validation_batches = 25,
    )
    # training
    nn.fit()
    # Restore model weights from previously saved model(best) : Note: xxx should be the same as xxx
    nn.restore()
    # Get the evaluation score on the whole validation-set
    nn.evaluate() 
    # prediction
    nn.predict()


