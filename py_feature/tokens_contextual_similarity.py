#! /usr/bin/env python3
"""
Created on Aug 21 2018

In order to capture similarity between tokens and surrounding tokens.
Here out similarity is defined by n-gram aka occurrence of words combination.

Add new features imitating what dilatedd cnn do on Sep 5 2018


@author: Ray

"""

import nltk, re, string, collections
from nltk.util import ngrams # function for making ngrams
import pandas as pd
import numpy as np
from datetime import datetime # for the newest version control
import os
import time
import multiprocessing as mp # for speeding up some process
import logging
from nltk import tag # for pos_tagging
from nltk.corpus import wordnet # for geting pos of wordnet
from nltk.stem import WordNetLemmatizer
import gc
from billiard import Pool
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

def get_the_preceding_word(row, window_size = 1):
    '''
    Get the preceding word given the token. 
    It's a helper function to compute the sequential feature of the word.
    '''
    try:
        the_former_ix = row.item_name.split().index(row.tokens) - window_size
        if the_former_ix < 0:
            return -1 # It means the former word is non-existent. # -1 is bettern than missing value
        else:
            return row.item_name.split()[the_former_ix]
    except Exception:
        pass # It will make missing value on this feature but it's fine

def get_the_succeeding_word(row, window_size = 1):
    '''
    Get the succeeding word given the token. 
    It's a helper function to compute the sequential feature of the word.
    '''
    try:
        the_latter_ix = row.item_name.split().index(row.tokens) + window_size
        if the_latter_ix >= len(row.item_name.split()):
            return -1 # It means the latter word is non-existent. 
        else:
            return row.item_name.split()[the_latter_ix]
    except Exception:
        pass # It will make missing value on this feature but it's fine

def succeeding_2_gram_given_current_token(row, esBigramFreq):
    if row.the_succeeding_word_given_current_token_w_1 == -1:
        return -1
    else:
        key = (row.tokens.lower(), row.the_succeeding_word_given_current_token_w_1.lower())
        return esBigramFreq[key]
    return row

def preceding_2_gram_given_current_token(row, esBigramFreq):
    if row.the_preceding_word_given_current_token_w_1 == -1:
        return -1
    else:
        key = (row.tokens.lower(), row.the_preceding_word_given_current_token_w_1.lower())
        return esBigramFreq[key]
    return row

def preceding_3_gram_given_current_token(row, esTrigramFreq):
    if row.the_preceding_word_given_current_token_w_1 != -1 and row.the_preceding_word_given_current_token_w_2 != -1:
        key = (row.tokens.lower(), row.the_preceding_word_given_current_token_w_1.lower(),
               row.the_preceding_word_given_current_token_w_2.lower())
        return esTrigramFreq[key]
    else:
        return -1
    return row

def succeeding_3_gram_given_current_token(row, esTrigramFreq):
    if row.the_succeeding_word_given_current_token_w_1 != -1 and row.the_succeeding_word_given_current_token_w_2 != -1:
        key = (row.tokens.lower(), row.the_succeeding_word_given_current_token_w_1.lower(),
               row.the_succeeding_word_given_current_token_w_2.lower())
        return esTrigramFreq[key]
    else:
        return -1
    return row

def preceding_4_gram_given_current_token(row, esFgramFreq):
    if (row.the_preceding_word_given_current_token_w_1 != -1) \
    and (row.the_preceding_word_given_current_token_w_2 != -1) \
    and (row.the_preceding_word_given_current_token_w_3 != -1):
        key = (row.tokens.lower(), 
               row.the_preceding_word_given_current_token_w_1.lower(),
               row.the_preceding_word_given_current_token_w_2.lower(),
               row.the_preceding_word_given_current_token_w_3.lower())
        return esFgramFreq[key]
    else:
        return -1
    return row

def succeeding_4_gram_given_current_token(row, esFgramFreq):
    if (row.the_succeeding_word_given_current_token_w_1 != -1) \
    and (row.the_succeeding_word_given_current_token_w_2 != -1) \
    and (row.the_succeeding_word_given_current_token_w_3 != -1):
        key = (row.tokens.lower(), 
               row.the_succeeding_word_given_current_token_w_1.lower(),
               row.the_succeeding_word_given_current_token_w_2.lower(),
               row.the_succeeding_word_given_current_token_w_3.lower())
        return esFgramFreq[key]
    else:
        return -1
    return row

def preceding_5_gram_given_current_token(row, esFivegramFreq):
    if (row.the_preceding_word_given_current_token_w_1 != -1) \
    and (row.the_preceding_word_given_current_token_w_2 != -1) \
    and (row.the_preceding_word_given_current_token_w_3 != -1) \
    and (row.the_preceding_word_given_current_token_w_4 != -1):
        key = (row.tokens.lower(), 
               row.the_preceding_word_given_current_token_w_1.lower(),
               row.the_preceding_word_given_current_token_w_2.lower(),
               row.the_preceding_word_given_current_token_w_3.lower(),
               row.the_preceding_word_given_current_token_w_4.lower())
        return esFivegramFreq[key]
    else:
        return -1
    return row

def succeeding_5_gram_given_current_token(row, esFivegramFreq):
    if (row.the_succeeding_word_given_current_token_w_1 != -1) \
    and (row.the_succeeding_word_given_current_token_w_2 != -1) \
    and (row.the_succeeding_word_given_current_token_w_3 != -1) \
    and (row.the_succeeding_word_given_current_token_w_4 != -1):
        key = (row.tokens.lower(), 
               row.the_succeeding_word_given_current_token_w_1.lower(),
               row.the_succeeding_word_given_current_token_w_2.lower(),
               row.the_succeeding_word_given_current_token_w_3.lower(),
               row.the_succeeding_word_given_current_token_w_4.lower())
        return esFivegramFreq[key]
    else:
        return -1
    return row

def preceding_6_gram_given_current_token(row, esSixgramFreq):
    if (row.the_preceding_word_given_current_token_w_1 != -1) \
    and (row.the_preceding_word_given_current_token_w_2 != -1) \
    and (row.the_preceding_word_given_current_token_w_3 != -1) \
    and (row.the_preceding_word_given_current_token_w_4 != -1) \
    and (row.the_preceding_word_given_current_token_w_5 != -1):
        key = (row.tokens.lower(), 
               row.the_preceding_word_given_current_token_w_1.lower(),
               row.the_preceding_word_given_current_token_w_2.lower(),
               row.the_preceding_word_given_current_token_w_3.lower(),
               row.the_preceding_word_given_current_token_w_4.lower(),
               row.the_preceding_word_given_current_token_w_5.lower())
        return esFivegramFreq[key]
    else:
        return -1
    return row

def succeeding_6_gram_given_current_token(row, esSixgramFreq):
    if (row.the_succeeding_word_given_current_token_w_1 != -1) \
    and (row.the_succeeding_word_given_current_token_w_2 != -1) \
    and (row.the_succeeding_word_given_current_token_w_3 != -1) \
    and (row.the_succeeding_word_given_current_token_w_4 != -1) \
    and (row.the_succeeding_word_given_current_token_w_5 != -1):
        key = (row.tokens.lower(), 
               row.the_succeeding_word_given_current_token_w_1.lower(),
               row.the_succeeding_word_given_current_token_w_2.lower(),
               row.the_succeeding_word_given_current_token_w_3.lower(),
               row.the_succeeding_word_given_current_token_w_4.lower(),
               row.the_succeeding_word_given_current_token_w_5.lower())
        return esFivegramFreq[key]
    else:
        return -1
    return row

def succeeding_d_2_k_2_given_current_token(row, esBigramFreq):
    '''
    The idea refered to the dlitated CNN. 
    dilation_rate : 2
    kernel_width: 2
    direction: forward
    '''
    if row.the_succeeding_word_given_current_token_w_2 == -1:
        return -1
    else:
        key = (row.tokens.lower(), row.the_succeeding_word_given_current_token_w_2.lower())
        return esBigramFreq[key]
    return row

def preceding_d_2_k_2_given_current_token(row, esBigramFreq):
    '''
    The idea refered to the dlitated CNN. 
    dilation_rate : 2
    kernel_width: 2
    direction: backward
    '''
    if row.the_preceding_word_given_current_token_w_2 == -1:
        return -1
    else:
        key = (row.tokens.lower(), row.the_preceding_word_given_current_token_w_2.lower())
        return esBigramFreq[key]
    return row

def succeeding_d_3_k_2_given_current_token(row, esBigramFreq):
    '''
    The idea refered to the dlitated CNN. 
    dilation_rate : 3
    kernel_width: 2
    direction: forward
    '''
    if row.the_succeeding_word_given_current_token_w_3 == -1:
        return -1
    else:
        key = (row.tokens.lower(), row.the_succeeding_word_given_current_token_w_3.lower())
        return esBigramFreq[key]
    return row

def preceding_d_3_k_2_given_current_token(row, esBigramFreq):
    '''
    The idea refered to the dlitated CNN. 
    dilation_rate : 3
    kernel_width: 2
    direction: backward
    '''
    if row.the_preceding_word_given_current_token_w_3 == -1:
        return -1
    else:
        key = (row.tokens.lower(), row.the_preceding_word_given_current_token_w_3.lower())
        return esBigramFreq[key]
    return row

def succeeding_d_4_k_2_given_current_token(row, esBigramFreq):
    '''
    The idea refered to the dlitated CNN. 
    dilation_rate : 4 For example, it would extract relationship between w0 and w-4.
    kernel_width: 2
    direction: forward
    '''
    if row.the_succeeding_word_given_current_token_w_4 == -1:
        return -1
    else:
        key = (row.tokens.lower(), row.the_succeeding_word_given_current_token_w_4.lower())
        return esBigramFreq[key]
    return row

def preceding_d_4_k_2_given_current_token(row, esBigramFreq):
    '''
    The idea refered to the dlitated CNN. 
    dilation_rate : 4
    kernel_width: 2
    direction: backward
    '''
    if row.the_preceding_word_given_current_token_w_4 == -1:
        return -1
    else:
        key = (row.tokens.lower(), row.the_preceding_word_given_current_token_w_4.lower())
        return esBigramFreq[key]
    return row

def succeeding_d_5_k_2_given_current_token(row, esBigramFreq):
    if row.the_succeeding_word_given_current_token_w_5 == -1:
        return -1
    else:
        key = (row.tokens.lower(), row.the_succeeding_word_given_current_token_w_5.lower())
        return esBigramFreq[key]
    return row

def preceding_d_5_k_2_given_current_token(row, esBigramFreq):
    if row.the_preceding_word_given_current_token_w_5 == -1:
        return -1
    else:
        key = (row.tokens.lower(), row.the_preceding_word_given_current_token_w_5.lower())
        return esBigramFreq[key]
    return row

def succeeding_d_6_k_2_given_current_token(row, esBigramFreq):
    if row.the_succeeding_word_given_current_token_w_6 == -1:
        return -1
    else:
        key = (row.tokens.lower(), row.the_succeeding_word_given_current_token_w_6.lower())
        return esBigramFreq[key]
    return row

def preceding_d_6_k_2_given_current_token(row, esBigramFreq):
    if row.the_preceding_word_given_current_token_w_6 == -1:
        return -1
    else:
        key = (row.tokens.lower(), row.the_preceding_word_given_current_token_w_6.lower())
        return esBigramFreq[key]
    return row

# kernel_width = 3

def succeeding_d_2_k_3_given_current_token(row, esTrigramFreq):
    '''
    The idea refered to the dlitated CNN. 
    dilation_rate : 2
    kernel_width: 3
    direction: forward
    '''
    if (row.the_succeeding_word_given_current_token_w_2 == -1) or (row.the_succeeding_word_given_current_token_w_4 == -1):
        return -1
    else:
        key = (row.tokens.lower(), row.the_succeeding_word_given_current_token_w_2.lower(), row.the_succeeding_word_given_current_token_w_4.lower())
        return esTrigramFreq[key]
    return row

def preceding_d_2_k_3_given_current_token(row, esTrigramFreq):
    '''
    The idea refered to the dlitated CNN. 
    dilation_rate : 2
    kernel_width: 3
    direction: backward
    '''
    if (row.the_preceding_word_given_current_token_w_2 == -1) or (row.the_preceding_word_given_current_token_w_4 == -1):
        return -1
    else:
        key = (row.tokens.lower(), row.the_preceding_word_given_current_token_w_2.lower(), row.the_preceding_word_given_current_token_w_4.lower())
        return esTrigramFreq[key]
    return row

def succeeding_d_3_k_3_given_current_token(row, esTrigramFreq):
    '''
    The idea refered to the dlitated CNN. 
    dilation_rate : 3
    kernel_width: 3
    direction: forward
    '''
    if (row.the_succeeding_word_given_current_token_w_3 == -1) or (row.the_succeeding_word_given_current_token_w_6 == -1):
        return -1
    else:
        key = (row.tokens.lower(), row.the_succeeding_word_given_current_token_w_3.lower(), row.the_succeeding_word_given_current_token_w_6.lower())
        return esTrigramFreq[key]
    return row

def preceding_d_3_k_3_given_current_token(row, esTrigramFreq):
    '''
    The idea refered to the dlitated CNN. 
    dilation_rate : 3
    kernel_width: 3
    direction: backward
    '''
    if (row.the_preceding_word_given_current_token_w_3 == -1) or (row.the_preceding_word_given_current_token_w_6 == -1):
        return -1
    else:
        key = (row.tokens.lower(), row.the_preceding_word_given_current_token_w_3.lower(), row.the_preceding_word_given_current_token_w_6.lower())
        return esTrigramFreq[key]
    return row

def pos_tagger(df):
    '''
    High-level pos-tagging
    '''
    try:
        tagged_sent = tag.pos_tag(df.tokens.tolist())
    except Exception:
        print (df.item_name.iloc[0])
        print (df.tokens.tolist())
    df['pos_tagger'] = [pos for token, pos in tagged_sent]
    df['pos_tagger'] = [pos.replace("''", '$') for pos in df.pos_tagger.tolist()]
    return df

def get_wordnet_pos(treebank_tag):
    '''
    It map the treebank tags to WordNet part of speech names
    '''
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        # As default pos in lemmatization is Noun
        return wordnet.NOUN

def parallelize_dataframe(df, func):
    '''
    speeding up DataFrame.apply() via parallelizing.

    '''
    #---------------
    # setting
    #---------------
    num_partitions = 10
    num_cores = 10

    # core
    df1,df2,df3,df4,df5,df6,df7,df8,df9,df10 = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, [df1,df2,df3,df4,df5,df6,df7,df8,df9,df10]))
    pool.close()
    pool.join()
    return df

def speed_up_func_for_preprocessing(df):
    '''
    Put the columns u need to apply()
    
    data: DataFrame
    '''
    df['the_preceding_word_given_current_token_w_1'] = df.apply(lambda x: get_the_preceding_word(x, window_size = 1), axis = 1)
    df['the_succeeding_word_given_current_token_w_1'] = df.apply(lambda x: get_the_succeeding_word(x, window_size = 1), axis = 1)
    df['the_preceding_word_given_current_token_w_2'] = df.apply(lambda x: get_the_preceding_word(x, window_size = 2), axis = 1)
    df['the_succeeding_word_given_current_token_w_2'] = df.apply(lambda x: get_the_succeeding_word(x, window_size = 2), axis = 1)
    df['the_preceding_word_given_current_token_w_3'] = df.apply(lambda x: get_the_preceding_word(x, window_size = 3), axis = 1)
    df['the_succeeding_word_given_current_token_w_3'] = df.apply(lambda x: get_the_succeeding_word(x, window_size = 3), axis = 1)
    df['the_preceding_word_given_current_token_w_4'] = df.apply(lambda x: get_the_preceding_word(x, window_size = 4), axis = 1)
    df['the_succeeding_word_given_current_token_w_4'] = df.apply(lambda x: get_the_succeeding_word(x, window_size = 4), axis = 1)
    # increase the window_size
    df['the_preceding_word_given_current_token_w_5'] = df.apply(lambda x: get_the_preceding_word(x, window_size = 5), axis = 1)
    df['the_succeeding_word_given_current_token_w_5'] = df.apply(lambda x: get_the_succeeding_word(x, window_size = 5), axis = 1)
    df['the_preceding_word_given_current_token_w_6'] = df.apply(lambda x: get_the_preceding_word(x, window_size = 6), axis = 1)
    df['the_succeeding_word_given_current_token_w_6'] = df.apply(lambda x: get_the_succeeding_word(x, window_size = 6), axis = 1)
    return df

def speed_up_func_for_feature_engineering(df):
    '''
    Put the columns u need to apply()
    
    data: DataFrame
    '''
    #------------
    # like what the standard cnn did. (Continuous)
    #------------

    # succeeding_2_gram_given_current_token
    df['succeeding_2_gram_given_current_token'] = df.apply(lambda x: succeeding_2_gram_given_current_token(x, esBigramFreq = esBigramFreq), axis = 1) 
    # preceding_2_gram_given_current_token
    df['preceding_2_gram_given_current_token'] = df.apply(lambda x: preceding_2_gram_given_current_token(x, esBigramFreq = esBigramFreq), axis = 1) 
    # succeeding_3_gram_given_current_token
    df['succeeding_3_gram_given_current_token'] = df.apply(lambda x: succeeding_3_gram_given_current_token(x, esTrigramFreq = esTrigramFreq), axis = 1) 
    # preceding_3_gram_given_current_token
    df['preceding_3_gram_given_current_token'] = df.apply(lambda x: preceding_3_gram_given_current_token(x, esTrigramFreq = esTrigramFreq), axis = 1) 
    # succeeding_4_gram_given_current_token
    df['succeeding_4_gram_given_current_token'] = df.apply(lambda x: succeeding_4_gram_given_current_token(x, esFgramFreq = esFgramFreq), axis = 1) 
    # preceding_4_gram_given_current_token
    df['preceding_4_gram_given_current_token'] = df.apply(lambda x: preceding_4_gram_given_current_token(x, esFgramFreq = esFgramFreq), axis = 1) 
    # succeeding_5_gram_given_current_token
    df['succeeding_5_gram_given_current_token'] = df.apply(lambda x: succeeding_5_gram_given_current_token(x, esFivegramFreq = esFivegramFreq), axis = 1) 
    # preceding_5_gram_given_current_token
    df['preceding_5_gram_given_current_token'] = df.apply(lambda x: preceding_5_gram_given_current_token(x, esFivegramFreq = esFivegramFreq), axis = 1) 
    # succeeding_6_gram_given_current_token
    df['succeeding_6_gram_given_current_token'] = df.apply(lambda x: succeeding_6_gram_given_current_token(x, esSixgramFreq = esSixgramFreq), axis = 1) 
    # preceding_6_gram_given_current_token
    df['preceding_6_gram_given_current_token'] = df.apply(lambda x: preceding_6_gram_given_current_token(x, esSixgramFreq = esSixgramFreq), axis = 1) 

    #------------
    # like what the dilatedd cnn did. (Descrete)
    #------------

    # kernel_width = 2, stride = 1, dilation_rate = 2~6, bidirectional: succeeding and preceeding

    df['succeeding_d_2_k_2_given_current_token'] = df.apply(lambda x: succeeding_d_2_k_2_given_current_token(x, esBigramFreq = esBigramFreq), axis = 1) 
    df['preceding_d_2_k_2_given_current_token'] = df.apply(lambda x: preceding_d_2_k_2_given_current_token(x, esBigramFreq = esBigramFreq), axis = 1) 
    df['succeeding_d_3_k_2_given_current_token'] = df.apply(lambda x: succeeding_d_3_k_2_given_current_token(x, esBigramFreq = esBigramFreq), axis = 1) 
    df['preceding_d_3_k_2_given_current_token'] = df.apply(lambda x: preceding_d_3_k_2_given_current_token(x, esBigramFreq = esBigramFreq), axis = 1) 
    df['succeeding_d_4_k_2_given_current_token'] = df.apply(lambda x: succeeding_d_4_k_2_given_current_token(x, esBigramFreq = esBigramFreq), axis = 1) 
    df['preceding_d_4_k_2_given_current_token'] = df.apply(lambda x: preceding_d_4_k_2_given_current_token(x, esBigramFreq = esBigramFreq), axis = 1) 
    df['succeeding_d_5_k_2_given_current_token'] = df.apply(lambda x: succeeding_d_5_k_2_given_current_token(x, esBigramFreq = esBigramFreq), axis = 1) 
    df['preceding_d_5_k_2_given_current_token'] = df.apply(lambda x: preceding_d_5_k_2_given_current_token(x, esBigramFreq = esBigramFreq), axis = 1) 
    df['succeeding_d_6_k_2_given_current_token'] = df.apply(lambda x: succeeding_d_6_k_2_given_current_token(x, esBigramFreq = esBigramFreq), axis = 1) 
    df['preceding_d_6_k_2_given_current_token'] = df.apply(lambda x: preceding_d_6_k_2_given_current_token(x, esBigramFreq = esBigramFreq), axis = 1) 

    # kernel_width = 3, stride = 1, dilation_rate = 2,3, bidirectional: succeeding and preceeding

    df['succeeding_d_2_k_3_given_current_token'] = df.apply(lambda x: succeeding_d_2_k_3_given_current_token(x, esTrigramFreq = esTrigramFreq), axis = 1) 
    df['preceding_d_2_k_3_given_current_token'] = df.apply(lambda x: preceding_d_2_k_3_given_current_token(x, esTrigramFreq = esTrigramFreq), axis = 1) 
    df['succeeding_d_3_k_3_given_current_token'] = df.apply(lambda x: succeeding_d_3_k_3_given_current_token(x, esTrigramFreq = esTrigramFreq), axis = 1) 
    df['preceding_d_3_k_3_given_current_token'] = df.apply(lambda x: preceding_d_3_k_3_given_current_token(x, esTrigramFreq = esTrigramFreq), axis = 1) 

    return df

def contextual_similarity(T):
	'''
	It's for using multi preprosessing to speed up feature extracting process.

	parameters:
	---------------------
	T: int. 1, 2, ..
	'''
	LEMMATIZING = False
	# preprocessed_data_path
	input_base_path = '../data/processed'

	#--------------------
	# laod data including label
	#--------------------	
	if T == 1:
		name = 'lazada_and_amazon' 
		df = pd.read_csv(os.path.join(input_base_path, 'mobile_training_w_word_id.csv'))
	elif T == 2:
		name = 'personal_care_and_beauty'
		df = pd.read_csv(os.path.join(input_base_path, 'personal_care_and_beauty.csv'))
	elif T == 3:
		name = 'beauty_amazon'
		df = pd.read_csv(os.path.join(input_base_path, 'beauty_amazon.csv'))
	elif T == 4:
		name = 'tv_laptop_amazon'
		df = pd.read_csv(os.path.join(input_base_path, 'tv_laptop_amazon.csv'))
	else:
		pass

	logging.info('input shape / {} : {}'.format(name, df.shape))
	#-------------------------
	# drop itemname and tokens with nan
	#-------------------------
	df.dropna(subset = ['item_name', 'tokens','clean_tokens','item_id', 'word_id'], axis = 0, inplace = True)
	#--------------------------
	# conver type
	#--------------------------
	df['tokens'] = df.tokens.astype(str)

	#--------------------------
	# preprocessing for contextual information
	#--------------------------

	df = parallelize_dataframe(df, speed_up_func_for_preprocessing)

	if LEMMATIZING == True:
		print ('LEMMATIZING : {}'.format(LEMMATIZING))
		lemmatizer = WordNetLemmatizer()
		df = df.groupby('item_name').apply(pos_tagger)
		# get_wordnet_pos
		df['wordnet_pos'] = df.pos_tagger.apply(get_wordnet_pos)
		# lemmatizing
		df['lemma'] = [lemmatizer.lemmatize(t.lower(), pos) for t, pos in zip(df.tokens.tolist(), df.wordnet_pos.tolist())]
		# drop columns that we don't need it
		df.drop(['pos_tagger','wordnet_pos'], axis = 1, inplace = True)
		# input of n-gram
		tokenized = [t.lower() for t in df.lemma.tolist()]
		# drop
		df.drop(['lemma'], axis = 1, inplace = True)
		gc.collect()
	else:
		# input of n-gram
		tokenized = [t.lower() for t in df.tokens.tolist()]

	#----------------------------
	# n-grame generator
	#----------------------------
	esBigrams = ngrams(tokenized, 2) # generater
	esTrigrams = ngrams(tokenized, 3) # generater
	esFgrams = ngrams(tokenized, 4) # generater
	esFivegrams = ngrams(tokenized, 5) # generater
	esSixegrams = ngrams(tokenized, 6) # generater

	# make the variables global
	global esBigramFreq
	global esTrigramFreq
	global esFgramFreq
	global esFivegramFreq
	global esSixgramFreq
	#----------------------------
	# get the frequency of each bigram in our corpus
	#----------------------------
	esBigramFreq = collections.Counter(esBigrams)
	esTrigramFreq = collections.Counter(esTrigrams)
	esFgramFreq = collections.Counter(esFgrams)
	esFivegramFreq = collections.Counter(esFivegrams)
	esSixgramFreq = collections.Counter(esSixegrams)

	##################################
	# feature engineering
	##################################
	df = parallelize_dataframe(df, speed_up_func_for_feature_engineering)

	col_need_to_be_drop = ['the_preceding_word_given_current_token_w_1','the_succeeding_word_given_current_token_w_1',
	'the_preceding_word_given_current_token_w_2','the_succeeding_word_given_current_token_w_2',
	'the_preceding_word_given_current_token_w_3','the_succeeding_word_given_current_token_w_3',
	'the_preceding_word_given_current_token_w_4','the_succeeding_word_given_current_token_w_4',
	'the_preceding_word_given_current_token_w_5','the_succeeding_word_given_current_token_w_5',
	'the_preceding_word_given_current_token_w_6','the_succeeding_word_given_current_token_w_6',
	]

	df.drop(col_need_to_be_drop, axis = 1, inplace = True)

	logging.info('output shape / {}: {}'.format(name, df.shape))

	#-------------------------
	# remove no need columns
	#-------------------------
	df.drop(['is_brand', 'is_valid','clean_tokens','item_id', 'word_id'], axis = 1 , inplace = True)
	#-------------------------
	# save
	#-------------------------
	output_dir = '../features/{}'.format(name)
	if not os.path.isdir(output_dir):
		os.makedirs(output_dir)
	df.to_csv('../features/{}/contextual_similarity.csv'.format(name), index = False)
	logging.info('finish saving {}'.format(name))

def multi(T):
    '''
    It's for using multi preprosessing to speed up each model training process.

    parameters:
    ---------------------
    T: int. 1, 2, 3, and 4.
    '''
    contextual_similarity(T)

if __name__ == '__main__':
    ##################################################
    # Main
    ##################################################
    import sys
    sys.path.append('../models')
    from tf_utils import init_logging
    import logging
    #--------------------
    # setting
    #--------------------
    # log path
    log_dir = 'log/'
    init_logging(log_dir)
    #--------------------
    # core
    #--------------------
    s = time.time()

    #----------------------------------------
    # way 1
    #----------------------------------------
    mp_pool = Pool(1)
    mp_pool.map(multi, [1])
    mp_pool.close()

    # #----------------------------------------
    # # way 2 --->250.94354033470154  secs
    # #----------------------------------------
    # for i in [1,2,3,4]:
    #     multi(i)
    e = time.time()
    print (e-s, ' secs')


