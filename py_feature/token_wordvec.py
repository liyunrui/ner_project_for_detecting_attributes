#! /usr/bin/env python3
"""
Created on Aug 24 2018

Add word2vec of tokens pretrained by Quanchi Weng
@author: Ray

"""

import pandas as pd
import numpy as np
from datetime import datetime # for the newest version control
import os
import time
import multiprocessing as mp # for speeding up some process
import gc
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

def word_vector(T):
	'''
	It's for using multi preprosessing to speed up feature extracting process.

	parameters:
	---------------------
	T: int. 1, 2, ..
	'''

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

	if DEBUG == True:
		df = df[: 5]
	logging.info('input shape / {} : {}'.format(name, df.shape))
	#-------------------------
	# drop itemname and tokens with nan
	#-------------------------
	df.dropna(subset = ['item_name', 'tokens'], axis = 0, inplace = True)

	#--------------------------
	# conver type
	#--------------------------
	df['tokens'] = df.tokens.astype(str)
	df['clean_tokens'] = df.clean_tokens.astype(str)
	#--------------------------
	# preprocessing for contextual information
	#--------------------------
	#df['tokens_lower_for_merge_wv'] = df.tokens.apply(lambda x: x.lower())
	##################################
	# feature engineering
	##################################

	df = pd.merge(df, word2vec, on = 'clean_tokens', how = 'left')


	# col_need_to_be_drop = [
	# 'tokens_lower_for_merge_wv',
	# ]

	# df.drop(col_need_to_be_drop, axis = 1, inplace = True)
	gc.collect()
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
	df.to_hdf('../features/{}/word_vector.h5'.format(name), 'word_vector')
	logging.info('finish saving {}'.format(name))

def multi(T):
    '''
    It's for using multi preprosessing to speed up each model training process.

    parameters:
    ---------------------
    T: int. 1, 2, 3, and 4.
    '''
    word_vector(T)

if __name__ == '__main__':
    ##################################################
    # Main
    ##################################################
    import sys
    sys.path.append('../models')
    from tf_utils import init_logging
    import logging

    word_embedding_path = '/data/ID_large_wordvec_300_2.h5'
    word2vec = pd.read_hdf(word_embedding_path)
    word2vec.rename(columns = {'word': 'clean_tokens'}, inplace = True)
    #--------------------
    # setting
    #--------------------
    # log path
    log_dir = 'log/'
    init_logging(log_dir)
    #--------------------
    # core
    #--------------------
    DEBUG = False
    s = time.time()
    mp_pool = mp.Pool(1)
    mp_pool.map(multi, [1])
    e = time.time()
    print (e-s, ' secs')





