#! /usr/bin/env python3
"""
Created on Sep 11 2018

In a large text corpus, stopwords words will be very present (e.g. “the”, “a”, “is” in English) hence carrying very little meaningful information about the actual contents of the document. 
So we re-weight the count features via tf-idf represent the meaning of the word.

The higher tf-idf, The more representitive the word is in the title of the item_name.

Reference:
	-explanation of tf-idf: https://medium.freecodecamp.org/how-to-process-textual-data-using-tf-idf-in-python-cd2bbc0a94a3

@author: Ray

"""

import pandas as pd
import numpy as np
from datetime import datetime # for the newest version control
import os
import time
import multiprocessing as mp # for speeding up some process
import gc
from billiard import Pool
from sklearn.feature_extraction.text import TfidfTransformer
import sys
sys.path.append('../models')
from tf_utils import init_logging
import logging
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

def get_count_metrix(df, wordSet):
    '''
    return the matrix, row is number of item_names, column is number of words aka size of vocabulary.
    Note:
        the element in matrix is called term frequency given the item title.
    args:
    ---------
    df: DataFrame
    wordSet: set
    '''
    # initalize empty dict
    wordDictA = dict.fromkeys(wordSet, 0) 
    # corpust states computation
    for word in df.tokens.tolist():
        wordDictA[word]+=1
    return pd.DataFrame([wordDictA])

def parallelize_dataframe(df, func, name):
    '''
    speeding up DataFrame.apply() via parallelizing.

    '''
    if name == 'beauty_amazon':
        #---------------
        # setting
        #---------------
        num_partitions = 32
        num_cores = 32

        # core
        item_name_ls = list(df.item_name.unique())
        item_name_num = df.item_name.nunique()
        n = int(item_name_num /num_partitions)
        # split df based on item_name
        df1 = df[df.item_name.isin(item_name_ls[:1*n])]
        df2 = df[df.item_name.isin(item_name_ls[1*n:2*n])]
        df3 = df[df.item_name.isin(item_name_ls[2*n:3*n])]
        df4 = df[df.item_name.isin(item_name_ls[3*n:4*n])]
        df5 = df[df.item_name.isin(item_name_ls[4*n:5*n])]
        df6 = df[df.item_name.isin(item_name_ls[5*n:6*n])]
        df7 = df[df.item_name.isin(item_name_ls[6*n:7*n])]
        df8 = df[df.item_name.isin(item_name_ls[7*n:8*n])]
        df9 = df[df.item_name.isin(item_name_ls[8*n:9*n])]
        df10 = df[df.item_name.isin(item_name_ls[9*n:10*n])]
        df11 = df[df.item_name.isin(item_name_ls[10*n:11*n])]
        df12 = df[df.item_name.isin(item_name_ls[11*n:12*n])]
        df13 = df[df.item_name.isin(item_name_ls[12*n:13*n])]
        df14 = df[df.item_name.isin(item_name_ls[13*n:14*n])]
        df15 = df[df.item_name.isin(item_name_ls[14*n:15*n])]
        df16 = df[df.item_name.isin(item_name_ls[15*n:16*n])]
        df17 = df[df.item_name.isin(item_name_ls[16*n:17*n])]
        df18 = df[df.item_name.isin(item_name_ls[17*n:18*n])]
        df19 = df[df.item_name.isin(item_name_ls[18*n:19*n])]
        df20 = df[df.item_name.isin(item_name_ls[19*n:20*n])]
        df21 = df[df.item_name.isin(item_name_ls[20*n:21*n])]
        df22 = df[df.item_name.isin(item_name_ls[21*n:22*n])]
        df23 = df[df.item_name.isin(item_name_ls[22*n:23*n])]
        df24 = df[df.item_name.isin(item_name_ls[23*n:24*n])]
        df25 = df[df.item_name.isin(item_name_ls[24*n:25*n])]
        df26 = df[df.item_name.isin(item_name_ls[25*n:26*n])]
        df27 = df[df.item_name.isin(item_name_ls[26*n:27*n])]
        df28 = df[df.item_name.isin(item_name_ls[27*n:28*n])]
        df29 = df[df.item_name.isin(item_name_ls[28*n:29*n])]
        df30 = df[df.item_name.isin(item_name_ls[29*n:30*n])]
        df31 = df[df.item_name.isin(item_name_ls[30*n:31*n])]
        df32 = df[df.item_name.isin(item_name_ls[31*n:])]
        print ('df1 : {}'.format(df1.shape))
        pool = Pool(num_cores)
        df = pd.concat(pool.map(func, [df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,
                                       df11,df12,df13,df14,df15,df16,df17,df18,df19,df20,
                                       df21,df22,df23,df24,df25,df26,df27,df28,df29,df30,
                                       df31,df32]))
        del df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16,df17,df18,df19,df20,df21,df22,df23,df24,df25,df26,df27,df28,df29,df30,df31,df32
        gc.collect()
        print ('======df=====', df.shape)
        pool.close()
        pool.join()
    	
    else:
        #---------------
        # setting
        #---------------
        num_partitions = 5
        num_cores = 5

        # core
        item_name_ls = list(df.item_name.unique())
        item_name_num = df.item_name.nunique()
        n = int(item_name_num /num_partitions)
        # split df based on item_name
        df1 = df[df.item_name.isin(item_name_ls[:n])]
        df2 = df[df.item_name.isin(item_name_ls[n:2*n])]
        df3 = df[df.item_name.isin(item_name_ls[2*n:3*n])]
        df4 = df[df.item_name.isin(item_name_ls[3*n:4*n])]
        df5 = df[df.item_name.isin(item_name_ls[4*n:])]
        pool = Pool(num_cores)
        df = pd.concat(pool.map(func, [df1,df2,df3,df4,df5]))
        del df1,df2,df3,df4,df5
        gc.collect()
        pool.close()
        pool.join()

    return df

def speed_up_func_for_feature_engineering(df):
    '''
    Put the columns u need to apply()
    
    data: DataFrame
    '''
    df = df.groupby('item_name').apply(lambda x: get_count_metrix(x, wordSet)).reset_index()
    return df

def tf_idf_feature(T):
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
		df = pd.read_csv(os.path.join(input_base_path, 'beauty_amazon.csv')) # 40649 num_items x 87402 words
	elif T == 4:
		name = 'tv_laptop_amazon'
		df = pd.read_csv(os.path.join(input_base_path, 'tv_laptop_amazon.csv')) # 16103 num_items x 8324 words
	else:
		pass

	logging.info('input shape / {} : {}'.format(name, df.shape))
	#-------------------------
	# drop itemname and tokens with nan
	#-------------------------
	df.dropna(subset = ['item_name', 'tokens'], axis = 0, inplace = True)
	#--------------------------
	# conver type
	#--------------------------
	df['tokens'] = df.tokens.astype(str)

	#--------------------------
	# preprocessing for tf_idf_feature 
	#--------------------------
	df['tokens'] = df.tokens.apply(lambda x: x.lower() if type(x)==str else x)
	'''
	if len(wordSet) > 35000:
		remove the low-frequent words (like removing stopwords)
		refer to what sklean did.
	'''
	global wordSet
	wordSet = df.tokens.unique()
	logging.info('size of vocabulary/ {} : {}'.format(name, len(wordSet)))

	##################################
	# feature engineering
	##################################
	tf_idf_path = '../features/{}/tf_idf.h5'.format(name)
	if os.path.exists(tf_idf_path) == True:
		tf_idf_df = pd.read_hdf(tf_idf_path)
	else:
		logging.info('Starting computing tf-idf / {}'.format(name))
		#---------------------------------
		# Step1: obtain tf-idf Dtaframe
		#---------------------------------
		# get count matrix
		df_count = parallelize_dataframe(df, speed_up_func_for_feature_engineering, name)
		df_count.reset_index(inplace=True, drop=True)
		columns_name = df_count.columns.tolist()[1:]
		counts = df_count.values[:,1:]
		item_name_df = df_count[['item_name']]
		del df_count
		gc.collect()
		logging.info('counts/ {} : {}'.format(name, counts.shape))
		# tfidf transform
		transformer = TfidfTransformer(smooth_idf = False)
		tfidf = transformer.fit_transform(counts.tolist())
		del counts
		gc.collect()
		tf_idf_df = pd.DataFrame(tfidf.toarray())
		tf_idf_df.columns = columns_name
		# tf_idf_df
		tf_idf_df = pd.concat([ item_name_df, tf_idf_df], axis = 1).set_index('item_name')
		'''
		for speeding up:
			不需要每次抽feature都要重新計算tf-idf
			1.save tf_idf to tf_idf.h5 for extracting test feature in the future
			3. In the beauty amazon case, we need to use 10 dataframe
		'''

		del tfidf, wordSet
		gc.collect()
		logging.info('Finish tf-idf computation / {}'.format(name))
	#---------------------------------
	# Step2: retrieve the tf_idf given item_name and tokens
	#---------------------------------
	tf_idf = [] # current token
	for ix, row in df.iterrows():
	    i_n = row.item_name
	    t = row.tokens
	    tf_idf.append(tf_idf_df.loc[i_n,t])# loc: for index which is name not int
	df['tf_idf'] = tf_idf
	'''
	extend to contextual -level

	'''
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
	# feature saving
	df.to_csv('../features/{}/tf_idf.csv'.format(name), index = False)
	# tf_idf matrix saving
	tf_idf_df.to_hdf('../features/{}/tf_idf.h5'.format(name), 'tf_idf')
	logging.info('finish saving {}'.format(name))

def multi(T):
    '''
    It's for using multi preprosessing to speed up each model training process.

    parameters:
    ---------------------
    T: int. 1, 2, 3, and 4.
    '''
    tf_idf_feature(T)


if __name__ == '__main__':
    ##################################################
    # Main
    ##################################################

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
    # for i in [3]:
    #     multi(i)

    e = time.time()
    print (e-s, ' secs')
