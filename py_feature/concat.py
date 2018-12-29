#! /usr/bin/env python3
import time
import pandas as pd
import numpy as np
import os
import gc
#from datetime import datetime # for the newest version control
import multiprocessing as mp # for speeding up some process
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

def tokens_characteristics(df, name):
	'''
	read features and merge different feature together.

	parameters:
	---------------------
	df: DataFrame
	name: train or test
	'''
	tokens_characteristics = pd.read_csv('../features/{}/tokens_characteristics.csv'.format(name))
	tokens_characteristics.dropna(subset = ['item_name', 'tokens'], axis = 0, inplace = True)
	tokens_characteristics.drop_duplicates(subset = ['item_name', 'tokens'], inplace = True)

	df = pd.merge(df, tokens_characteristics, 
		on = ['item_name', 'tokens'],
		how = 'left'
		)
	return df

def tokens_context(df, name):
	'''
	read features and merge different feature together.

	parameters:
	---------------------
	df: DataFrame
	name: train or test
	'''
	tokens_context = pd.read_csv('../features/{}/tokens_context.csv'.format(name))
	tokens_context.dropna(subset = ['item_name', 'tokens'], axis = 0, inplace = True)
	tokens_context.drop_duplicates(subset = ['item_name', 'tokens'], inplace = True)

	df = pd.merge(df, tokens_context, 
		on = ['item_name', 'tokens'],
		how = 'left'
		)
	return df

def contextual_token_level(df, name):
	'''
	read features and merge different feature together.

	parameters:
	---------------------
	df: DataFrame
	name: train or test
	'''
	contextual_token_level =  pd.read_csv('../features/{}/contextual_features.csv'.format(name))
	contextual_token_level.dropna(subset = ['item_name', 'tokens'], axis = 0, inplace = True)
	contextual_token_level.drop_duplicates(subset = ['item_name', 'tokens'], inplace = True)

	df = pd.merge(df, contextual_token_level, 
		on = ['item_name', 'tokens'],
		how = 'left'
		)
	return df

def tokens_contextual_similarity(df, name):
	'''
	read features and merge different feature together.

	parameters:
	---------------------
	df: DataFrame
	name: train or test
	'''
	tokens_contextual_similarity =  pd.read_csv('../features/{}/contextual_similarity.csv'.format(name))
	tokens_contextual_similarity.dropna(subset = ['item_name', 'tokens'], axis = 0, inplace = True)
	tokens_contextual_similarity.drop_duplicates(subset = ['item_name', 'tokens'], inplace = True)

	df = pd.merge(df, tokens_contextual_similarity, 
		on = ['item_name', 'tokens'],
		how = 'left'
		)
	return df

def word_vector(df, name):
	'''
	read features and merge different feature together.

	parameters:
	---------------------
	df: DataFrame
	name: train or test
	'''
	word_vector =  pd.read_hdf('../features/{}/word_vector.h5'.format(name))
	word_vector.dropna(subset = ['item_name', 'tokens'], axis = 0, inplace = True)
	word_vector.drop_duplicates(subset = ['item_name', 'tokens'], inplace = True)

	df = pd.merge(df, word_vector, 
		on = ['item_name', 'tokens'],
		how = 'left'
		)
	return df

def tf_idf(df, name):
	'''
	read features and merge different feature together.

	parameters:
	---------------------
	df: DataFrame
	name: train or test
	'''
	tf_idf =  pd.read_csv('../features/{}/tf_idf.csv'.format(name))
	tf_idf.dropna(subset = ['item_name', 'tokens'], axis = 0, inplace = True)
	tf_idf.drop_duplicates(subset = ['item_name', 'tokens'], inplace = True)
	tf_idf.rename(columns = {'tokens': 'tokens_lower'}, inplace = True)
	#-----------
	# merge
	#-----------
	df['tokens_lower'] = df.tokens.apply(lambda x: x.lower() if type(x)==str else x)
	df = pd.merge(df, tf_idf, 
		on = ['item_name', 'tokens_lower'],
		how = 'left'
		)
	# drop auxiliary field
	df.drop(['tokens_lower'], axis = 1 , inplace = True)
	return df

def concat_all_features(T):
	'''
	It's for using multi preprosessing to speed up concating feature process.

	parameters:
	---------------------
	T: int. 1 or -1, 0
	'''
	#--------------------
	# path setting
	#--------------------
	# preprocessed_data_path
	input_base_path = '../data/processed'

	#--------------------
	# laod data including label
	#--------------------	
	if T == 1:
		name = 'lazada_and_amazon'
		input_col = ['item_name', 'tokens', 'is_brand', 'is_valid','clean_tokens','item_id', 'word_id']
		df = pd.read_csv(os.path.join(input_base_path, '{}.csv'.format('mobile_training_w_word_id')))[input_col]
	elif T == 2:
		name = 'personal_care_and_beauty'
		input_col = ['item_name', 'tokens', 'is_brand', 'is_valid']
		df = pd.read_csv(os.path.join(input_base_path, '{}.csv'.format(name)))[input_col]
	elif T == 3:
		name = 'tv_laptop_amazon'
		input_col = ['item_name', 'tokens', 'is_brand', 'is_valid']
		df = pd.read_csv(os.path.join(input_base_path, '{}.csv'.format(name)))[input_col]
	elif T == 4:
		name = 'beauty_amazon'
		input_col = ['item_name', 'tokens', 'is_brand', 'is_valid']
		df = pd.read_csv(os.path.join(input_base_path, '{}.csv'.format(name)))[input_col]
	else:
		pass
	#--------------------
	# preprocessing
	#--------------------
	# drop nan
	df.dropna(subset = ['item_name', 'tokens'], axis = 0, inplace = True)
	# drop duplicated 
	df.drop_duplicates(subset = ['item_name', 'tokens'], inplace = True)
	# change columns name
	df.rename(columns = {'is_brand' : 'label'}, inplace = True)

	#--------------------
	# concat features
	#--------------------	
	df = tokens_characteristics(df, name)
	print ('tokens_characteristics features : {} / {}'.format(df.shape, name))
	# df = tokens_context(df, name)
	# print ('tokens_context features : {} / {}'.format(df.shape, name))
	df = contextual_token_level(df, name)
	print ('contextual features : {} / {}'.format(df.shape, name))
	df = tokens_contextual_similarity(df, name)
	print ('contextual_similarity features : {} / {}'.format(df.shape, name))
	df = word_vector(df, name) # done
	print ('word_vector features : {} / {}'.format(df.shape, name))
	df = tf_idf(df, name) # done
	print ('tf_idf features : {} / {}'.format(df.shape, name))

	#--------------------
	# feature engineering(Grouby)
	#--------------------

	#--------------------
	# output
	#--------------------
	df.dropna(subset = ['item_name', 'tokens'], axis = 0, inplace = True)
	if output_format == 'hdf':
		df.to_hdf('../features/{}/all_features.h5'.format(name), 'all_features')
	elif output_format == 'csv':
		df.to_csv('../features/{}/all_features.csv'.format(name), index = False)
	# logging
	print ('output all features : {} / {}'.format(df.shape, name))

def multi(T):
	'''
	It's for using multi preprosessing to speed up concating feature process.

	parameters:
	---------------------
	T: int. 1 or -1 for telling difference between tv_and_laptop and personal_care_and_beauty.
	'''
	concat_all_features(T)

if __name__ == '__main__':
	##################################################
	# Main
	##################################################
	s = time.time()
	output_format = 'hdf'
	# mp_pool = mp.Pool(1)
	# mp_pool.map(multi, [1])
	multi(1)
	e = time.time()
	print (e-s, ' secs ')






