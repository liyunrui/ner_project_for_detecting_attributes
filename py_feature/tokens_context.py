#! /usr/bin/env python3

import pandas as pd
import numpy as np
import os
import gc
from datetime import datetime # for the newest version control
from collections import Counter # for statistical features
from nltk import tag # for pos_tagging
from nltk.corpus import wordnet # for geting pos of wordnet
from nltk.stem import WordNetLemmatizer
from billiard import Pool
import time
import sys
sys.path.append('../models')
from tf_utils import init_logging
import logging
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

def position_of_the_tokens(row):
    '''
    Returning positon of the token in the item_name
    '''
    try:
        return row.item_name.split().index(row.tokens) + 1
    except Exception:
        pass # It will make missing value on this feature but it's fine

def is_first_token_in_item_name(row):
    '''
    Check if the token is the first token in the itemname.
    '''
    list_of_tokens = row.item_name.split()
    try:
        if list_of_tokens.index(row.tokens) == 0:
            return 1
        else:
            return 0
    except Exception:
    	print ('tokens', row.tokens)
    	print ('list_of_tokens', list_of_tokens)

def is_second_token_in_item_name(row):
    '''
    Check if the token is the second token in the itemname.
    '''
    list_of_tokens = row.item_name.split()
    if list_of_tokens.index(row.tokens) == 1:
        return 1
    else:
        return 0

def position_of_the_tokens_from_the_bottom(row):
    '''
    Returning positon of the token in the item_name
    '''
    try:
        item_name_ls = row.item_name.split()
        item_name_ls.reverse()
        return item_name_ls.index(row.tokens) + 1
    except Exception:
        pass # It will make missing value on this feature but it's fine

def position_of_the_tokens_in_ratio(row):
    '''
    Returning positon of the token in the item_name
    '''
    try:
        return 1.0 * row.item_name.split().index(row.tokens) / len(row.item_name.split())
    except Exception:
        pass # It will make missing value on this feature but it's fine

def len_of_item_name(row):
	'''
	return how many tokens we have given a item name. Maybe the itemname is unbranded, 
	the length of them is shorter.
	'''
	try:
		return len(row.item_name.split())
	except Exception:
		pass # It will make missing value on this feature but it's fine

def one_hot_encoder(df, ignore_feature, nan_as_category = True):
    '''
    It's helper function for pos_tagger to do One-hot encoding for categorical columns with get_dummies.

    paras:
    ----------------
    ignore_feature: list of string.
    nan_as_category: boolean. If we think of nan as a value of a certain field.
    '''
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    categorical_columns = [col for col in categorical_columns if col not in ignore_feature]
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

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

def make_word_idx(item_names):
    '''
    It's a helper function for encode_text.
    
    Return a dict including count of each token happening in the whole dataset.
    
    parameters:
    ------------
    item_names: list, including all tokens in the complete dataset
    
    
    '''
    words = [word.lower() for word in item_names]
    word_counts = Counter(words)

    max_id = 1
    
    word_idx = {}
    for word, count in word_counts.items():
        if count < 3:
            word_idx[word] = 0
        else:
            word_idx[word] = max_id
            max_id += 1

    return word_idx

def encode_text(text, word_idx):
    '''
    encode token into code, thinkg of this as a term freq
    High frequent words usually are noise word.
    
    parameters:
    --------------
    text: str
    word_idx
    '''
    return int(word_idx[text.lower()])

def speed_up_func_for_feature_engineering(df):
    '''
    Put the columns u need to apply()

    data: DataFrame
    '''    
    # position_of_the_tokens_from_the_bottom
    df['position_of_the_tokens_from_the_bottom'] = df.apply(position_of_the_tokens_from_the_bottom, axis = 1)
    # position of the tokens
    df['position_of_the_tokens'] = df.apply(position_of_the_tokens, axis = 1)

    # position_of_the_tokens_in_ratio
    df['position_of_the_tokens_in_ratio'] = df.apply(position_of_the_tokens_in_ratio, axis = 1)
    # len_of_item_name
    df['len_of_item_name'] = df.apply(len_of_item_name, axis = 1)

    df, cat_cols_tv_shopee = one_hot_encoder(df, 
                                          nan_as_category = False, 
                                          ignore_feature = ['item_name', 'tokens', 'is_brand', 'is_valid',
                                          'clean_tokens','item_id', 'word_id'])
    df['is_first_token_in_item_name'] = df.apply(is_first_token_in_item_name, axis = 1)
    # is_second_token_in_item_name
    df['is_second_token_in_item_name'] = df.apply(is_second_token_in_item_name, axis = 1)

    return df

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

def tokens_context(T):
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
    # create word_idx for token_freq features
    #--------------------------
    if LEMMATIZING == True:
        print ('LEMMATIZING : {}'.format(LEMMATIZING))
        lemmatizer = WordNetLemmatizer()
        # get_wordnet_pos
        df['wordnet_pos'] = df.pos_tagger.apply(get_wordnet_pos)
        # lemmatizing
        df['lemma'] = [lemmatizer.lemmatize(t.lower(), pos) for t, pos in zip(df.tokens.tolist(), df.wordnet_pos.tolist())]
        # drop columns that we don't need it
        df.drop(['wordnet_pos'], axis = 1, inplace = True)
        # make word_index
        word_idx_for_tv_and_laptop = make_word_idx(pd.concat([df], axis = 0).lemma.tolist())
        # token_freq
        df['token_freq'] = df['lemma'].map(lambda x: encode_text(x, word_idx_for_tv_and_laptop))
        df.drop(['lemma'], axis = 1, inplace = True)
    else:
        word_idx_for_tv_and_laptop = make_word_idx(pd.concat([df], axis = 0).tokens.tolist())
        # token_freq
        df['token_freq'] = df['tokens'].map(lambda x: encode_text(x, word_idx_for_tv_and_laptop))
        # pos tagger
        df = df.groupby('item_name').apply(pos_tagger)

    ##################################
    # feature engineering
    ##################################

    df = parallelize_dataframe(df, speed_up_func_for_feature_engineering)
    
    #-------------------------
    # remove no need columns
    #-------------------------
    df.drop(['is_brand', 'is_valid','clean_tokens','item_id', 'word_id'], axis = 1 , inplace = True)
    logging.info('output shape / {}: {}'.format(name, df.shape))
    #-------------------------
    # save
    #-------------------------
    output_dir = '../features/{}'.format(name)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    df.to_csv('../features/{}/tokens_context.csv'.format(name), index = False)
    logging.info('finish saving {}'.format(name))

def multi(T):
    '''
    It's for using multi preprosessing to speed up each model training process.

    parameters:
    ---------------------
    T: int. 1, 2, 3, and 4.
    '''
    tokens_context(T)

if __name__ == '__main__':
    ##################################################
    # Main
    ##################################################
    #--------------------
    # setting
    #--------------------
    LEMMATIZING = False
    # log path
    log_dir = 'log/'
    #--------------------
    # core
    #--------------------
    s = time.time()
    mp_pool = Pool(1)
    mp_pool.map(multi, [1])
    e = time.time()
    print (e-s, ' secs')







