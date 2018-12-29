#! /usr/bin/env python3

import pandas as pd
import numpy as np
import os
import gc
from datetime import datetime # for the newest version control
from collections import Counter # for statistical features
import enchant # English spellchecking system
import multiprocessing as mp # for speeding up some process
import time
from billiard import Pool


def hasNumbers(row):
    '''
    Returning if the tokens string includes number (binary feature)
    otherwise, it's letters
    '''
    inputString = row.tokens
    return 1 if any(char.isdigit() for char in inputString) == True else 0

def consist_only_of_digits(row):
	'''
	Return true if all characters in the string are digits and there is at least one character, false otherwise.
	str.isdigit():
	'''
	return 1 if row.tokens.isdigit() else 0

def do_consist_hyphen(row):
	'''
	Return true if all characters in the string are digits and there is at least one character, false otherwise.
	str.isdigit():
	'''
	return 1 if '-' in row.tokens else 0

def if_it_is_by(row):
    '''
    Return true if all characters in the string are digits and there is at least one character, false otherwise.
    str.isdigit():
    '''
    by_set = {'by'}
    return 1 if row.tokens.lower() in by_set else 0

def if_it_is_and(row):
    '''
    Return true if all characters in the string are digits and there is at least one character, false otherwise.
    str.isdigit():
    '''
    and_set = {'&', 'and'}
    return 1 if row.tokens.lower() in and_set else 0

def if_start_with_capital_chars(row):
    '''
    Returning if the tokens start with captital characters (binary feature)
    '''
    fisrt_char = row.tokens[0]
    return 1 if fisrt_char.isupper() == True else 0

def percentage_of_upper_chars_in_token(row):
    '''
    Calculate percentage of N captal characters in the tokens
    '''
    return sum(1 for char in row.tokens if char.isupper()) / len(row.tokens)

def check_if_english_word(row):
    '''
    judge if this token is english word or not
    '''
    return 1 if dict_for_english.check(row.tokens) else 0

def len_of_token(row):
	'''
	return number of characters that one token has.
	'''
	return len(row.tokens)

def is_all_character_captilized(row):
	'''
	check if all characters in the tokens is captilized.
	'''
	return 1 if row.tokens.isupper() else 0

def is_all_character_lowercase(row):
	'''
	check if all characters in the tokens is lowercase.
	'''
	return 1 if row.tokens.islower() else 0

def is_first_character_digit(row):
	'''
	check if the first character in the tokens is digit.
	'''
	return 1 if row.tokens[0].isdigit() else 0

def is_first_character_uppercase(row):
	'''
	check if the first character in the tokens is uppercase.
	'''
	return 1 if row.tokens[0].isupper() else 0

def is_second_character_uppercase(row):
    '''
    check if the first character in the tokens is uppercase.
    '''
    try:
        return 1 if row.tokens[1].isupper() else 0
    except Exception:
        # for token with only one character.
        return -1

def if_it_is_a_sale_word(row):
    '''
    check if the token is word that was used for sale.
    '''
    sale_words_set = {'free','sale','promo', 'best', 
    'seller','ready', 'share', 'stock', 'hot', 'big','gratis'} # w.lower()
    return 1 if row.tokens.lower() in sale_words_set else 0

def speed_up_func_for_feature_engineering(df):
    '''
    Put the columns u need to apply()

    data: DataFrame
    '''

    # hasNumbers
    df['if_tokens_has_numbers_in_the_str'] = df.apply(hasNumbers, axis = 1)
    # if_start_with_capital_chars
    df['if_start_with_capital_chars'] = df.apply(if_start_with_capital_chars, axis = 1)
    # percentage_of_upper_chars_in_token
    df['percentage_of_upper_chars_in_token'] = df.apply(percentage_of_upper_chars_in_token, axis = 1)
    # check_if_english_word
    df['check_if_english_word'] = df.apply(check_if_english_word, axis = 1)
    # len_of_token
    df['len_of_token'] = df.apply(len_of_token, axis = 1)
    # is_all_character_captilized
    df['is_all_character_captilized'] = df.apply(is_all_character_captilized, axis = 1)
    # is_all_character_lowercase
    df['is_all_character_lowercase'] = df.apply(is_all_character_lowercase, axis = 1)
    # consist_only_of_digits
    df['consist_only_of_digits'] = df.apply(consist_only_of_digits, axis = 1)
    # is_first_character_digit
    df['is_first_character_digit'] = df.apply(is_first_character_digit, axis = 1)
    # is_first_character_uppercase
    df['is_first_character_uppercase'] = df.apply(is_first_character_uppercase, axis = 1)
    # do_consist_hyphen
    df['do_consist_hyphen'] = df.apply(do_consist_hyphen, axis = 1)
    # if_it_is_and
    df['if_it_is_and'] = df.apply(if_it_is_and, axis = 1)
    # is_second_character_uppercase
    df['is_second_character_uppercase'] =  df.apply(is_second_character_uppercase, axis = 1)
    # if_it_is_a_sale_word
    df['if_it_is_a_sale_word'] =  df.apply(if_it_is_a_sale_word, axis = 1)
    # if_it_is_by
    df['if_it_is_by'] = df.apply(if_it_is_by, axis = 1)
    

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

def tokens_chrarateristic(T):
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
    df.to_csv('../features/{}/tokens_characteristics.csv'.format(name), index = False)
    logging.info('finish saving {}'.format(name))

def multi(T):
    '''
    It's for using multi preprosessing to speed up each model training process.

    parameters:
    ---------------------
    T: int. 1, 2, 3, and 4.
    '''
    tokens_chrarateristic(T)

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
    # for check if it's English word
    dict_for_english = enchant.Dict("en_US")

    #--------------------
    # core
    #--------------------
    s = time.time()
    mp_pool = Pool(1)
    mp_pool.map(multi, [1])
    e = time.time()
    print (e-s, ' secs')




