#! /usr/bin/env python3
"""
Created on Aug 1 2018

In order to capture information about tokens neiboring a given token.
@author: Ray

"""
import enchant # English spellchecking system
from collections import Counter # for statistical features
import pandas as pd
import numpy as np
from datetime import datetime # for the newest version control
import os
import time
import multiprocessing as mp # for speeding up some process
from nltk import tag # for pos_tagging
from nltk.corpus import wordnet # for geting pos of wordnet
from nltk.stem import WordNetLemmatizer
from billiard import Pool
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

def hasNumbers(x):
    '''
    Returning if the tokens string includes number (binary feature)
    otherwise, it's letters
    '''
    try:
        return 1 if any(char.isdigit() for char in x) == True else 0
    except:
        return x
    
def consist_only_of_digits(x):
    '''
    Return true if all characters in the string are digits and there is at least one character, false otherwise.
    str.isdigit():
    '''
    try:
        return 1 if x.isdigit() else 0
    except:
        return x
    
def do_consist_hyphen(x):
    '''
    Return true if all characters in the string are digits and there is at least one character, false otherwise.
    str.isdigit():
    '''
    try:
        return 1 if '-' in x else 0
    except:
        return x

def if_it_is_and(x):
    '''
    Return true if all characters in the string are digits and there is at least one character, false otherwise.
    str.isdigit():
    '''
    and_set = {'&', 'and'}
    try:
        return 1 if x.lower() in and_set else 0
    except:
        return x

def if_it_is_by(x):
    '''
    Return true if all characters in the string are digits and there is at least one character, false otherwise.
    str.isdigit():
    '''
    by_set = {'by'}
    try:
        return 1 if x.lower() in by_set else 0
    except:
        return x

def if_it_is_a_sale_word(x):
    '''
    check if the token is word that was used for sale.
    '''
    sale_words_set = {'free','sale','promo', 'best', 
    'seller','ready', 'share', 'stock', 'hot', 'big','gratis'} # w.lower()
    try:
        return 1 if x.lower() in sale_words_set else 0
    except:
        return x

def if_start_with_capital_chars(x):
    '''
    Returning if the tokens start with captital characters (binary feature)
    '''
    fisrt_char = x
    try:
        return 1 if fisrt_char.isupper() == True else 0
    except:
        return x

def percentage_of_upper_chars_in_token(x):
    '''
    Calculate percentage of N captal characters in the tokens
    '''
    try:
        return sum(1 for char in x if char.isupper()) / len(x)
    except:
        return x

def check_if_english_word(x):
    '''
    judge if this token is english word or not
    '''
    try:
        return 1 if dict_for_english.check(x) else 0
    except:
        return x

def len_of_token(x):
    '''
    return number of characters that one token has.
    '''
    try:
        return len(x)
    except:
        return x

def is_all_character_captilized(x):
    '''
    check if all characters in the tokens is captilized.
    '''
    try:
        return 1 if x.isupper() else 0
    except:
        return x

def is_all_character_lowercase(x):
    '''
    check if all characters in the tokens is lowercase.
    '''
    try:
        return 1 if x.islower() else 0
    except:
        return x

def is_first_character_digit(x):
    '''
    check if the first character in the tokens is digit.
    '''
    try:
        return 1 if x[0].isdigit() else 0
    except:
        return x
    
def is_first_character_uppercase(x):
    '''
    check if the first character in the tokens is uppercase.
    '''
    try:
        return 1 if x[0].isupper() else 0
    except:
        return x
    
def is_second_character_uppercase(x):
    '''
    check if the first character in the tokens is uppercase.
    '''
    try:
        return 1 if x[1].isupper() else 0
    except:
        # for token with only one character.
        return -1
        
def position_of_the_tokens(row, str_for_field):
    '''
    Returning positon of the token in the item_name
    '''
    try:
        return row.item_name.split().index(row[str_for_field])
    except Exception:
        return row[str_for_field]

def position_of_the_tokens_from_the_bottom(row, str_for_field):
    '''
    Returning positon of the token in the item_name
    '''
    try:
        item_name_ls = row.item_name.split()
        item_name_ls.reverse()
        return item_name_ls.index(row[str_for_field])
    except Exception:
        return row[str_for_field]

def position_of_the_tokens_in_ratio(row, str_for_field):
    '''
    Returning positon of the token in the item_name
    '''
    try:
        return 1.0 * row.item_name.split().index(row[str_for_field]) / len(row.item_name.split())
    except Exception:
        return row[str_for_field]

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
    try:
        if type(text) == str:
            return int(word_idx[text.lower()])
        else:
            return text
    except Exception:
        #print (text)
        pass 

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

	for i in ['the_preceding', 'the_succeeding']:
		for w in [1, 2, 3, 4, 5, 6]:
			df['{}_w_{}-hasNumbers'.format(i, w)] = df['{}_word_given_current_token_w_{}'.format(i, w)].apply(hasNumbers)
			df['{}_w_{}-consist_only_of_digits'.format(i, w)] = df['{}_word_given_current_token_w_{}'.format(i,w)].apply(consist_only_of_digits)
			df['{}_w_{}-do_consist_hyphen'.format(i, w)] = df['{}_word_given_current_token_w_{}'.format(i, w)].apply(do_consist_hyphen)
			df['{}_w_{}-if_it_is_and'.format(i, w)] = df['{}_word_given_current_token_w_{}'.format(i, w)].apply(if_it_is_and)
			df['{}_w_{}-if_start_with_capital_chars'.format(i, w)] = df['{}_word_given_current_token_w_{}'.format(i, w)].apply(if_start_with_capital_chars)
			df['{}_w_{}-percentage_of_upper_chars_in_token'.format(i, w)] = df['{}_word_given_current_token_w_{}'.format(i, w)].apply(percentage_of_upper_chars_in_token)
			df['{}_w_{}-check_if_english_word'.format(i, w)] = df['{}_word_given_current_token_w_{}'.format(i, w)].apply(check_if_english_word)
			df['{}_w_{}-len_of_token'.format(i, w)] = df['{}_word_given_current_token_w_{}'.format(i, w)].apply(len_of_token)
			df['{}_w_{}-is_all_character_captilized'.format(i, w)] = df['{}_word_given_current_token_w_{}'.format(i, w)].apply(is_all_character_captilized)
			df['{}_w_{}-is_all_character_lowercase'.format(i, w)] = df['{}_word_given_current_token_w_{}'.format(i, w)].apply(is_all_character_lowercase)
			df['{}_w_{}-is_first_character_digit'.format(i, w)] = df['{}_word_given_current_token_w_{}'.format(i, w)].apply(is_first_character_digit)
			df['{}_w_{}-is_first_character_uppercase'.format(i, w)] = df['{}_word_given_current_token_w_{}'.format(i, w)].apply(is_first_character_uppercase)
			df['{}_w_{}-is_second_character_uppercase'.format(i, w)] = df['{}_word_given_current_token_w_{}'.format(i, w)].apply(is_second_character_uppercase)
			df['{}_w_{}-position_of_the_tokens'.format(i, w)] = df.apply(lambda row: position_of_the_tokens(row, str_for_field = '{}_word_given_current_token_w_{}'.format(i, w)), axis = 1)
			df['{}_w_{}-position_of_the_tokens_from_the_bottom'.format(i, w)] = df.apply(lambda row: position_of_the_tokens_from_the_bottom(row, str_for_field = '{}_word_given_current_token_w_{}'.format(i, w)), axis = 1)
			df['{}_w_{}-position_of_the_tokens_in_ratio'.format(i, w)] = df.apply(lambda row: position_of_the_tokens_in_ratio(row, str_for_field = '{}_word_given_current_token_w_{}'.format(i, w)), axis = 1)
			df['{}_w_{}-token_freq'.format(i, w)] = df['{}_word_given_current_token_w_{}'.format(i, w)].apply(lambda x: encode_text(x, word_idx))
			df['{}_w_{}-if_it_is_by'.format(i, w)] = df['{}_word_given_current_token_w_{}'.format(i, w)].apply(if_it_is_by)
			# add at 9/3
			# if_it_is_a_sale_word
			df['{}_w_{}-if_it_is_a_sale_word'.format(i, w)] = df['{}_word_given_current_token_w_{}'.format(i, w)].apply(if_it_is_a_sale_word)

			#---------------------
			# drop the columns we no need
			#---------------------
			df.drop(['{}_word_given_current_token_w_{}'.format(i, w)], axis = 1 , inplace = True)

	return df

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
    It map the treebank tags to WordNet part of speech names for getting lemma.
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

def contextual_features(T):
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
	df.dropna(subset = ['item_name', 'tokens'], axis = 0, inplace = True)
	#--------------------------
	# conver type
	#--------------------------
	df['tokens'] = df.tokens.astype(str)

	global word_idx
	if LEMMATIZING == True:
		print ('LEMMATIZING : {}'.format(LEMMATIZING))
		lemmatizer = WordNetLemmatizer()
		df = df.groupby('item_name').apply(pos_tagger)
		# get_wordnet_pos
		df['wordnet_pos'] = df.pos_tagger.apply(get_wordnet_pos)
		# lemmatizing
		df['lemma'] = [lemmatizer.lemmatize(t.lower(), pos) for t, pos in zip(df.tokens.tolist(), df.wordnet_pos.tolist())]
		# drop columns that we don't need it.
		df.drop(['pos_tagger','wordnet_pos'], axis = 1, inplace = True)
		# make word_index
		word_idx = make_word_idx(pd.concat([df], axis = 0).lemma.tolist())
	else:
		# foe token_freq
		word_idx = make_word_idx(pd.concat([df], axis = 0).tokens.tolist())

	#--------------------------
	# preprocessing for contextual information
	#--------------------------
	df = parallelize_dataframe(df, speed_up_func_for_preprocessing)

	##################################
	# feature engineering
	##################################
	df = parallelize_dataframe(df, speed_up_func_for_feature_engineering)

	if LEMMATIZING == True:
		df.drop(['lemma'], axis = 1, inplace = True)

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
	df.to_csv('../features/{}/contextual_features.csv'.format(name), index = False)
	logging.info('finish saving {}'.format(name))

def multi(T):
    '''
    It's for using multi preprosessing to speed up each model training process.

    parameters:
    ---------------------
    T: int. 1, 2, 3, and 4.
    '''
    contextual_features(T)


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
    #----------------------------------------
    # way 1 ---> 460.75708651542664 secs
    #----------------------------------------
    mp_pool = Pool(1)
    mp_pool.map(multi, [1])
    mp_pool.close()

    #----------------------------------------
    # way 2 ---> 273.1357443332672  secs
    #----------------------------------------
    # for i in [1,2,3,4]:
    #     multi(i)

    e = time.time()
    print (e-s, ' secs')



