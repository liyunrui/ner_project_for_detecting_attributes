#! /usr/bin/env python3
"""
Created on Oct 2 2018

Updated on Oct 24 2018

Prepare data for the following tensorflow model.

Noticed:
    It may take around 13.935157557328543 mins, the real number depends on ur machine.

@author: Ray


Reference:
    - LSTM character embedding : https://github.com/cristianoBY/Sequence-Tagging-NER-TensorFlow:
    - character embedding: https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html
"""
import os
import time
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
sys.path.append('/home/ld-sgdev/yunrui_li/ner_project/brand_recognition_bio_FE/preprocessing')
sys.path.append('/home/ld-sgdev/yunrui_li/ner_project/brand_recognition_bio_FE/py_model')
from clean_helpers import clean_name_for_word_embedding
from utils import init_logging
from data_utils import get_glove_vocab
from data_utils import write_vocab
from data_utils import load_vocab_and_return_word_to_id_dict
from data_utils import export_glove_vectors
from data_utils import get_char_vocab
import logging
import gc

pd.options.display.max_columns = 100
pd.options.display.max_rows = 5000
pd.options.display.max_colwidth = 1000


def pad_1d(array, max_len, word_padding = True):
    if word_padding == True:
        array = array[:max_len]
        length = len(array)
        padded = array + [9858]*(max_len - len(array)) # padded index of unknown.
    else:
        array = array[:max_len]
        length = len(array)
        padded = array + [0]*(max_len - len(array)) # padded with zero.
    return padded, length

def encode_word_to_idx(word, word_to_id, vocabulary_set, lowercase = True, allow_unknown = True):
    '''encode a word (string) into id'''

    # 1. preprocess word
    if lowercase:
        word = word.lower()
    if word.isdigit():
        word = NUM

    # 2. get id of word
    if word in vocabulary_set:
        return word_to_id[word]
    else:
        if allow_unknown:
            return word_to_id[UNK]
        else:
            raise Exception("Unknow key is not allowed. Check that your vocab (tags?) is correct")

#--------------------
# setting
#--------------------
TRACE_CODE = False # for tracing funtionality and developing quickly
TRUNCATED = False # for reducing memory 
USE_CHARS = True # for character embedding
LOWERCASE = True
ALLOW_UNKNOWN = True
dim_word = 300
UNK = "$UNK$" # for the word in our own courpus which is unknown in embedding
NUM = "$NUM$" # for the word which is number

base_dir = "../models/data/wordvec"
filename_words_voc = "../models/data/wordvec/words_vocab.txt"
filename_chars_voc = "../models/data/wordvec/chars_vocab.txt"

pre_trained_word_embedding_path = "/data/ID_largewv_300_2.txt"
filename_words_vec = "../models/data/wordvec/word2vec.npz".format(dim_word)
log_dir = 'log/' # log path
init_logging(log_dir)


#-------------------
# loading data
#-------------------
if TRACE_CODE == True:
    df = pd.read_csv('../data/processed/mobile_training_v2.csv', nrows = 19)
else:
    df = pd.read_csv('../data/processed/mobile_training_v2.csv')

s = time.time()

# preprocessing
df['clean_tokens'] = df.tokens.apply(lambda x: clean_name_for_word_embedding(x) if type(x)==str else x)
if LOWERCASE:
	df['clean_tokens'] = df.clean_tokens.apply(lambda x: x.lower() if type(x)==str else x)
df['clean_tokens'] = df.clean_tokens.astype(str)

# item_id
item_dict = {}
for i, i_n in enumerate(df.item_name.unique().tolist()):
    item_dict[i_n] = i+1
df['item_id'] = [item_dict[i_n] for i_n in df.item_name.tolist()]

e = time.time()
logging.info('it spend {} mins on preprocessing'.format( (e-s) / 60.0))

#-------------------
# word embedding 
#-------------------
s = time.time()
# Build Word vocab (vocabulary set) for our customized task
vocab_glove = get_glove_vocab(pre_trained_word_embedding_path) # word set from pre-trained word_embedding
vocab_words = set(df.clean_tokens.tolist()) # word set from our own whole corpurs including train, dev, and test
vocab_set = vocab_words & vocab_glove # 這裡面的字, 肯定都有相對應的vector, 不是zero vector(The reason we did是可以節省我們使用的embbedding大小, 不用沒用到的pre-trained字也佔memory)
vocab_set.add(UNK)
vocab_set.add(NUM)

# Save vocab
write_vocab(vocab_set, base_dir, filename_words_voc)
# create dictionary mapping word to index
word_to_id_dict = load_vocab_and_return_word_to_id_dict(filename_words_voc)
# save word embedding matrix 
export_glove_vectors(word_to_id_dict, glove_filename = pre_trained_word_embedding_path,
                     output_filename = filename_words_vec, dim = dim_word)
# encode a word (string) into id
df['word_id'] = df.clean_tokens.apply( lambda x: encode_word_to_idx(x, word_to_id_dict, vocab_set, LOWERCASE, ALLOW_UNKNOWN))

e = time.time()
logging.info('it spend {} mins on word embedding '.format( (e-s) / 60.0)) # it spend 16.531146574020386 mins on word embedding over 37788 words in volcabulary.



#-------------------
# chracter embedding 
#-------------------
if USE_CHARS == True:
    dim_char = 100
    # Build Char vocab (vocabulary set) 
    vocab_chars = get_char_vocab(df.tokens)
    # Save Char vocab
    write_vocab(vocab_chars, base_dir, filename_chars_voc)
    # create dictionary mapping char to index
    chars_to_id_dict = load_vocab_and_return_word_to_id_dict(filename_chars_voc)
    # get max_word_length for padding later
    word_len_distribution = [len(w) for w in df.tokens]
    max_word_length = max(word_len_distribution)



seq_len_distribution = df.groupby('item_name').tokens.apply( lambda x : len(x.tolist())).to_frame('seq_len').reset_index()

if TRUNCATED == False:
    if TRACE_CODE == True:
        max_seq_length = 122
    else:
        max_seq_length = seq_len_distribution.seq_len.max()
else:
    max_seq_length = 100

logging.info('max_seq_length : {}'.format( max_seq_length)) # max length of sentence
logging.info('max_word_length : {}'.format( max_word_length)) # max length of word

#-------------------
# output
#-------------------

for i in range(3):
    # 
    if i == 0:
        name = 'train'
        output = df[df.eval_set == 'train']
    elif i == 1:
        name = 'val'
        output = df[df.eval_set == 'val']
    else:
        name = 'test'
        output = df.copy()
    # setting
    num_sentences = output['item_name'].nunique()
    logging.info('number of sequences : {}'.format(num_sentences))
    # 1-D
    eval_set = np.zeros(shape=[num_sentences], dtype='S5')
    item_id = np.zeros(shape=[num_sentences], dtype=np.int32) # for recording
    history_length = np.zeros(shape=[num_sentences], dtype=np.int32) # length of sentence
    # 2-D
    word_id = np.zeros(shape=[num_sentences, max_seq_length], dtype=np.int32)
    label = np.zeros(shape=[num_sentences, max_seq_length], dtype=np.int32)
    word_length = np.zeros(shape=[num_sentences, max_seq_length], dtype=np.int32) # length of words 
    # 3-D
    char_id = np.zeros(shape=[num_sentences, max_seq_length, max_word_length], dtype=np.int32)
    i = 0
    for ix, df_ in tqdm(output.groupby('item_name')):
        #logging.info('item_id : {}'.format(i))
        # 1-D
        eval_set[i] = df_['eval_set'].iloc[0]
        item_id[i] = df_['item_id'].iloc[0]
        # 2-D
        word_id[i, :], history_length[i] = pad_1d(list(map(int, df_['word_id'])), max_len = max_seq_length, word_padding = True)
        label[i, :], _ = pad_1d(list(map(int, df_['label'])), max_len = max_seq_length, word_padding = False)
        word_length[i, :], _ = pad_1d([len([char for char in w]) for w in df_['tokens'].tolist()], max_len = max_seq_length, word_padding = False)
        if USE_CHARS == True:
            # 3-D
            for i_word_axis, w in enumerate(df_['tokens'].tolist()):
                char_id[i, i_word_axis, : ], _ = pad_1d([chars_to_id_dict[char] for char in w], max_len = max_word_length, word_padding = False)

        i += 1

    #--------------------------
    # save
    #--------------------------
    base_path = 'data/{}'.format(name)
    if not os.path.isdir(base_path):
        os.makedirs(base_path)    

    if TRACE_CODE == True:
        np.save(os.path.join(base_path, 'eval_set_0.npy'), eval_set)
        np.save(os.path.join(base_path, 'word_id_0.npy'), word_id)
        np.save(os.path.join(base_path, 'history_length_0.npy'), history_length)
        np.save(os.path.join(base_path, 'label_0.npy'), label)
        np.save(os.path.join(base_path, 'item_id_0.npy'), item_id)
        if USE_CHARS == True:
            np.save(os.path.join(base_path, 'char_id_0.npy'), char_id)
            np.save(os.path.join(base_path, 'word_length_0.npy'), word_length)
    else:
        np.save(os.path.join(base_path, 'eval_set.npy'), eval_set)
        np.save(os.path.join(base_path, 'word_id.npy'), word_id)
        np.save(os.path.join(base_path, 'history_length.npy'), history_length)
        np.save(os.path.join(base_path, 'label.npy'), label)
        np.save(os.path.join(base_path, 'item_id.npy'), item_id)
        if USE_CHARS == True:
            np.save(os.path.join(base_path, 'char_id.npy'), char_id)
            np.save(os.path.join(base_path, 'word_length.npy'), word_length)


logging.info('shape of df : {}'.format(df.shape))
df.to_csv('../data/processed/mobile_training_w_word_id.csv', index = False)








