#! /usr/bin/env python3
"""
Created on Nov 5 2018

See README.md for details

@author: Ray

"""
import pandas as pd
import numpy as np
import os
from glob import glob
from sklearn.model_selection import train_test_split
import ast # real json-like string



#-------------
# setting
#-------------
out_col = ['itemid']
val_size = 0.1
seed = 19921030 # fixed
#-------------
# core
#-------------
for category in os.listdir('/data/ner_task/dress/shopee_data_tagging_result/')[-1:]:
    #print ('category', category)
    path = glob('/data/ner_task/dress/shopee_data_tagging_result/{}/*.csv'.format(category))[0]
    df = pd.read_csv(path)
    ground_truth = df.columns[-1]
    attribute_types = list(df[ground_truth].apply( lambda x: ast.literal_eval(x)).iloc[0].keys())
    X = df[out_col].values
    for attr in attribute_types[:]:
        #print ('attr', attr.strip())
        y = df[ground_truth]. \
        apply( lambda x: ast.literal_eval(x)). \
        apply( lambda x: x[attr][0][0] if x[attr] != 'no value' else x[attr]).values
        # stratified split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = val_size, random_state = seed)
        X_train = pd.DataFrame(X_train, columns = out_col)
        X_test = pd.DataFrame(X_test, columns = out_col)
        #--------------
        # save
        #--------------
        save_dir = '../data/train_val_split/{}/{}'.format(category,attr.strip())
        print ('save_dir', save_dir)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        X_train.to_csv(os.path.join(save_dir, 'train.csv'), index = False)
        X_test.to_csv(os.path.join(save_dir, 'val.csv'), index = False)

