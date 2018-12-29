#! /usr/bin/env python3
"""
Created on Oct 2 2018

Prepare training data for the following model.

@author: Ray

"""

import sys
import ast # real json-like string
import pandas as pd
import gc
import sys
import os
sys.path.append('../../brand_recognition_bio_FE/preprocessing')
sys.path.append('../../brand_recognition_bio_FE/py_model')
from preprocessing import sequence_labeling_w_bio_encoding
import logging


AMAZON = False
output_column = ['item_name', 'brand']
#--------------------------
# loading data
#--------------------------
lazada_mobile = pd.read_csv('/data/ner_task/mobile/mobile_ID_attribute_brand.csv')
shoope_mobile = pd.read_csv('/data/ner_task/mobile/mobile_ID_attribute_tagging.csv')

#--------------------------
# preprocessing data
#--------------------------
shoope_mobile.drop_duplicates(subset = ['itemid'], inplace = True)
lazada_mobile.brand = lazada_mobile.brand.apply(lambda x : x.lower())
lazada_mobile.rename(columns = {'title':'item_name'}, inplace = True)

brand_list_lazada = set(lazada_mobile.brand.unique())
# brand_list_shopee = set(shoope_mobile.brand.unique())
# print ('difference betwee target and source dataset', brand_list_lazada.difference(brand_list_shopee))

if AMAZON == True:
    #--------------------------
    # loading data
    #--------------------------
    amazon_dataset_path = '/home/ld-sgdev/yunrui_li/grouping/tv_and_laptop_grouping/raw_data/amazon/products.csv'
    amazon_dataset = pd.read_csv( amazon_dataset_path, header = None)
    amazon_dataset.rename(columns = {0: 'item_name',
                                     1: 'what_brand_name',
                                     2: 'description',
                                     3: 'category'
                                    }, inplace = True)
    #--------------------------
    # processing
    #--------------------------
    amazon_dataset.item_name = amazon_dataset.item_name.astype(str)
    amazon_dataset.category = amazon_dataset.category.astype(str)
    # covert what_brand_name to lower for creating supervised data.
    amazon_dataset.what_brand_name = amazon_dataset.what_brand_name.apply(lambda x: x.lower() if type(x) == str else x)
    # filling na with Others
    amazon_dataset.what_brand_name = amazon_dataset.what_brand_name.fillna('Others')
    # catalogue selection
    amazon_mobile = amazon_dataset[amazon_dataset.what_brand_name.isin(list(brand_list_lazada))]
    amazon_mobile = amazon_mobile[amazon_mobile.category.str.contains('Cell Phones')]
    amazon_mobile.rename(columns = {'what_brand_name':'brand'}, inplace = True)
    amazon_mobile.reset_index( inplace = True)

#--------------------------
# data configuration: take Lazada data as testing and shoppe as training data.
#-------------------------- 

col = ['item_name','brand']
if AMAZON == True:
    df = pd.concat([lazada_mobile[col], amazon_mobile[col]], axis = 0)
    
#
df.rename(columns = {'brand':'what_brand_name'}, inplace = True)
# train/val configue: # 0:training # 1:validating
df = df.sample(len(df))
df['is_valid'] = ['train' if i < 0.8 * len(df) else 'val' for i in range(len(df))]
# BIO tagging
df = df.groupby('item_name').apply(lambda x: sequence_labeling_w_bio_encoding(x, NORMALIZED = False)).reset_index(drop = True)

gc.collect()
#--------------------------
# save
#--------------------------
base_path = '../data/processed'

if not os.path.isdir(base_path):
	os.makedirs(base_path)

df.to_csv(os.path.join(base_path,'mobile_training.csv') , index = False)

