#! /usr/bin/env python3
"""
Created on Oct 24 2018

Text preprocessing and sequential labeling.

TO do list in the future:
    - pull data again using pyspark for getting original item_name when need feature engineering.

@author: Ray

"""

import sys
import os
import ast # real json-like string
import pandas as pd
import gc
sys.path.append('../../brand_recognition_bio_FE/preprocessing')
from clean_helpers import clean_name

pd.options.display.max_columns = 100
pd.options.display.max_rows = 5000
pd.options.display.max_colwidth = 1000


def filter_data_for_ner_task(df, out_col, FLAG, attr = ' Brand'):
    """
    Return reliable data that we generally make sure attribute you assigned exist in the title.
    """
    num_filter = 0
    keep = []
    attr_tags = []
    for ix, row in df.iterrows():
        if FLAG == 'shopee':
            title = row['title'] # str
            tagging_dict = ast.literal_eval(row['mobile']) # dict
            #--------------
            # preprocessing for tagging filed in raw shaopee data
            #--------------
            if tagging_dict[attr] != 'no value':
                attr_tag = tagging_dict[attr][0][0] # str
            else:
                attr_tag = tagging_dict[attr] # str
            #--------------
            # core
            #--------------
            if attr_tag == 'no value':
                num_filter += 1
                keep.append(0)
                attr_tags.append(0)
            elif attr_tag.lower() in title.lower():
                keep.append(1)
                attr_tags.append(attr_tag)
            else:
                num_filter += 1
                keep.append(0)
                attr_tags.append(0)
        elif FLAG == 'lazada':
            title = row['item_name'] # str
            attr_tag = row['brand'] # str
            if attr_tag.lower() in title.lower():
                keep.append(1)
                attr_tags.append(attr_tag)
            else:
                num_filter += 1
                keep.append(0)
                attr_tags.append(0)
            
        else:
            # if the assertion fails, Python uses ArgumentExpression as the argument for the AssertionError. 
            assert False,  '========= the FLAG only accecpt shopee and lazada lah fuck u =========' # condition, AssertionError
    #-----------------
    # output
    #-----------------
    df['keep'] = keep
    df['attr_tags'] = attr_tags

    if FLAG == 'shopee':
        pass
    elif FLAG == 'lazada':
        df.drop(['brand'], axis = 1, inplace = True)
    else:
        assert False, '========= the FLAG only accecpt shopee and lazada lah fuck u ========='

    df.rename(columns = {'title': out_col[0], 'attr_tags':out_col[1]}, inplace = True)
    df = df[df.keep == 1]
    df = df[out_col]
    gc.collect()
    return df, num_filter

def sequence_labeling_w_bio_encoding(row, NORMALIZED = False):
    '''
    BIO encoding is a distant supervision approach to automatically generate training data for training machine-learning based model. 
    
        # B-B: 2
        # I-B: 1
        # O: 0
    Reference for distant supervision approach: http://deepdive.stanford.edu/distant_supervision
    Reference for BIO : Attribute Extraction from Product Titles in eCommerce.
    Assumption:
        - We assume that one sku only has one brand name.(kind of non-realistic)
    parameters:
    --------------
    df: DataFrame
    if_assumption: str. if True, we assume we only have one-single brand_word in one item_name. 
    Otherwise, we can have multiple token with positive lable in one item_name.
    '''

    # initialize variables
    word_list = []
    tagging = [] # multi-class label, {0:not part of the brand name, 1: intermediate part of the brand name, 2:beginning of the brand name}
    item_name = []
    val = [] 
    #---------------
    # sequential labeling with BIO encoding
    #---------------
    brand_started = False
    b_ix = 0
    brand = row.what_brand_name.iloc[0].split(' ')
    title = clean_name(row['item_name'].iloc[0]).split(' ')
    # filter
    title = [t for t in title if '' != t]
    for w_ix, word in enumerate(title):
        if word.lower() == brand[0].lower():
            tagging.append(2) # B-B: 2
            brand_started = True
            b_ix += 1
        elif (len(brand) > 1) and (brand_started):
            if b_ix >= len(brand):
                # for avoiding . For example, if 'BUMBLE AND BUMBLE by Bumble and Bumble: QUENCHING CONDITIONER 8.5 OZ'
                tagging.append(0) # O: 0
                brand_started = False  
                b_ix = 0                
            else:
                if word.lower() == brand[b_ix].lower():
                    tagging.append(1) # I-B: 1
                    b_ix += 1
                    if b_ix == len(brand):
                        # go back to orginal state because we already marked what we want
                        brand_started = False
                        b_ix = 0
                else:
                    tagging.append(0) # O: 0
                    brand_started = False     
                    # if we need to modified the labeling we priviously marked.
                    if b_ix < len(brand):
                        go_back_to_modified = 0
                        for i in range(b_ix):
                            #print ('w_ix', w_ix) # w_ix 對應的不是整個 tagging的list: 兩個解法, 1.groupby 2.w_ix要一直被加上
                            go_back_to_modified += 1
                            #print ('go back', w_ix - go_back_to_modified)
                            tagging[w_ix - go_back_to_modified] = 0 # O: 0
                        # Once removing privous labeling, we update b_ix to zero
                        b_ix = 0         
        else:
            brand_started = False
            tagging.append(0) # O: 0
        #---------------------------
        # for output dataframe
        #---------------------------
        if NORMALIZED == True:
            word_list.append(word.lower())
        else:
            word_list.append(word)
        item_name.append(clean_name(row['item_name'].iloc[0]))
        val.append(row['eval_set'].iloc[0])
    #---------------------------
    # output
    #---------------------------
    df = pd.DataFrame({'tokens':word_list, 
                'label': tagging,
                'eval_set': val,
                'item_name':item_name})[['item_name','tokens','label','eval_set']]
    return df

if __name__ == '__main__':
    #--------------------------
    # setting
    #--------------------------
    output_column = ['item_name', 'brand', 'eval_set']
    val_size = 0.1
    #--------------------------
    # loading data
    #--------------------------
    lazada_mobile = pd.read_csv('/data/ner_task/mobile/mobile_ID_attribute_brand.csv')
    shoope_mobile = pd.read_csv('/data/ner_task/mobile/mobile_ID_attribute_tagging.csv')

    #--------------------------
    # preprocessing data
    #--------------------------
    # droup duplicated: make our result trustworthy, don't count the same itemname with same brand 
    shoope_mobile.drop_duplicates(subset = ['itemid'], inplace = True)
    lazada_mobile.drop_duplicates(subset = ['brand','title'], inplace = True)

    lazada_mobile.brand = lazada_mobile.brand.apply(lambda x : x.lower())
    lazada_mobile.rename(columns = {'title':'item_name'}, inplace = True)

    print ('num_sku from raw lazada: {}'.format(len(lazada_mobile)))
    print ('num_sku from raw shopee: {}'.format(len(shoope_mobile)))   

    #--------------------------
    # data configuration: take Lazada data as testing and shoppe as training data.
    #-------------------------- 

    lazada_mobile['eval_set'] = ['test' for i in range(len(lazada_mobile))]
    shoope_mobile['eval_set'] = ['train' for i in range(len(shoope_mobile))]

    # using hold-out method as validating strategy: switch to shuffle with item_id
    val_item_name = set(pd.Series(shoope_mobile.title.unique()).sample(frac = val_size).unique())
    for ix, row in shoope_mobile.iterrows():
        if row['title'] in val_item_name:
            shoope_mobile['eval_set'].iloc[ix] = 'val' 
            
    #--------------------------
    # filter: In order to get high-quality data, we remove the sku that his attribute name do not exist in title from our shopee data.
    #--------------------------
    attr = ' Brand'
    lazada_mobile, num_filter_l = filter_data_for_ner_task(lazada_mobile, FLAG = 'lazada', out_col = output_column, attr =  attr)
    shoope_mobile, num_filter_s = filter_data_for_ner_task(shoope_mobile, FLAG = 'shopee', out_col = output_column, attr =  attr)

    print ('num_sku_removed from lazada: {}'.format(num_filter_l))
    print ('num_sku_removed from shopee: {}'.format(num_filter_s))
    print ('# testing sku : {}'.format(len(lazada_mobile)))
    print ('# training sku : {}'.format(len(shoope_mobile[shoope_mobile.eval_set == 'train'])))
    print ('# validating sku : {}'.format(len(shoope_mobile[shoope_mobile.eval_set == 'val'])))
    df = pd.concat([lazada_mobile[output_column], shoope_mobile[output_column]], axis = 0)
    del lazada_mobile, shoope_mobile
    gc.collect()

    #--------------------------
    # BIO tagging
    #--------------------------
    df.rename(columns = {'brand':'what_brand_name'}, inplace = True)

    df = df.groupby('item_name').apply(lambda x: sequence_labeling_w_bio_encoding(x, NORMALIZED = False)).reset_index(drop = True)

    #-------------------------
    # post-processing: remove some sku we cannot tag trough our sequential labeling.
    #-------------------------
    '''
    It's very little and the reason why this happening after text processomg is, for example, let's say a title of sku and brand is buy 1 get 1 free kasus and asus
    Then, he won't be filtered by text processing.
    '''
    no_label_item_name = df.groupby('item_name').label.mean().to_frame().reset_index()
    no_label_item_name = no_label_item_name[no_label_item_name.label == 0].item_name.tolist()
    df = df[~df.item_name.isin(no_label_item_name)]

    gc.collect()
    #--------------------------
    # save
    #--------------------------
    base_path = '../data/processed'

    if not os.path.isdir(base_path):
        os.makedirs(base_path)

    df.to_csv(os.path.join(base_path,'mobile_training_v2.csv') , index = False)