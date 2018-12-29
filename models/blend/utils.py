#! /usr/bin/env python3
"""
Created on Oct 19 2018

Updated on Oct 22 2018

It provide utility function for doing Out-Of-Fold validating strategies,  
inference of f1 score on sentence-level using Lightgbm, and  
Bayesian optimization for lightgbm and xgboost model.

@author: Ray

"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from collections import Counter, OrderedDict
from bayes_opt import BayesianOptimization
import lightgbm as lgb
import time
from datetime import datetime
import logging
import os

class HyperParameterTuning(object):
    """Interface containing some boilerplate code for hyper-parameter tuning of Tree-based models, sucha lightgbm or xgboost.

    Args:
        
    """
    def __init__(self, train, features, target, log_dir, group_fold = True, n_splits = 5,group_by = 'item_id',
        params_bound = {
                        'num_leaves': (25, 50),
                        'lambda_l2': (0.0, 0.05),
                        'lambda_l1': (0.0, 0.05),
                        'min_child_samples': (20, 120),
                        'bagging_fraction': (0.5, 1.0),
                        'feature_fraction': (0.5, 1.0),
                        }
                        ):
        self.train = train
        self.features = features
        self.target = target
        self.group_fold = group_fold
        self.group_by = group_by
        self.n_splits = n_splits
        self.params_bound = params_bound
        self.log_dir = log_dir
        init_logging(self.log_dir)

    def cv(self, fit_params):
        """Perform the cross-validation with given paramaters.

        Args:
            -
        Return:
            mean cv result(float).
        """
        #----------------------
        # Step1: K-fold split
        #----------------------
        unique_vis = np.array(sorted(self.train[self.group_by].astype(str).unique()))
        folds = GroupKFold(self.n_splits)
        ids = np.arange(self.train.shape[0]) # index of row for whole data

        fold_ids = [] # saving training and validating index of row for each fold
        for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
            # trn_vis: 1-D array with index of training row
            # val_vis: 1-D array with index of validating row
            fold_ids.append([
                    ids[self.train[self.group_by].astype(str).isin(unique_vis[trn_vis])],
                    ids[self.train[self.group_by].astype(str).isin(unique_vis[val_vis])]
                ])
        #----------------------
        # Step2: cross-validation
        #----------------------
        cv_result = []
        for fold_id, (trn_idx, val_idx) in enumerate(fold_ids):
            #---------------------
            # train-val split
            #---------------------
            devel = self.train[self.features].iloc[trn_idx]
            y_devel = self.train[self.target].iloc[trn_idx]
            valid = self.train[self.features].iloc[val_idx]
            y_valid = self.train[self.target].iloc[val_idx]

            # for custom_f1: 1-D array with shape of (num_samples,), which each element is item_id
            item_id_for_f1 = self.train[self.group_by].iloc[val_idx].values.reshape(len(val_idx))                 
            logging.info ("Fold : {}".format(fold_id))            
            #--------------------
            # covert pd.DataFrame into lgb.Dataset
            #--------------------
            dtrain = lgb.Dataset(devel, label= y_devel, free_raw_data = False)
            dvalid = lgb.Dataset(valid, label= y_valid, free_raw_data = False, reference= dtrain)

            evals_result = {} # for saving the evaluation metric of validating set during training
            model = lgb.train(params = fit_params, 
                              train_set = dtrain, 
                              num_boost_round = 1000, # when doing parameter tuning, do not use too large num_boost_round for speeding up the whole process.
                              valid_sets = dvalid, 
                              evals_result = evals_result,
                              verbose_eval = 100,
                              feval = customized_eval(data = item_id_for_f1, threshold = 0.5, verbose = False), 
                                 )
            res_score_ls = evals_result['valid_0']['f1-score-on-sentence-level']
            
            cv_result.append(max(res_score_ls))
        return np.mean(cv_result)

    def lgb_eval(self, num_leaves, lambda_l2,
                 lambda_l1, min_child_samples,bagging_fraction,
                 feature_fraction):
        '''
        Notice that:
            Bayesian optimization is designed to find optimal value through maximization.
            So, if yur target function is loss function. For example, rmse. the lower, the better. 
            Don't forget to put negative into the target function.
            However, if your target function is evluation metric. For example, f1-score. the higher, the better.
            There is no need to put negative when returning.
        '''
        params = {
        "objective" : "binary",
        "metric" : "None", 
        "num_leaves" : int(num_leaves),
        "lambda_l2" : lambda_l2,
        "lambda_l1" : lambda_l1,
        "min_child_samples" : int(min_child_samples),
        "bagging_fraction" : bagging_fraction,
        "feature_fraction" : feature_fraction,
        "subsample_freq" : 1, # In practice, we fixed this number.
        "bagging_seed" : int(time.time()),
        "max_depth" : -1,
        "learning_rate" : 0.1, # fix not that smalle learning rate for speeding up whole process of parameter tuning(by default, 0.03)
        "num_threads": 16,
        "early_stopping_rounds": 25, # fix not that large early_stopping_rounds for speeding up whole process of parameter tuning
        }

        if self.group_fold == True:
            #print ('params',params)
            cv_result = self.cv(fit_params = params) # return the f1-score on sentence-level. The higher, The better.
            
        else:
            cv_result = self.cv(fit_params = params)
        logging.info('best_parameters : {}'.format(params))
        logging.info('best_score : {}'.format(cv_result))
        return cv_result

    def param_tuning(self, init_points, num_iter, **args):
        '''
        Args:
            -init_points: Number of random points to probe when kicking start.
            -n_iter: Total number of times the process is to repeated.

        Notice that:
            init_points + n_iter is equal to number of model u will train during the process of hyper-parameter tuning.
        '''
        lgbBO = BayesianOptimization(self.lgb_eval, self.params_bound)

        lgbBO.maximize(init_points=init_points, n_iter= num_iter, **args)
        return lgbBO

class KFoldValidation(object):
    """Interface containing some boilerplate code for k-fold cross validation

    Note:
        It's built for feature selection rather than parameter tuning.
    """
    def __init__(self, data, group_by = 'item_id', n_splits=5, feature_importance_dir = 'OOF_FI.csv'):
        ''''''
        unique_vis = np.array(sorted(data[group_by].astype(str).unique()))
        folds = GroupKFold(n_splits)
        ids = np.arange(data.shape[0]) # index of row for whole data
        
        self.fold_ids = [] # saving training and validating index of row for each fold
        for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
            # trn_vis: 1-D array with index of training row
            # val_vis: 1-D array with index of validating row
            self.fold_ids.append([
                    ids[data[group_by].astype(str).isin(unique_vis[trn_vis])],
                    ids[data[group_by].astype(str).isin(unique_vis[val_vis])]
                ])

    def validate(self, train, test, features, target_col, use_which_model ='lgb', 
                 name="", prepare_stacking=False, 
                 fit_params={"early_stopping_rounds": 50, "verbose": 100, "eval_metric": "rmse"}):
        '''
        Args:
            -test: DataFrame. Only needed, if prepare_stacking is True.
            -features: list.
            -target_col: str.
            -use_which_model: Boolean.
            -name: DataFrame. Only needed, if prepare_stacking is True.
            -prepare_stacking: Boolean.
            -fit_params: Dict.
        Return:
            final measure of performance of K-fold. (float)
        '''
        col_need_for_computing_f1 = ['item_id']
        
        #----------------------
        # initialization
        #----------------------
        self.FeatImp = pd.DataFrame(index=features) # Feature Importance 
        full_score = 0 # Final Meaure of Performance
        
        if prepare_stacking:
            test[name] = 0
            train[name] = np.NaN
        
        for fold_id, (trn_idx, val_idx) in enumerate(self.fold_ids):
            #---------------------
            # train-val split
            #---------------------
            devel = train[features].iloc[trn_idx]
            y_devel = train[target_col].iloc[trn_idx]
            valid = train[features].iloc[val_idx]
            y_valid = train[target_col].iloc[val_idx]
            
            # for custom_f1: 1-D array with shape of (num_samples,), which each element is item_id
            item_id_for_f1 = train[col_need_for_computing_f1].iloc[val_idx].values.reshape(len(val_idx))                 
            print("Fold ", fold_id, ":")            
            #--------------------
            # covert pd.DataFrame into lgb.Dataset
            #--------------------
            if use_which_model == 'lgb':
                dtrain = lgb.Dataset(devel, label= y_devel, free_raw_data = False)
                dvalid = lgb.Dataset(valid, label= y_valid, free_raw_data = False, reference= dtrain)
                        
                #evals_result = {} # for saving the evaluation metric of validating set during training
                model = lgb.train(params = fit_params, 
                                      train_set = dtrain, 
                                      num_boost_round = 10,
                                      valid_sets = dvalid, 
                                      #evals_result = evals_result,
                                      feval = customized_eval(data=item_id_for_f1,threshold = 0.5, verbose = True), 
                                     )
            #-----------------------
            # feature importance for each fold
            #----------------------- 
            if len(model.feature_importances_) == len(features):  # some bugs in catboost?
                self.FeatImp['fold' + str(fold_id)] = model.feature_importances_ / model.feature_importances_.sum()


            #----------------------
            # compute the score of each fold
            #----------------------
            predictions = model.predict(valid) # 1-D array with shape of (num_samples,)
            fold_score = model.best_score['valid_0']['f1-score on sentence-level']
            print("Fold ", fold_id, " f1-score : ", fold_score) 
            
            #----------------------
            # compute final measure of performance(Average)
            #----------------------
           
            full_score += fold_score / len(self.fold_ids) # len(self.fold_ids) == n_splits

            if prepare_stacking:
                train[name].iloc[val_idx] = predictions  
                test_predictions = model.predict(test[features])
                test[name] += test_predictions / len(self.fold_ids)


        #----------------------
        # save
        #----------------------
        self.FeatImp.to_csv(feature_importance_dir, index = False)
        print("Final score: ", full_score)
        return full_score

class OrderedCounter(Counter, OrderedDict):
    'Counter that remembers the order elements are first encountered'

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)
    

def customized_eval(data, threshold = 0.5, verbose = False):
    # closure for feeding parameter into custom_system_f1 and return this function
    def custom_system_f1(y_pred, y):
        '''
        It's a customized evaluation metric for computing f1-score on sentence-level.
        
        Args:
        If binary classificiton:
            y_pred: 1-D array, with shape of (num_sample, ), where each elemeent is prob belong to the class 1.
            y: same shape as y_pred.
            
        If multi-class classification:
            y_pred: 2-D array, with shape of (nun_sample, num_class), where each class is prob belong to this class.
        
            data: 1-D array, with shape of (num_sample,), where each elemeent is item_id.
        Return:
            f1-score on sentence-level instead of token-level.
        ''' 
        # get y_true
        y_true = y.get_label().astype("int")
        # get y_pred
        y_pred = np.array([1 if p > threshold else 0 for p in y_pred])
        # get helper dict
        id_lengh_dict = OrderedCounter(list(data)) # need Counter is ordered key
        #---------------------
        # initialization
        #---------------------
        ix = 0
        correct_preds, total_correct, total_preds = 0., 0., 0.

        for item_id, item_length in id_lengh_dict.items():
            y_t_sentence = list(y_true[ix: ix + item_length])
            y_p_sentence = list(y_pred[ix: ix + item_length])
            #----------
            # core
            #----------
            if all(v == 0 for v in y_t_sentence):
                pass
            else:
                # there is exiting atual y_true
                total_correct += 1.0
                if np.array_equal(y_t_sentence, y_p_sentence):
                    # givne the case that we have actual y_ture and y_ture == y_pred
                    correct_preds += 1.0
            if all(v == 0 for v in y_p_sentence):
                pass
            else:
                total_preds += 1.0
            ix += item_length
        #----------
        # output
        #----------
        p   = correct_preds / total_preds if correct_preds > 0 else 0
        r   = correct_preds / total_correct if correct_preds > 0 else 0
        f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        if verbose == True:
            print('f1: {}'.format(f1))
            print('precision: {}'.format(p))
            print('recall: {}'.format(r))
        return 'f1-score-on-sentence-level', f1, True # (eval_name, eval_result, is_higher_better)

    return custom_system_f1

#--------------------
# logging config
#--------------------

def init_logging(log_dir):
    '''
    for recording the experiments.

    log_dir: path
    '''
    #--------------
    # setting
    #--------------
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    date_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
    log_file = 'log_{}.txt'.format(date_str)
    #--------------
    # config
    #--------------
    logging.basicConfig(
        filename = os.path.join(log_dir, log_file),
        level = logging.INFO,
        format = '[[%(asctime)s]] %(message)s',
        datefmt = '%m/%d/%Y %I:%M:%S %p'
    )
    logging.getLogger().addHandler(logging.StreamHandler())
