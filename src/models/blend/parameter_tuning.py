import time
import sys
import os
import pandas as pd
from datetime import datetime
from utils import HyperParameterTuning
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from tf_utils import init_logging
# import logging

#---------------------
# setting
#---------------------
log_dir = 'logs'
BINARY_SCENARIO = None
#---------------------
# load features
#---------------------
feature_dir = '../../features/lazada_and_amazon/all_features.h5'
df = pd.read_hdf(feature_dir)
#---------------------
# label post-processing
#---------------------
if df.label.nunique() == 2: 
    BINARY_SCENARIO = True
    # binary class
    df['label'] = df.label.apply(lambda x: 1 if x == 2 else 0) # for customized f1 score inference of lgb
else:
    # multi-class(B, I or O)
    pass

#logging.info('BINARY_SCENARIO : {}'.format(BINARY_SCENARIO))

#-----------------------
# parameter tuning
#-----------------------

# setting
features = df.columns.tolist()[7:]
target = 'label'
n_splits = 2
pbounds = {
'num_leaves': (25, 50),
'lambda_l2': (0.0, 0.05),
'lambda_l1': (0.0, 0.05),
'min_child_samples': (20, 120),
'bagging_fraction': (0.5, 1.0),
'feature_fraction': (0.5, 1.0),
} # 6 parameters to tune

# Create objec HyperParameterTuning for helping us tuning
HP_tuning = HyperParameterTuning(train = df, features= features, target = target, n_splits = n_splits, 
	log_dir = log_dir, params_bound = pbounds)

#-----------------------
# save
#-----------------------
date_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
s = time.time()
result = HP_tuning.param_tuning(init_points = 5, num_iter = 25)
e = time.time()
print ('It took {} mins'.format((e-s)/60.0))
print (pp.pprint(result.res['max']['max_params']))

result.points_to_csv('logs/param_tuning_for_lgb_{}.csv'.format(date_str))



