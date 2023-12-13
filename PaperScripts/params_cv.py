# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from convst.utils.dataset_utils import (load_sktime_arff_file_resample_id,
    return_all_dataset_names, load_sktime_dataset_split)
from convst.utils.experiments_utils import UCR_stratified_resample, run_pipeline

from convst.transformers import R_DST
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler 

#TODO : UPDATE TO THE NEW API

csv_name = 'params_csv.csv'
n_cv = 10
base_UCR_resamples_path = r"/home/prof/guillaume/Shapelets/resamples/"


params = {
    'n_shapelets':[100,1000,10000,20000,30000],
    'p_norm':[1.0,0.9,0.8,0.7,0.6,0.5],
    'shapelet_sizes':[[5],[7],[9],[11], [5,7,9],[7,9,11]],
    'percentiles': [[0,5],[5,10],[10,15],[15,20],[0,10],[5,15],[10,20]]
}

defaults = {'n_shapelets':10_000,'p_norm':0.9, 
            'shapelet_sizes':[7,9,11], 'percentiles':[5,15]
            }

# In[]:

df = pd.DataFrame(columns=['Dataset','n_shapelets','p_norm','shapelet_sizes','percentiles','acc_mean','acc_std'])

dataset_names = return_all_dataset_names()
"""
If you want the same ones as used in the paper, look at the params_csv.csv :
    df = pd.read_csv('results\params_csv.csv',index_col=0)
    dataset_names = df['Dataset'].unique()
"""
dataset_names = np.random.choice(dataset_names, size=40, replace=False)

#Small run for numba compilations if needed
pipe = make_pipeline(
    R_DST(n_shapelets=1),
    StandardScaler(with_mean=True),
    RidgeClassifierCV(alphas=np.logspace(-6,6,20))
)
X_train, X_test, y_train, y_test, _ = load_sktime_dataset_split(
    'SmoothSubspace', normalize=True)
pipe.fit(X_train, y_train)
p = pipe.predict(X_test)

# In[]

count = 0
for name in dataset_names:
    ds_path = base_UCR_resamples_path+"{}/{}".format(name, name)
    X_train, X_test, y_train, y_test, _ = load_sktime_arff_file_resample_id(
        ds_path, 0, normalize=True)
    splitter = UCR_stratified_resample(n_cv, ds_path)
    print(name)
    for k in params.keys():
        print(k)
        ks = set(defaults.keys())
        ks.remove(k)
        p_dict = {key:defaults[key] for key in ks}
        p_dict.update({k:None})
        for val in params[k]:
            p_dict[k] = val            
            
            pipe = make_pipeline(
                R_DST(**p_dict),
                StandardScaler(with_mean=True),
                RidgeClassifierCV(alphas=np.logspace(-6,6,20))
            )
            acc_mean, acc_std, f1_mean, f1_std, time_mean, time_std = run_pipeline(
                pipe, X_train, X_test, y_train, y_test, splitter, 10)
            df.loc[count,'Dataset'] = name
            df.loc[count, k] = str(val)
            df.loc[count, 'acc_mean'] = acc_mean
            df.loc[count, 'acc_std'] = acc_std
            count += 1
            df.to_csv(csv_name)
    print('-----')
                
