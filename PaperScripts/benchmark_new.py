# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
# 忽略所有warning
import warnings
warnings.filterwarnings('ignore')

from sklearn.base import clone

from convst.utils.dataset_utils import load_UCR_UEA_dataset_split
from convst.classifiers import R_DST_Ridge, R_DST_Ensemble

from sktime.classification.hybrid import HIVECOTEV2
from sktime.classification.dictionary_based import TemporalDictionaryEnsemble
from sktime.classification.interval_based import DrCIF
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.classification.kernel_based import RocketClassifier

from timeit import default_timer as timer

# 定义计时函数
def time_pipe(pipeline, X_train, y_train):
    """
    计算管道的运行时间。

    参数：
    pipeline：管道对象
    X_train：训练数据集
    y_train：训练标签集

    返回：
    运行时间（秒）
    """
    t0 = timer()
    pipeline.fit(X_train, y_train)
    pipeline.predict(X_train)
    t1 = timer()
    return t1-t0


# 验证步骤的数量
n_cv = 10
# 每种方法的并行线程数
n_jobs=90

models = {'RDST Prime':R_DST_Ridge(n_jobs=n_jobs, prime_dilations=True),
          'RDST Ensemble Prime':R_DST_Ensemble(n_jobs=n_jobs, prime_dilations=True),
          'RDST':R_DST_Ridge(n_jobs=n_jobs, prime_dilations=False),
          'RDST Ensemble':R_DST_Ensemble(n_jobs=n_jobs, prime_dilations=False)}

# 执行所有模型一次以进行可能的numba编译
X_train, _, y_train, _, _ = load_UCR_UEA_dataset_split("SmoothSubspace")
for name in models:
    time_pipe(clone(models[name]), X_train, y_train)

    # [样本基准]:
csv_name = 'n_samples_benchmarks.csv'    

X_train, _, y_train, _, _ = load_UCR_UEA_dataset_split("Crop")

# 为了在我们的集群上获得结果，必须减少样本数量。
n_samples = X_train.shape[0]//3

stp = n_samples//6
lengths = np.arange(stp,(n_samples)+stp,stp)
df = pd.DataFrame(index=lengths)
for name in models.keys():
    df[name] = pd.Series(0, index=df.index)

from sklearn.utils import resample

for l in lengths:
    x1 = resample(X_train, replace=False, n_samples=l, stratify=y_train, random_state=0)
    y1 = resample(y_train, replace=False, n_samples=l, stratify=y_train, random_state=0)
    print(x1.shape)
    for name in models:
        timing = []
        for i_cv in range(n_cv):
            print("{}/{}/n_cv:{}".format(name, l, i_cv))
            mod = clone(models[name])
            timing.append(time_pipe(mod, x1, y1))
        df.loc[l, name] = np.mean(timing)
        df.loc[l, name+'_std'] = np.std(timing)
        df.to_csv(csv_name)

csv_name = 'benchmark.csv'    

X_train, _, y_train, _, _ = load_UCR_UEA_dataset_split("Rock")
# 为了在我们的集群上获得结果，必须减少时间戳数量。
n_timestamps = X_train.shape[2]

stp = n_timestamps//6
lengths = np.arange(stp,n_timestamps+stp,stp)
df = pd.DataFrame(index=lengths)
for name in models.keys():
    df[name] = pd.Series(0, index=df.index)
