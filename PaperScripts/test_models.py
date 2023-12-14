# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from convst.utils.dataset_utils import return_all_dataset_names, return_all_univariate_dataset_names
from convst.utils.experiments_utils import cross_validate_UCR_UEA

from convst.classifiers import R_DST_Ensemble, R_DST_Ridge
# 忽略所有warnings
import warnings
warnings.filterwarnings('ignore')

print("导入成功")
#n_cv = 1 to test only on original train test split.
n_cv = 30
n_jobs = -1
csv_name = 'CV_{}_results_default.csv'.format(
    n_cv)

# 测试的数据集列表，这里使用所有的数据集（单变量、多变量、可变长度等），可以在dataset_utils中查看其他选择。
# 例如，要测试所有数据集，修改为：
# dataset_names = return_all_dataset_names()
dataset_names = return_all_univariate_dataset_names()


# 测试的模型列表，这里使用所有的模型，可以在classifiers中查看其他选择。
dict_models = {
    "RDST Prime":R_DST_Ridge,
    "RDST Ensemble Prime":R_DST_Ensemble,
}
# False：不继续上次的实验。True：继续上次的实验。
resume=False
# 初始化结果DataFrame
if resume :
    df = pd.read_csv(csv_name, index_col=0)
else:
    df = pd.DataFrame(0, index=np.arange(dataset_names.shape[0]*len(dict_models)),
                      columns=['dataset','model','acc_mean','acc_std',
                               'time_mean','time_std']
    )
    df.to_csv(csv_name)

for model_name, model_class in dict_models.items():
    print("编译 {}".format(model_name))
    X = np.random.rand(5,3,50)
    y = np.array([0,0,1,1,1])
    if model_name == 'R_DST_Ridge':
        model_class(n_shapelets=1, prime_dilations=True,n_jobs=-1).fit(X,y).predict(X)
    if model_name == 'R_DST_Ensemble':
        model_class(n_shapelets_per_estimator=1, prime_dilations=True,n_jobs=-1).fit(X,y).predict(X)



i_df=0
for name in dataset_names:
    print(name)
    for model_name, model_class in dict_models.items():
        print(model_name)
        if pd.isna(df.loc[i_df, 'acc_mean']) or df.loc[i_df, 'acc_mean'] == 0.0:
            pipeline = model_class(
                n_jobs=n_jobs
            )
            
            # 默认情况下，使用准确率作为评分，但也可以传递其他评分器作为参数（例如，默认评分器为{"accuracy":accuracy_score}）
            _scores = cross_validate_UCR_UEA(n_cv, name).score(pipeline)
            df.loc[i_df, 'acc_mean'] = _scores['accuracy'].mean()
            df.loc[i_df, 'acc_std'] = _scores['accuracy'].std()
            df.loc[i_df, 'time_mean'] = _scores['time'].mean()
            df.loc[i_df, 'time_std'] = _scores['time'].std()
            df.loc[i_df, 'dataset'] = name
            df.loc[i_df, 'model'] = model_name
            df.to_csv(csv_name)
        else:
            print('跳过 {} : {}'.format(model_name, df.loc[i_df, 'acc_mean']))
        i_df+=1
    print('---------------------')    

