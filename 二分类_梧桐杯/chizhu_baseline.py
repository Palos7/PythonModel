#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
os.listdir("../data/")


# In[2]:


train = pd.read_csv("../data/train_set.csv")
label = pd.read_csv("../data/train_label.csv")
testa = pd.read_csv("../data/result_predict_A.csv")
test = pd.read_csv("../data/result_predict_B.csv")


# In[3]:


data = pd.concat([train,testa,test],ignore_index=True)


# In[4]:


data['product_top3']=data['product_no'].map(lambda x:int(str(x)[:3]))
data['product_top4']=data['product_no'].map(lambda x:int(str(x)[:4]))
data['product_top5']=data['product_no'].map(lambda x:int(str(x)[:5]))

data['product_tail3']=data['product_no'].map(lambda x:int(str(x)[-3:]))
data['X1'] = data['X1'].map({"女士":0,"先生":1})
data['X5'] = data['X5'].map({"大众用户":0,"农村用户":1,"校园用户":2,"集团用户":3})
data['X5_freq'] = data['X5'].map(data['X5'].value_counts())
data['product_top3_freq']=data['product_top3'].map(data['product_top3'].value_counts())


# In[5]:


data['arpu_mean'] = data.apply(lambda x:(x['X6']+x['X7']+x["X8"])/3,1)


# In[6]:


data['arpu_diff1']=data.apply(lambda x:x['X6']-x['X7'],1)
data['arpu_diff2']=data.apply(lambda x:x['X7']-x['X8'],1)


# In[7]:


data['arpu_diff3']=data.apply(lambda x:x['X6']-x['X8'],1)


# In[8]:


data['dou_mean'] = data.apply(lambda x:(x['X9']+x['X10']+x["X11"])/3,1)
data['dou_diff1'] = data.apply(lambda x:x['X9']-x['X10'],1)
data['dou_diff2'] = data.apply(lambda x:x['X10']-x['X11'],1)
data['dou_diff3'] = data.apply(lambda x:x['X10']-x['X12'],1)

data['mou_mean'] = data.apply(lambda x:(x['X12']+x['X13']+x["X14"])/3,1)
data['mou_diff1'] = data.apply(lambda x:x['X12']-x['X13'],1)
data['mou_diff2'] = data.apply(lambda x:x['X13']-x['X14'],1)
data['mou_diff3'] = data.apply(lambda x:x['X13']-x['X15'],1)


# In[9]:


data = data.merge(label,on="user_id",how="left")


# In[10]:


data['X5'] = data['X5'].fillna(4)


# In[11]:


data['x5_x33_mean'] = data.groupby('X5',as_index=False)['X33'].transform("mean")
data['x5_x33_std'] = data.groupby('X5',as_index=False)['X33'].transform("std")
data['x5_x33_max'] = data.groupby('X5',as_index=False)['X33'].transform("max")
data['x5_x33_min'] = data.groupby('X5',as_index=False)['X33'].transform("min")
data['x5_x33_skew'] = data.groupby('X5',as_index=False)['X33'].transform("skew")
# data['x5_x33_kurt'] = data.groupby('X5',as_index=False)['X33'].transform("kurt")

data['x5_x32_mean'] = data.groupby('X5',as_index=False)['X32'].transform("mean")
data['x5_x32_std'] = data.groupby('X5',as_index=False)['X32'].transform("std")
data['x5_x32_max'] = data.groupby('X5',as_index=False)['X32'].transform("max")
data['x5_x32_min'] = data.groupby('X5',as_index=False)['X32'].transform("min")
data['x5_x32_skew'] = data.groupby('X5',as_index=False)['X32'].transform("skew")
# data['x5_x32_kurt'] = data.groupby('X5',as_index=False)['X32'].transform("kurt")

# data['x5_x6_mean'] = data.groupby('X5',as_index=False)['X6'].transform("mean")
# data['x5_x6_std'] = data.groupby('X5',as_index=False)['X6'].transform("std")
# data['x5_x6_max'] = data.groupby('X5',as_index=False)['X6'].transform("max")
# data['x5_x6_min'] = data.groupby('X5',as_index=False)['X6'].transform("min")
# data['x5_x6_skew'] = data.groupby('X5',as_index=False)['X6'].transform("skew")

# data['x5_x12_mean'] = data.groupby('X5',as_index=False)['X12'].transform("mean")
# data['x5_x12_std'] = data.groupby('X5',as_index=False)['X12'].transform("std")
# data['x5_x12_max'] = data.groupby('X5',as_index=False)['X12'].transform("max")
# data['x5_x12_min'] = data.groupby('X5',as_index=False)['X12'].transform("min")
# data['x5_x12_skew'] = data.groupby('X5',as_index=False)['X12'].transform("skew")


# In[12]:


data['level_time']=data['X3']+data['X4']


# In[13]:


data['x2_x33_mean'] = data.groupby('X2',as_index=False)['X33'].transform("mean")
data['x2_x33_std'] = data.groupby('X2',as_index=False)['X33'].transform("std")

data['x2_x32_mean'] = data.groupby('X2',as_index=False)['X32'].transform("mean")
data['x2_x32_std'] = data.groupby('X2',as_index=False)['X32'].transform("std")


# In[14]:


train = data[data['label'].notnull()]
test = data[data['label'].isnull()]


# In[15]:


need = pd.read_csv("../data/result_predict_B.csv")


# In[16]:


need = need[['user_id']]


# In[17]:


test = need.merge(test,on="user_id",how='left')


# In[18]:


train.shape,test.shape


# In[19]:


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold,GroupKFold
from itertools import product

class MeanEncoder:
    def __init__(self, categorical_features, n_splits=5, target_type='classification', prior_weight_func=None):
        """
        :param categorical_features: list of str, the name of the categorical columns to encode
        :param n_splits: the number of splits used in mean encoding
        :param target_type: str, 'regression' or 'classification'
        :param prior_weight_func:
        a function that takes in the number of observations, and outputs prior weight
        when a dict is passed, the default exponential decay function will be used:
        k: the number of observations needed for the posterior to be weighted equally as the prior
        f: larger f --> smaller slope
        '''
        >>>example:
        mean_encoder = MeanEncoder(
                        categorical_features=['regionidcity',
                          'regionidneighborhood', 'regionidzip'],
                target_type='regression'
                )

        X = mean_encoder.fit_transform(X, pd.Series(y))
        X_test = mean_encoder.transform(X_test)


        """

        self.categorical_features = categorical_features
        self.n_splits = n_splits
        self.learned_stats = {}

        if target_type == 'classification':
            self.target_type = target_type
            self.target_values = []
        else:
            self.target_type = 'regression'
            self.target_values = None

        if isinstance(prior_weight_func, dict):
            self.prior_weight_func = eval('lambda x: 1 / (1 + np.exp((x - k) / f))', dict(prior_weight_func, np=np))
        elif callable(prior_weight_func):
            self.prior_weight_func = prior_weight_func
        else:
            self.prior_weight_func = lambda x: 1 / (1 + np.exp((x - 2) / 1))

    @staticmethod
    def mean_encode_subroutine(X_train, y_train, X_test, variable, target, prior_weight_func):
        X_train = X_train[[variable]].copy()
        X_test = X_test[[variable]].copy()

        if target is not None:
            nf_name = '{}_pred_{}'.format(variable, target)
            X_train['pred_temp'] = (y_train == target).astype(int)  # classification
        else:
            nf_name = '{}_pred'.format(variable)
            X_train['pred_temp'] = y_train  # regression
        prior = X_train['pred_temp'].mean()

        col_avg_y = X_train.groupby(by=variable, axis=0)['pred_temp'].agg({'mean': 'mean', 'beta': 'size'})
        col_avg_y['beta'] = prior_weight_func(col_avg_y['beta'])
        col_avg_y[nf_name] = col_avg_y['beta'] * prior + (1 - col_avg_y['beta']) * col_avg_y['mean']
        col_avg_y.drop(['beta', 'mean'], axis=1, inplace=True)

        nf_train = X_train.join(col_avg_y, on=variable)[nf_name].values
        nf_test = X_test.join(col_avg_y, on=variable).fillna(prior, inplace=False)[nf_name].values

        return nf_train, nf_test, prior, col_avg_y

    def fit_transform(self, X, y):
        """
        :param X: pandas DataFrame, n_samples * n_features
        :param y: pandas Series or numpy array, n_samples
        :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        X_new = X.copy()
        if self.target_type == 'classification':
            skf = StratifiedKFold(self.n_splits ,random_state=1017, shuffle=True)
        else:
            skf =  StratifiedKFold(self.n_splits ,random_state=1017, shuffle=True)

        if self.target_type == 'classification':
            self.target_values = sorted(set(y))
            self.learned_stats = {'{}_pred_{}'.format(variable, target): [] for variable, target in
                                  product(self.categorical_features, self.target_values)}
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}'.format(variable, target)
                X_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(y, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, target, self.prior_weight_func)
                    X_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        else:
            self.learned_stats = {'{}_pred'.format(variable): [] for variable in self.categorical_features}
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                X_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(y, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, None, self.prior_weight_func)
                    X_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        return X_new

    def transform(self, X):
        """
        :param X: pandas DataFrame, n_samples * n_features
        :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        X_new = X.copy()

        if self.target_type == 'classification':
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}'.format(variable, target)
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
                        nf_name]
                X_new[nf_name] /= self.n_splits
        else:
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
                        nf_name]
                X_new[nf_name] /= self.n_splits

        return X_new



# In[20]:


# In[51]:


mean_encoder = MeanEncoder(
                        categorical_features=['X32',"X33"                        ],
                target_type='classification'
                )

train = mean_encoder.fit_transform(train, train['label'])
test = mean_encoder.transform(test)


# In[21]:


features = [x for x in train.columns if x not in ["user_id","label","product_no"]]

label='label'
len(features)


# In[22]:


import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score,f1_score
from sklearn.model_selection import StratifiedKFold,KFold,GroupKFold
def run_cv_model(train, test, target, model_fn, params={}, eval_fn=None, label='model'):
    folds=5
    
    kf = StratifiedKFold(n_splits=folds, random_state=1017, shuffle=True)
#     fold_splits = kf.split(train, target)
#     kf = GroupKFold(n_splits=folds,)
    fold_splits = kf.split(train, target)
    cv_scores = []
    pred_full_test = 0
    
    pred_train = np.zeros((train.shape[0], ))
    feature_importance_df = pd.DataFrame()
    i = 1
    for dev_index, val_index in fold_splits:
        print( label + ' | FOLD ' + str(i) + '/'+str(folds))
        if isinstance(train, pd.DataFrame):
            dev_X, val_X = train.iloc[dev_index], train.iloc[val_index]
            dev_y, val_y = target[dev_index], target[val_index]
        else:
            dev_X, val_X = train[dev_index], train[val_index]
            dev_y, val_y = target[dev_index], target[val_index]
        params2 = params.copy()
        pred_val_y, pred_test_y, importances = model_fn(dev_X, dev_y, val_X, val_y, test, params2)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index] = pred_val_y
       
        if eval_fn is not None:
            cv_score = eval_fn(val_y, pred_val_y>0.34)
            cv_scores.append(cv_score)
          
           
            print(label + ' cv score {}: ACC {} '.format(i, cv_score))
            print("##"*40)
        fold_importance_df = pd.DataFrame()
        fold_importance_df['feature'] =train.columns.values
        fold_importance_df['importance'] =importances
        fold_importance_df['fold'] = i
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)        
        i += 1
#     print('{} cv RMSE scores : {}'.format(label, cv_scores))
    print('{} cv mean ACC score : {}'.format(label, np.mean(cv_scores)))
    print('{} cv std ACC score : {}'.format(label, np.std(cv_scores)))
   

    
    pred_full_test = pred_full_test / float(folds)
    results = {'label': label,
               'train': pred_train, 'test': pred_full_test,
                'cv': cv_scores, 
               'importance': feature_importance_df,
               }
    return results

def runLGB(train_X, train_y, test_X, test_y, test_X2, params):
#     print('Prep LGB')
    d_train = lgb.Dataset(train_X, label=train_y)
    d_valid = lgb.Dataset(test_X, label=test_y)
    watchlist = [d_train, d_valid]
#     print('Train LGB')
    num_rounds = params.pop('num_rounds')
    verbose_eval = params.pop('verbose_eval')
    early_stop = None
    if params.get('early_stop'):
        early_stop = params.pop('early_stop')
    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=num_rounds,
                      valid_sets=watchlist,
#                       fobj=softkappaObj,
                      verbose_eval=verbose_eval,
#                       feval=kappa_scorer,
                      early_stopping_rounds=early_stop)
    print('Predict 1/2')
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
   
    print('Predict 2/2')
    pred_test_y2 = model.predict(test_X2, num_iteration=model.best_iteration)
    
   
    return pred_test_y,pred_test_y2, model.feature_importance()


# In[23]:


params = {
            'objective': 'binary', 
          'boosting': 'gbdt',
          'metric': ['logloss','auc'],
          'num_leaves': 32,
          'learning_rate': 0.01,
          'bagging_fraction': 0.7,
            "bagging_freq":3,
           'feature_fraction': 0.4,
          'verbosity': -1,
          "data_random_seed":17,
          "random_state":1017,
            'num_rounds': 6000,
    'verbose_eval': 200,
    'early_stop':200,
#          'device': 'gpu',
#     'gpu_platform_id': 0,
#     'gpu_device_id': 0,
}
results = run_cv_model(train[features], test[features], train[label], runLGB, params, f1_score , 'LGB')
lgb_train=[r for r in results['train']]
lgb_test=[r for r in results['test']]



# In[30]:


# In[63]:


imports = results['importance'].groupby('feature')['feature', 'importance'].mean().reset_index()
imp=imports.sort_values('importance', ascending=False)
imp.index=range(len(imp))
imp.iloc[:560]


# In[31]:


sub = test[['user_id']]
sub['label']=lgb_test


# In[32]:


sub['label']=sub['label'].map(lambda x :1 if x>0.24 else 0)


# In[33]:


sub['label'].value_counts()


# In[34]:


if not os.path.exists("submit"):
    os.mkdir("submit")
sub.to_csv("submit/baseline_final.csv",index=False)


# In[35]:


train['label'].value_counts()/len(train)


# In[ ]:





# In[ ]:





# In[ ]:




