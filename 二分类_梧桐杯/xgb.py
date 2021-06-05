import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn import  metrics
from sklearn.feature_selection import f_regression 
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import f1_score,precision_recall_fscore_support
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn import preprocessing
LE = preprocessing.LabelEncoder()


########
#数据导入
#########
train_set=pd.read_csv('C:/Users/27Palos/Desktop/train_set.csv')

test_set=pd.read_csv('C:/Users/27Palos/Desktop/result_predict_B.csv')

train_label=pd.read_csv('C:/Users/27Palos/Desktop/train_label.csv')

dataType=pd.DataFrame(train_set.dtypes).rename(columns={0:'dataType'})
dataNull=pd.DataFrame(train_set.isnull().sum()).rename(columns={0:'nullSum'})
dataInf=pd.concat([dataType,dataNull],axis=1)

#x1性别和x5细分市场编码
X1_map={'先生':1,'女士':2}
train_set['X1']=train_set['X1'].map(X1_map)
test_set['X1']=test_set['X1'].map(X1_map)

X5_map={'大众用户':0,'农村用户':1,'集团用户':2,'校园用户':3}
train_set['X5']=train_set['X5'].map(X5_map)
test_set['X5']=test_set['X5'].map(X5_map)

#用户独热编码
dummies = pd.get_dummies(train_set['X5'],prefix='X5')
train_set = pd.concat([train_set,dummies],axis=1)

dummies2 = pd.get_dummies(test_set['X5'],prefix='X5')
test_set = pd.concat([test_set,dummies2],axis=1)


#合并训练集与label

train_set=pd.merge(train_set,train_label,on='user_id',how='left')
train_set.sort_values(['label'], ascending=False, inplace=True)
train_set.index=[x for x in range(140000)]


###########
#数据清洗
###########

#简单空值填充
feature_names = list(filter(lambda x: x not in ['product_no','label', 'user_id'],
                            train_set.columns))
#均值
cols=['X6','X7','X8','X9','X10','X11','X12','X13','X14',
      'X15','X16','X17','X18','X19','X20','X21','X22','X23',
      'X32','X33']
for col in cols:
    col_mean=train_set[col].mean()
    train_set[col]=train_set[col].fillna(col_mean)
    test_set[col]=test_set[col].fillna(col_mean)

#0
cols=['X26','X27','X38']
for col in cols:
    train_set[col]=train_set[col].fillna(0)
    test_set=test_set.fillna(0)
    
#-1
for col in feature_names:
    train_set[col]=train_set[col].fillna(-1)
    test_set[col]=test_set[col].fillna(-1)

#去掉无关列
del train_set['product_no']
del test_set['product_no']
   

###########
#特征工程
###########
#敏感信息离散特征
def getMG_feature(feat):
    data=feat.copy()
    #X3星级离散特征提取
    data['X3_MG']=data['X3'].map(lambda x: 1 if x>=4 else 0)
    #X26宽带带宽离散特征提取
    data['X26_MG']=data['X26'].map(lambda x: 1 if x>=100 else 0)
    #X32用户套餐总值离散特征提取
    data['X32_MG']=data['X32'].map(lambda x: 1 if abs(x-98.5)==0.5 else 0)
    
    return data

train_set=getMG_feature(train_set)
test_set=getMG_feature(test_set)

#用户特征提取
def getYH_feature(feat):
    data=feat.copy()
    #四种用户消费信息均值
    keys=['X5']
    pivot=pd.pivot_table(data,index=keys,values='X15',aggfunc=np.mean)
    pivot=pd.DataFrame(pivot).rename(columns=
                            {'X15':'YH_apru_mean'}).reset_index()
    data=pd.merge(data,pivot,on=keys,how='left')
    
    pivot=pd.pivot_table(data,index=keys,values='X16',aggfunc=np.mean)
    pivot=pd.DataFrame(pivot).rename(columns=
                            {'X16':'YH_dou_mean'}).reset_index()
    data=pd.merge(data,pivot,on=keys,how='left')
    
    pivot=pd.pivot_table(data,index=keys,values='X17',aggfunc=np.mean)
    pivot=pd.DataFrame(pivot).rename(columns=
                            {'X17':'YH_mou_mean'}).reset_index()
    data=pd.merge(data,pivot,on=keys,how='left')
    del data['X5']#去掉辅助列
    
    #高低星级消费信息
    keys=['X3_MG']
    pivot=pd.pivot_table(data,index=keys,values='X15',aggfunc=np.mean)
    pivot=pd.DataFrame(pivot).rename(columns=
                            {'X15':'XJ_apru_mean'}).reset_index()
    data=pd.merge(data,pivot,on=keys,how='left')
    
    pivot=pd.pivot_table(data,index=keys,values='X16',aggfunc=np.mean)
    pivot=pd.DataFrame(pivot).rename(columns=
                            {'X16':'XJ_dou_mean'}).reset_index()
    data=pd.merge(data,pivot,on=keys,how='left')
    
    pivot=pd.pivot_table(data,index=keys,values='X17',aggfunc=np.mean)
    pivot=pd.DataFrame(pivot).rename(columns=
                            {'X17':'XJ_mou_mean'}).reset_index()
    data=pd.merge(data,pivot,on=keys,how='left')
    
    return data

train_set=getYH_feature(train_set)
test_set=getYH_feature(test_set)

#三个月消费特征提取
def getXF_feature(feat,DS):
    data=feat.copy()
    #消费均值正反排序
    data['arpu_T_rank']=data['X15'].rank(ascending=True)
    data['arpu_F_rank']=data['X15'].rank(ascending=False)
    data['arpu_T_rank']=data['arpu_T_rank'].map(lambda x:float((x-DS/2)/DS))
    data['arpu_F_rank']=data['arpu_F_rank'].map(lambda x:float((x-DS/2)/DS))
    
    data['dou_T_rank']=data['X16'].rank(ascending=True)
    data['dou_F_rank']=data['X16'].rank(ascending=False)
    data['dou_T_rank']=data['dou_T_rank'].map(lambda x:float((x-DS/2)/DS))
    data['dou_F_rank']=data['dou_F_rank'].map(lambda x:float((x-DS/2)/DS))
    
    data['mou_T_rank']=data['X17'].rank(ascending=True)
    data['mou_F_rank']=data['X17'].rank(ascending=False)
    data['mou_T_rank']=data['mou_T_rank'].map(lambda x:float((x-DS/2)/DS))
    data['mou_F_rank']=data['mou_F_rank'].map(lambda x:float((x-DS/2)/DS))
    '''
    #消费最值情况
    data['apru_max']=list(map(lambda x,y,z:max([x,y,z]),data['X6'],
                              data['X7'],data['X8']))
    data['dou_max']=list(map(lambda x,y,z:max([x,y,z]),data['X9'],
                              data['X10'],data['X11']))
    data['mou_max']=list(map(lambda x,y,z:max([x,y,z]),data['X12'],
                              data['X13'],data['X14']))

    data['apru_min']=list(map(lambda x,y,z:min([x,y,z]),data['X6'],
                              data['X7'],data['X8']))
    data['dou_min']=list(map(lambda x,y,z:min([x,y,z]),data['X9'],
                              data['X10'],data['X11']))
    data['mou_min']=list(map(lambda x,y,z:min([x,y,z]),data['X12'],
                              data['X13'],data['X14']))
    #消费的标准差
    data['apru_var']=list(map(lambda x,y,z:np.std([x,y,z],ddof=1),
                              data['X6'],data['X7'],data['X8']))
    data['dou_var']=list(map(lambda x,y,z:np.std([x,y,z],ddof=1),
                              data['X9'],data['X10'],data['X11']))
    data['mou_var']=list(map(lambda x,y,z:np.std([x,y,z],ddof=1),
                              data['X12'],data['X13'],data['X14']))
    '''
    return data

train_set=getXF_feature(train_set,140000)
test_set=getXF_feature(test_set,10000)

#超额特征提取
def getCE_feature(feat):
    data=feat.copy()
    #三个月语音流量超额均值
    data['yy_mean']=list(map(lambda x,y,z:(x+y+z)/3,
                             data['X18'],data['X19'],data['X20']))
    data['ll_mean']=list(map(lambda x,y,z:(x+y+z)/3,
                             data['X21'],data['X22'],data['X23']))
    '''
    #三个月语音流量超额最值
    data['yy_max']=list(map(lambda x,y,z:max(x,y,z),
                             data['X18'],data['X19'],data['X20']))
    data['ll_max']=list(map(lambda x,y,z:max(x,y,z),
                             data['X21'],data['X22'],data['X23']))
    
    data['yy_min']=list(map(lambda x,y,z:min(x,y,z),
                             data['X18'],data['X19'],data['X20']))
    data['ll_min']=list(map(lambda x,y,z:min(x,y,z),
                             data['X21'],data['X22'],data['X23']))
    '''
    return data

#train_set=getCE_feature(train_set)
#test_set=getCE_feature(test_set)

def getQT_feature(feat):
    data=feat.copy()
    #5G普及度
    data['QT_5G']=list(map(lambda x,y:1 if (x==1 or y==1) else 0,
                           data['X42'],data['X43']))
    
    #换机玩家当月各种消费均值(6 9 12)
    data['xj6']=list(map(lambda x,y:x if y==0 else 0,data['X6'],data['X41']))
    data['xj9']=list(map(lambda x,y:x if y==0 else 0,data['X9'],data['X41']))
    data['xj12']=list(map(lambda x,y:x if y==0 else 0,data['X12'],data['X41']))
    
    del data['X41']
    return data

train_set=getQT_feature(train_set)
test_set=getQT_feature(test_set)

#########
#数据观察
#########
#打标率观察器get_data_label
def get_data_label(train,s):
    pivot=pd.pivot_table(train,index=s,values='label',aggfunc=len)
    pivot=pd.DataFrame(pivot).rename(columns={'label':'sums'}).reset_index()
    sdata=pivot.copy()
    pivot=pd.pivot_table(train,index=s,values='label',aggfunc=np.sum)
    pivot=pd.DataFrame(pivot).rename(columns={'label':'label_sums'}).reset_index()
    sdata=pd.merge(sdata,pivot,on=s,how='left')
    sdata[s+'_label_rate']=list(map(lambda x,y:x/y if y!=0 else 0,sdata['label_sums'],sdata['sums']))
    return sdata

#特征数
feature_names = list(filter(lambda x: x not in ['product_no','label', 'user_id'],
                            train_set.columns))

#数据打标分布观察
MG_list=[]
DE_list=[]
for col in feature_names:
    MG=get_data_label(train_set,col)
    MG_list.append(MG)
    DE=train_set[col].describe()
    DE_list.append(DE)
    #plt.scatter(train_set[col], train_set['label'], color='r', marker='o')
    #plt.title(col)
    #plt.show()
    

#########
#XGB模型 
#########   
#自定义测评函数F1
def myFeval(preds,dtrain):
    label=dtrain.get_label()
    preds = (preds >= 0.5)*1
    label = (label >= 0.5)*1
    F=f1_score(label,preds)  
    return 'myFeval', F    

#模型
def model_xgb_cv(train, test):
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'silent': 1,
              'eta': 0.028,
              'max_depth':9,
              'min_child_weight': 2,
              'gamma': 0.1,
              'lambda': 1,
              'alpha': 1,
              'colsample_bylevel': 0.8,
              'colsample_bytree': 0.8,
              'subsample': 0.9,
              'scale_pos_weight': 1,
              'max_delta_step': 1,
              'seed':2021}
    
    score_list=[]
    train_Y= train['label']
    train_X=train.drop(columns=['label','user_id'],axis=1)
    KF = StratifiedKFold(5,True,2021)
    f1=0
    f1_list=[]
    for fold_id, (train_index, val_index) in enumerate(KF.split(train_X,train_Y)):
        train_x=train_X.loc[train_index]
        val_x = train_X.loc[val_index]
        train_y=train_Y.loc[train_index]
        val_y = train_Y.loc[val_index]
        
        dtrain = xgb.DMatrix(train_x, label=train_y)
        dtest = xgb.DMatrix(test.drop(['user_id'], axis=1))
        
        print('\nFold_{} Training ================================\n'.format(fold_id+1))
        
        watchlist = [(dtrain, 'train')]
        #watchlist = [(dtrain, 'train'),(dval,'val')]
        model = xgb.train(params, dtrain, num_boost_round=1000,
                          evals=watchlist,feval=myFeval)
        dval_v = xgb.DMatrix(val_x)
        predict = model.predict(dval_v)  
        predict = (predict >= 0.5)*1
        f1_list.append(metrics.f1_score(val_y,predict))
        f1=f1+metrics.f1_score(val_y,predict)
        predict = model.predict(dtest)  
        predict = pd.DataFrame(predict, columns=['label'])
        score_list.append(predict['label'])
        
    
    feat_importance = pd.DataFrame(columns=['feature_name', 'importance'])
    feat_importance['feature_name'] = model.get_score().keys()
    feat_importance['importance'] = model.get_score().values()
    feat_importance.sort_values(['importance'], ascending=False, inplace=True)
    
    score_list=pd.DataFrame(score_list)
    predict=score_list.mean().rename('label')
    #predict=predict.map(lambda x:1 if x>=0.5 else 0)
    result = pd.concat([test_set['user_id'], predict], axis=1)
    f1=f1/5
    print("Mean F1_score :",f1,'\n')
    return result,f1_list    

#result,f1_list= model_xgb_cv(train_set, test_set)
result,YzTs=model_xgb_cv(train_set,test_set)

# 保存
result.to_csv('C:/Users/27Palos/Desktop/Xgb.csv', index=False)
train_set.to_csv('C:/Users/27Palos/Desktop/feature.csv', index=False)

#0.6806706194782226
