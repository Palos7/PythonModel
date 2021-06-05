import pandas as pd    
import lightgbm as lgb 
import numpy as np
from sklearn.model_selection import train_test_split
 
    
#打标    
def markdata(data):    
    data_set=data.copy()    
    data_set['label']=list(map(lambda x,y:1 if (y-x).total_seconds()/(60*60*24)<=15 else 0,  data_set['date_received'],data_set['date']))    
    return data_set       
#数据的预处理    
def predo(data):    
    data_set=data.copy()    
    data_set['Distance'].fillna(-1,inplace=True)    
    data_set['Coupon_id'].fillna(-1,inplace=True)    
    data_set['Coupon_id']=data_set['Coupon_id'].map(int)    
    data_set['date_received']=pd.to_datetime(data_set['Date_received'],format='%Y%m%d')    
    data_set['discount_rate']=data_set['Discount_rate'].map(lambda x: float(x) if ':' not in str(x)  else (float(str(x).split(':')[0])-float(str(x).split(':')[1]))/(float(str(x).split(':')[0])))  
    if 'Date' in data_set.columns.tolist():    
        data_set['date']=pd.to_datetime(data_set['Date'],format='%Y%m%d')    
        data_set=markdata(data_set)    
    return data_set    
    
#简单特征提   
def get_simple_feature(label_field):    
    data=label_field.copy()    
    data['cnt']=1    
    feature=data.copy()    
    #用户领卷数量    
    keys=['User_id']    
    prefixs='simple_'+'_'.join(keys)+'_'    
    pivot=pd.pivot_table(data,index=keys,values='cnt',aggfunc=len)    
    pivot=pd.DataFrame(pivot).rename(columns={'cnt':prefixs+'receive_cnt'}).reset_index()    
    feature=pd.merge(feature,pivot,on=keys,how='left')    
    #用户领取特定优惠卷数    
    keys=['User_id','Coupon_id']    
    prefixs='simple_'+'_'.join(keys)+'_'    
    pivot=pd.pivot_table(data,index=keys,values='cnt',aggfunc=len)    
    pivot=pd.DataFrame(pivot).rename(columns={'cnt':prefixs+'receive_cnt'}).reset_index()    
    feature=pd.merge(feature,pivot,on=keys,how='left')    
    #用户当天领取特定优惠卷数量    
    keys=['User_id','Coupon_id','Date_received']    
    prefixs='simple_'+'_'.join(keys)+'_'    
    pivot=pd.pivot_table(data,index=keys,values='cnt',aggfunc=len)    
    pivot=pd.DataFrame(pivot).rename(columns={'cnt':prefixs+'receive_cnt'}).reset_index()    
    feature=pd.merge(feature,pivot,on=keys,how='left')    
    #用户当天领取优惠卷数量    
    keys=['User_id','Date_received']    
    prefixs='simple_'+'_'.join(keys)+'_'    
    pivot=pd.pivot_table(data,index=keys,values='cnt',aggfunc=len)    
    pivot=pd.DataFrame(pivot).rename(columns={'cnt':prefixs+'receive_cnt'}).reset_index()    
    feature=pd.merge(feature,pivot,on=keys,how='left')    
    #用户是否在同一天重复领取特定优惠卷    
    keys=['User_id','Coupon_id','Date_received']    
    prefixs='simple_'+'_'.join(keys)+'_'    
    pivot=pd.pivot_table(data,index=keys,values='cnt',aggfunc=lambda x:1 if len(x)>1 else 0)    
    pivot=pd.DataFrame(pivot).rename(columns={'cnt':prefixs+'repeat_receive'}).reset_index()    
    feature=pd.merge(feature,pivot,on=keys,how='left') 
    #用户领取特定商家优惠卷数量
    keys=['User_id','Merchant_id','Date_received']    
    prefixs='simple_'+'_'.join(keys)+'_'    
    pivot=pd.pivot_table(data,index=keys,values='cnt',aggfunc=lambda x:1 if len(x)>1 else 0)    
    pivot=pd.DataFrame(pivot).rename(columns={'cnt':prefixs+'is_oneday_MerUse'}).reset_index()    
    feature=pd.merge(feature,pivot,on=keys,how='left')
    #用户是否同一天领取特定商家优惠卷数量
    keys=['User_id','Merchant_id']    
    prefixs='simple_'+'_'.join(keys)+'_'    
    pivot=pd.pivot_table(data,index=keys,values='cnt',aggfunc=len)    
    pivot=pd.DataFrame(pivot).rename(columns={'cnt':prefixs+'Mer_receive_cnt'}).reset_index()    
    feature=pd.merge(feature,pivot,on=keys,how='left')
    #用户领取不同距离优惠卷数量    
    keys=['User_id','Distance']    
    prefixs='simple_'+'_'.join(keys)+'_'    
    pivot=pd.pivot_table(data,index=keys,values='cnt',aggfunc=len)    
    pivot=pd.DataFrame(pivot).rename(columns={'cnt':prefixs+'DisReceive_cnt'}).reset_index()    
    feature=pd.merge(feature,pivot,on=keys,how='left')
    
    #用户是否在某个商家最后一次领卷
    keys=['User_id','Merchant_id']    
    prefixs='simple_'+'_'.join(keys)+'_'
    pivot=pd.pivot_table(data,index=keys,values='Date_received',aggfunc=max)
    pivot=pd.DataFrame(pivot).rename(columns={'Date_received':'LastDay'}).reset_index()
    data=pd.merge(data,pivot,on=keys,how='left') 
    feature[prefixs+'IsLastDay']=list(map(lambda x,y:1 if x==y else 0,data['LastDay'],data['Date_received']))
    #用户是否第一次领卷
    keys=['User_id']    
    prefixs='simple_'+'_'.join(keys)+'_'
    pivot=pd.pivot_table(data,index=keys,values='Date_received',aggfunc=min)
    pivot=pd.DataFrame(pivot).rename(columns={'Date_received':'FirstDay'}).reset_index()
    data=pd.merge(data,pivot,on=keys,how='left') 
    feature[prefixs+'IsFirstDay']=list(map(lambda x,y:1 if x==y else 0,data['FirstDay'],data['Date_received']))
    
    #用户领卷距离正反排序
    feature['simple_DisTrue_rank']=feature.groupby(keys)['Distance'].rank(ascending=True)  
    feature['simple_DisFalse_rank']=feature.groupby(keys)['Distance'].rank(ascending=False)
    #用户领卷时间正反排序
    feature['simple_TimeTrue_rank']=feature.groupby(keys)['date_received'].rank(ascending=True)
    feature['simple_TimeFalse_rank']=feature.groupby(keys)['date_received'].rank(ascending=False)
    #用户领卷折扣率正反排序
    feature['simple_RateTrue_rank']=feature.groupby(keys)['discount_rate'].rank(ascending=True)
    feature['simple_RateFalse_rank']=feature.groupby(keys)['discount_rate'].rank(ascending=False)
    
    #商家发放优惠卷数量    
    keys=['Merchant_id']    
    prefixs='simple_'+'_'.join(keys)+'_'    
    pivot=pd.pivot_table(data,index=keys,values='cnt',aggfunc=len)    
    pivot=pd.DataFrame(pivot).rename(columns={'cnt':prefixs+'handout_cnt'}).reset_index()    
    feature=pd.merge(feature,pivot,on=keys,how='left')
    #商家每天发放优惠卷数量    
    keys=['Merchant_id','date_received']    
    prefixs='simple_'+'_'.join(keys)+'_'    
    pivot=pd.pivot_table(data,index=keys,values='cnt',aggfunc=len)    
    pivot=pd.DataFrame(pivot).rename(columns={'cnt':prefixs+'oneday_handout_cnt'}).reset_index()    
    feature=pd.merge(feature,pivot,on=keys,how='left')
    #商家发放特定优惠卷数量    
    keys=['Merchant_id','Coupon_id']    
    prefixs='simple_'+'_'.join(keys)+'_'    
    pivot=pd.pivot_table(data,index=keys,values='cnt',aggfunc=len)    
    pivot=pd.DataFrame(pivot).rename(columns={'cnt':prefixs+'Specialhandout_cnt'}).reset_index()    
    feature=pd.merge(feature,pivot,on=keys,how='left')    
    #商家在同一天重复发放特定优惠卷 
    keys=['Merchant_id','Coupon_id','Date_received']    
    prefixs='simple_'+'_'.join(keys)+'_'    
    pivot=pd.pivot_table(data,index=keys,values='cnt',aggfunc=lambda x:1 if len(x)>1 else 0)    
    pivot=pd.DataFrame(pivot).rename(columns={'cnt':prefixs+'Specialhandout1day_cnt'}).reset_index()    
    feature=pd.merge(feature,pivot,on=keys,how='left')
    #商家对用户距离正反排序
    keys=['Merchant_id','User_id']
    feature['simple_MaUTrue_rank']=feature.groupby(keys)['Distance'].rank(ascending=True)  
    feature['simple_MaUFalse_rank']=feature.groupby(keys)['Distance'].rank(ascending=False)       
    
    #每种优惠卷数量        
    keys=['Coupon_id']
    prefixs='simple_'+'_'.join(keys)+'_' 
    pivot=pd.pivot_table(data,index=keys,values='cnt',aggfunc=len)    
    pivot=pd.DataFrame(pivot).rename(columns={'cnt':prefixs+'Coupon_cnt'}).reset_index()    
    feature=pd.merge(feature,pivot,on=keys,how='left')
    #优惠卷折扣率排序
    feature['simple_RealCoupon_rank']=feature.groupby(keys)['discount_rate'].rank(ascending=True)
    feature['simple_FalseCoupon_rank']=feature.groupby(keys)['discount_rate'].rank(ascending=False)
    #优惠领取距离排序
    feature['simple_RealCouponDis_rank']=feature.groupby(keys)['Distance'].rank(ascending=True)
    feature['simple_FalseCouponDis_rank']=feature.groupby(keys)['Distance'].rank(ascending=False)
    #优惠领取日期排序
    feature['simple_RealCouponDate_rank']=feature.groupby(keys)['date_received'].rank(ascending=True)
    feature['simple_FalseCouponDate_rank']=feature.groupby(keys)['date_received'].rank(ascending=False)
    #每种优惠卷不同距离数量
    keys=['Coupon_id','Distance']
    prefixs='simple_'+'_'.join(keys)+'_' 
    pivot=pd.pivot_table(data,index=keys,values='cnt',aggfunc=len)    
    pivot=pd.DataFrame(pivot).rename(columns={'cnt':prefixs+'Coupon_Distance_cnt'}).reset_index()    
    feature=pd.merge(feature,pivot,on=keys,how='left')
    #每种优惠卷在不同日期的领取数量
    keys=['Coupon_id','Date_received']
    prefixs='simple_'+'_'.join(keys)+'_' 
    pivot=pd.pivot_table(data,index=keys,values='cnt',aggfunc=len)    
    pivot=pd.DataFrame(pivot).rename(columns={'cnt':prefixs+'Coupon_Date_cnt'}).reset_index()    
    feature=pd.merge(feature,pivot,on=keys,how='left')
    #删除辅助的计数列cnt    
    feature.drop(['cnt'],axis=1,inplace=True)    
    return feature    

#距离和时间特征    
def dis_week_feature(label_field):    
    data_set=label_field.copy()    
    data_set['Nulldistance']=data_set['Distance'].map(lambda x:1 if x==-1 else 0)    
    data_set['Date_received']=data_set['Date_received'].map(int)    
    feature=data_set.copy()    
    feature['weekday']=feature['date_received'].map(lambda x:x.weekday())    
    feature['isweekend']=feature['date_received'].map(lambda x:1 if x==5 or x==6 else 0)    
    feature=pd.concat([feature,pd.get_dummies(feature['weekday'],prefix='week')],axis=1)    
    feature.index=range(len(feature))    
    return feature    

#优惠卷特征 
def get_coupon_feature(label_field):    
    data_set=label_field.copy()    
    data_set['isManjian']=data_set['Discount_rate'].map(lambda x:1 if ':'in str(x) else 0)    
    data_set['min_cost']=data_set['Discount_rate'].map(lambda x:0 if ':' not in str(x) else int(str(x).split(':')[0]))    
    return data_set    
        
#历史用户的特征   
def get_user_feature(history_field,label_field):    
    data=history_field.copy()  
    data['Date_received']=data['Date_received'].map(int)    
    data['cnt']=1    
    feature=label_field.copy()   
     
    keys=['User_id']    
    prefixs='history_field_'+'_'.join(keys)+'_'  
     #用户领卷数目  
    pivot=pd.pivot_table(data,index=keys,values='cnt',aggfunc=len)  
    pivot=pd.DataFrame(pivot).rename(columns={'cnt':prefixs+'receive_cnt'}).reset_index()  
    feature=pd.merge(feature,pivot,on=keys,how='left')  
    #用户的消费数目      
    pivot=pd.pivot_table(data[data['Date'].map(lambda x:str(x)!='nan')],index=keys,values='cnt',aggfunc=len)    
    pivot=pd.DataFrame(pivot).rename(columns={'cnt':prefixs+'useout_cnt'}).reset_index()    
    feature=pd.merge(feature,pivot,on=keys,how='left')    
    #用户未消费数目    
    pivot=pd.pivot_table(data[data['Date'].map(lambda x:str(x)=='nan')],index=keys,values='cnt',aggfunc=len)    
    pivot=pd.DataFrame(pivot).rename(columns={'cnt':prefixs+'nouseout_cnt'}).reset_index()    
    feature=pd.merge(feature,pivot,on=keys,how='left')     
    #用户消费数目/领卷数目（消费率）
    pivot=list(map(lambda x,y:x/y if y!=0 else 0,feature[prefixs+'useout_cnt'],  
                                                 feature[prefixs+'receive_cnt']))  
    feature[prefixs+'RandU_rate']=pivot  
    #用户核销数目    
    pivot=pd.pivot_table(data[data['label']==1],index=keys,values='cnt',aggfunc=len)    
    pivot=pd.DataFrame(pivot).rename(columns={'cnt':prefixs+'useoutin15_cnt'}).reset_index()    
    feature=pd.merge(feature,pivot,on=keys,how='left')   
    #用户未核销数目    
    pivot=pd.pivot_table(data[data['label']==0],index=keys,values='cnt',aggfunc=len)   
    pivot=pd.DataFrame(pivot).rename(columns={'cnt':prefixs+'Nuseoutin15_cnt'}).reset_index()    
    feature=pd.merge(feature,pivot,on=keys,how='left') 
    #用户核销数目/用户领卷数目（核销率）
    pivot=list(map(lambda x,y:x/y if y!=0 else 0,feature[prefixs+'useoutin15_cnt'],
                                                 feature[prefixs+'receive_cnt']))
    
    #用户领取并消费优惠卷平均折扣率  
    pivot=pd.pivot_table(data[data['Date'].map(lambda x:str(x)!='nan')],index=keys,values='discount_rate',aggfunc=np.mean)  
    pivot=pd.DataFrame(pivot).rename(columns={'discount_rate':prefixs+'useDiscount_rate_mean'}).reset_index()    
    feature=pd.merge(feature,pivot,on=keys,how='left') 
    #用户领取并消费优惠卷的平均距离  
    data['Distance']=data['Distance'].map(lambda x: None if x==-1 else x)  
    pivot=pd.pivot_table(data[data['Date'].map(lambda x:str(x)!='nan')],index=keys,values='Distance',aggfunc=np.mean)  
    pivot=pd.DataFrame(pivot).rename(columns={'Distance':prefixs+'useDistance_mean'}).reset_index()  
    feature=pd.merge(feature,pivot,on=keys,how='left')
    #用户领取到消费优惠卷的平均时间间隔
    data['JianGe']=list(map(lambda x,y:(x-y).total_seconds()/60*60*24, data['date'],data['date_received']))
    pivot=pd.pivot_table(data[data['JianGe'].map(lambda x:str(x)!='nan')],index=keys,values='JianGe',aggfunc=np.mean)
    pivot=pd.DataFrame(pivot).rename(columns={'JianGe':prefixs+'UseJianGe_mean'}).reset_index()
    feature=pd.merge(feature,pivot,on=keys,how='left')
    
    #在多少不同商家领取优惠卷  
    pivot=pd.pivot_table(data,index=keys,values='Merchant_id',aggfunc=lambda x: len(set(x)))  
    pivot=pd.DataFrame(pivot).rename(columns={'Merchant_id':prefixs+'Mr_cnt'}).reset_index()  
    feature=pd.merge(feature,pivot,on=keys,how='left')  
    #在多少不同商家领取并消费优惠卷  
    pivot=pd.pivot_table(data[data['Date'].map(lambda x:str(x)!='nan')],index=keys,values='Merchant_id',aggfunc=lambda x:len(set(x)))  
    pivot=pd.DataFrame(pivot).rename(columns={'Merchant_id':prefixs+'Mrandu_cnt'}).reset_index()  
    feature=pd.merge(feature,pivot,on=keys,how='left')  
    #在多少不同商家领取并消费优惠卷/在不同商家领取优惠卷  
    feature['MerUse_rate']=list(map(lambda x,y:x/y if y!=0 else 0,feature[prefixs+'Mrandu_cnt'],  
                                                                  feature[prefixs+'Mr_cnt']))  
  
    return feature 

#历史优惠卷特征   
def get_history_coupon(history_field,label_field):  
    data=history_field.copy()  
    data['cnt']=1  
    feature=label_field.copy()  
    #主键  
    keys=['Coupon_id']  
    prefixs='history_field_'+'_'.join(keys)+'_'  
    #不同优惠卷的领取数量  
    pivot=pd.pivot_table(data,index=keys,values='cnt',aggfunc=len)  
    pivot=pd.DataFrame(pivot).rename(columns={'cnt':prefixs+'Couponget_cnt'}).reset_index()  
    feature=pd.merge(feature,pivot,on=keys,how='left')  
    #不同优惠卷被核销数量  
    pivot=pd.pivot_table(data[data['label']==1],index=keys,values='cnt',aggfunc=len)  
    pivot=pd.DataFrame(pivot).rename(columns={'cnt':prefixs+'Couponuse_cnt'}).reset_index()  
    feature=pd.merge(feature,pivot,on=keys,how='left')
    #不同优惠卷未被核销数量  
    pivot=pd.pivot_table(data[data['label']==0],index=keys,values='cnt',aggfunc=len)  
    pivot=pd.DataFrame(pivot).rename(columns={'cnt':prefixs+'Coupon_nouse_cnt'}).reset_index()  
    feature=pd.merge(feature,pivot,on=keys,how='left')  
    #不同优惠卷的核销率  
    feature['Coupon_userate']=list(map(lambda x,y:x/y if y!=0 else 0,feature[prefixs+'Couponuse_cnt'],  
                                       feature[prefixs+'Couponget_cnt']))  
    #不同优惠卷在不同距离的核销数目  
    keys=['Coupon_id','Distance']  
    prefixs='history_field_'+'_'.join(keys)+'_'  
    pivot=pd.pivot_table(data[data['label']==1],index=keys,values='cnt',aggfunc=len)  
    pivot=pd.DataFrame(pivot).rename(columns={'cnt':prefixs+'DisandCouuse_cnt'}).reset_index()  
    feature=pd.merge(feature,pivot,on=keys,how='left')
    #不同优惠卷在不同距离的未核销数目 
    pivot=pd.pivot_table(data[data['label']==0],index=keys,values='cnt',aggfunc=len)  
    pivot=pd.DataFrame(pivot).rename(columns={'cnt':prefixs+'DisandCou_nouse_cnt'}).reset_index()  
    feature=pd.merge(feature,pivot,on=keys,how='left')
    #不同优惠卷在不同距离的领取数目  
    pivot=pd.pivot_table(data,index=keys,values='cnt',aggfunc=len)  
    pivot=pd.DataFrame(pivot).rename(columns={'cnt':prefixs+'DisandCouget_cnt'}).reset_index()  
    feature=pd.merge(feature,pivot,on=keys,how='left')  
    #不同优惠卷在不同距离的核销率  
    feature['DisandCou_rate']=list(map(lambda x,y:x/y if y!=0 else 0,feature[prefixs+'DisandCouuse_cnt'],  
                                       feature[prefixs+'DisandCouget_cnt']))
    #优惠卷被核销的时间平均间隔
    keys=['Coupon_id']
    prefixs='history_field_'+'_'.join(keys)+'_' 
    data['JianGe']=list(map(lambda x,y:(x-y).total_seconds()/60*60*24, data['date'],data['date_received']))
    pivot=pd.pivot_table(data[data['JianGe'].map(lambda x:str(x)!='nan')],index=keys,values='JianGe',aggfunc=np.mean)
    pivot=pd.DataFrame(pivot).rename(columns={'JianGe':prefixs+'UseJianGe_mean'}).reset_index()
    feature=pd.merge(feature,pivot,on=keys,how='left')
    return feature      

#历史商家特征
def get_history_merchant(history_field,label_field):
    data=history_field.copy()  
    data['cnt']=1
    feature=label_field.copy()
    #主键
    keys=['Merchant_id']
    prefixs='history_field_'+'_'.join(keys)+'_' 
    #商家优惠卷被领取数量
    pivot=pd.pivot_table(data,index=keys,values='cnt',aggfunc=len)
    pivot=pd.DataFrame(pivot).rename(columns={'cnt':prefixs+'Mer_cnt'}).reset_index()
    feature=pd.merge(feature,pivot,on=keys,how='left')
    #商家优惠卷未核销数量
    pivot=pd.pivot_table(data[data['label']==0],index=keys,values='cnt',aggfunc=len)
    pivot=pd.DataFrame(pivot).rename(columns={'cnt':prefixs+'N_hexiaoMer_cnt'}).reset_index()
    feature=pd.merge(feature,pivot,on=keys,how='left')
    #商家优惠卷核销数量
    pivot=pd.pivot_table(data[data['label']==1],index=keys,values='cnt',aggfunc=len)
    pivot=pd.DataFrame(pivot).rename(columns={'cnt':prefixs+'hexiaoMer_cnt'}).reset_index()
    feature=pd.merge(feature,pivot,on=keys,how='left')
    #商家优惠卷的核销率
    feature[prefixs+'Mer_HXrate']=list(map(lambda x,y:x/y if y!=0 else 0,feature[prefixs+'hexiaoMer_cnt'],
                                                 feature[prefixs+'Mer_cnt']))
    #商家优惠卷核销的平均折扣率
    pivot=pd.pivot_table(data[data['label']==1],index=keys,values='discount_rate',aggfunc=np.mean)
    pivot=pd.DataFrame(pivot).rename(columns={'discount_rate':prefixs+'DisRate_mean'}).reset_index()
    feature=pd.merge(feature,pivot,on=keys,how='left')
    #商家优惠卷被核销的平均距离
    data['Distance']=data['Distance'].map(lambda x: None if x==-1 else x) 
    pivot=pd.pivot_table(data[data['label']==1],index=keys,values='Distance',aggfunc=np.mean)
    pivot=pd.DataFrame(pivot).rename(columns={'Distance':prefixs+'Distance_mean'}).reset_index()
    feature=pd.merge(feature,pivot,on=keys,how='left')
    return feature

def get_dataset(history_field,mid_field,label_field):    
    simple_feature = get_simple_feature(label_field)    
    disandweek_feature = dis_week_feature(label_field)    
    coupon_feature = get_coupon_feature(label_field)    
    user_feature = get_user_feature(history_field,label_field)    
    hiscoupon_feature = get_history_coupon(history_field, label_field) 
    hismerchant_feature=get_history_merchant(history_field, label_field)
    #找出共有特征    
    share_character=list(set(simple_feature.columns.tolist())    
                         &set(disandweek_feature.columns.tolist())    
                         &set(coupon_feature.columns.tolist())    
                         &set(user_feature.columns.tolist())  
                         &set(hiscoupon_feature.columns.tolist())
                         &set(hismerchant_feature.columns.tolist()))    
    simple_feature.index=range(len(simple_feature))    
    disandweek_feature.index=range(len(disandweek_feature))    
    coupon_feature.index=range(len(coupon_feature))    
    user_feature.index=range(len(user_feature))   
    hiscoupon_feature.index=range(len(hiscoupon_feature)) 
    hismerchant_feature.index=range(len(hismerchant_feature))
    dataset=pd.concat([simple_feature,disandweek_feature.drop(share_character,axis=1),    
                       coupon_feature.drop(share_character,axis=1),    
                       user_feature.drop(share_character,axis=1),  
                       hiscoupon_feature.drop(share_character,axis=1),
                       hismerchant_feature.drop(share_character,axis=1)],axis=1)    
    #删除无用标签，label置于最后一列    
    if 'Date' in dataset.columns.tolist():    
        dataset.drop(['Merchant_id','Discount_rate','Date','date_received','date'],axis=1,inplace=True)    
        label=dataset['label'].tolist()    
        dataset.drop(['label'],axis=1,inplace=True)    
        dataset['label']=label    
    else:    
        dataset.drop(['Merchant_id','Discount_rate','date_received'],axis=1,inplace=True)    
    dataset['User_id']=dataset['User_id'].map(int)    
    dataset['Coupon_id']=dataset['Coupon_id'].map(int)    
    dataset['Date_received']=dataset['Date_received'].map(int)    
    dataset['Distance']=dataset['Distance'].map(int)    
    if 'label' in dataset.columns.tolist():    
        dataset['label']=dataset['label'].map(int)    
    return dataset    
  
def model_lgb(train, test):    
    params = {'num_leaves':2**5-1, 'reg_alpha':0.25, 'reg_lambda':0.25,'metric': 'auc','max_depth':-1, 'learning_rate':0.05, 'min_child_samples':5, 
              'random_state':7778,'min_data_in_leaf': 80,'min_sum_hessian_in_leaf': 10.0,'bagging_fraction': 0.7,'feature_fraction': 0.6,
              'num_threads': -1,'objective': 'binary','verbosity': -1,'n_estimators':750, 'subsample':0.9, 'colsample_bytree':0.7
             }
    target=train['label']
    X_train, X_test, y_train, y_test = train_test_split(train.drop(['User_id','Coupon_id','label','Date_received'],axis=1),
                                                        target, test_size=0.2,random_state=666666)
    dtrain = lgb.Dataset(X_train, y_train)
    dtest = lgb.Dataset(X_test, y_test, reference=dtrain)
    xtest=test.drop(['User_id','Coupon_id','Date_received'],axis=1).copy()
    model=lgb.train(params,dtrain,valid_sets=dtest)
    predict=model.predict(xtest)    
    predict=pd.DataFrame(predict,columns=['prob'])    
    result=pd.concat([test[['User_id','Coupon_id','Date_received']],predict],axis=1)      
    return result


#主函数   
#数据引入    
data_train=pd.read_csv('C:/Users/27Palos/Desktop/ccf_offline_stage1_train.csv')    
data_test=pd.read_csv('C:/Users/27Palos/Desktop/ccf_offline_stage1_test_revised.csv')    
offline=data_train.copy()    
offline=predo(offline)    
offtest=data_test.copy()    
offtest=predo(offtest)    
    
#数据划分    
#训练集的特征区间    
train_feature=offline[offline['date_received'].isin(pd.date_range('2016/3/16',periods=60))]    
#训练集中间区间    
train_mid=offline[offline['date'].isin(pd.date_range('2016/5/15',periods=15))]    
#训练集标记区间    
train_label=offline[offline['date_received'].isin(pd.date_range('2016/5/31',periods=31))]    
#测试集特征区间    
verify_feature=offline[offline['date_received'].isin(pd.date_range('2016/1/1',periods=60))]    
#测试集中间区间    
verify_mid=offline[offline['date'].isin(pd.date_range('2016/3/1',periods=15))]    
#测试集标记区间    
verify_label=offline[offline['date_received'].isin(pd.date_range('2016/3/16',periods=31))]    
#验证机特征区间    
test_feature=offline[offline['date_received'].isin(pd.date_range('2016/4/17',periods=60))]    
#验证集中间区间    
test_mid=offline[offline['date'].isin(pd.date_range('2016/6/16',periods=15))]    
#验证集标记区间    
test_label=offtest.copy()    
    
train=get_dataset(train_feature,train_mid,train_label)    
verify=get_dataset(verify_feature,verify_mid,verify_label)    
test=get_dataset(test_feature,test_mid,test_label)    
#调用模型    
big_train = pd.concat([train,verify], axis=0)    
result = model_lgb(big_train, test)
result.to_csv('C:/Users/27Palos/Desktop/Lgb.csv', index=False, header=None) 
