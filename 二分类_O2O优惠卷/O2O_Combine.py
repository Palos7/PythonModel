import pandas as pd

r1=pd.read_csv('C:/Users/27Palos/Desktop/Xgb.csv')
r2=pd.read_csv('C:/Users/27Palos/Desktop/Lgb.csv')

result=r1.copy()
#result.iloc[:,3]=list(map(lambda x,y:pow(x,0.23)*pow(y,0.77),r1.iloc[:,3],r2.iloc[:,3]))
result.iloc[:,3]=list(map(lambda x,y:x*0.225+y*0.775,r1.iloc[:,3],r2.iloc[:,3]))

result.to_csv('C:/Users/27Palos/Desktop/Result.csv', index=False, header=None) 
