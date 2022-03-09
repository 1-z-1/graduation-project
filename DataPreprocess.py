import pandas as pd
import matplotlib as plt
import os
os.chdir(r'C:/Users/24238/Desktop/毕业设计/毕业论文代码/')

#数据截取
data = pd.read_excel('MetaBTCData.xlsx')
data = data[data['date']>='2020-08-01']
data.to_excel('CleanedData.xlsx',index=False)

#为逻辑回归添加逻辑值
data = pd.read_excel('CleanedData.xlsx')
data.drop(0)
res = []
for i in range(data.index.size):
    if(data.at[i,'OPEN'] >= data.at[i,'CLOSE']):
        res.append(1)
    else:
        res.append(0)
data['RESULT'] = res
data.to_excel('LogisticData.xlsx',index=False)


