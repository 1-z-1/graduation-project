import pandas as pd
import matplotlib as plt
import os

os.chdir(r'C:/Users/24238/Desktop/毕业设计/毕业论文代码/')

# 数据截取
data = pd.read_excel('MetaBTCData.xlsx')
data = data[data['date'] >= '2020-10-01']
data.to_excel('CleanedData.xlsx', index=False)

# 为逻辑回归添加逻辑值
data = pd.read_excel('CleanedData.xlsx')
data.drop(0)
res = []
for i in range(data.index.size):
    if (data.at[i, 'OPEN'] >= data.at[i, 'CLOSE']):
        res.append(1)
    else:
        res.append(0)
data['RESULT'] = res
# 单位万元
data['OPEN'] = data['OPEN'] / 10000 * 6.3
data['CLOSE'] = data['CLOSE'] / 10000 * 6.3
data['MAX'] = data['MAX'] / 10000 * 6.3
data['MIN'] = data['MIN'] / 10000 * 6.3
data['VOLUME'] = data['VOLUME'] / 10000 * 6.3

# 准备对技术指标进行移位
data1 = data[
    ['MFI', 'ROC', 'TRIX', 'ADOSC', 'CCI', 'SLOWK', 'RSI', 'WR', 'MACD', 'FASTK', 'FASTD', 'SLOWD', 'VOLUME', 'MAX',
     'MIN']]  # 昨日能看到的
data2 = data[['OPEN', 'RESULT', 'date', 'CLOSE']]  # 今日能看到的(神经网络时需将CLOSE放下面)
data1 = data1.drop(data1.index[525]).reset_index(drop=True)
data2 = data2.drop(data1.index[0]).reset_index(drop=True)
data = data1.join(data2)
print(data)
data.to_excel('LogisticData.xlsx', index=False)
