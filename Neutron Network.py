import os
from torch import optim
import torch.nn as nn
import pandas as pd
import torch
import torch.utils.data as td
import numpy as np
import matplotlib.pyplot as plt

os.chdir('C:/Users/24238/Desktop/毕业设计/毕业论文代码')


# 构建模型
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear1 = nn.Linear(6, 512)  # 第一隐藏层
        self.Linear2 = nn.Linear(512, 128)
        self.Linear3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.sigmoid(self.Linear1(x))
        x = torch.relu(self.Linear2(x))
        x = self.Linear3(x)
        return x


if __name__ == '__main__':
    # 数据准备
    data = pd.read_excel('LogisticData.xlsx')
    fig, ax = plt.subplots(figsize=(3, 2), constrained_layout=True)

    xData = torch.tensor(np.array(data[['CCI', 'MACD', 'RSI', 'WR', 'TRIX', 'VOLUME']]),
                         dtype=torch.float32)
    yRes = torch.tensor(np.array(data['OPEN']), dtype=torch.float32)

    # test

    # 创建模型
    model = MyModel()

    # Loss函数
    lossFunc = nn.MSELoss()
    # 优化过程
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10000):
        yp = model(xData)
        # 计算loss
        loss = lossFunc(yp.squeeze(-1), yRes)
        # 计算梯度
        loss.backward()
        # 反向传播
        optimizer.step()
        # 梯度清0
        optimizer.zero_grad()
        if epoch % 100 == 0:
            print('epoch: {}, loss: {}'.format(epoch, loss.data.item()), 'accuracy:', )
    plt.plot(range(525), data['OPEN'])
    plt.plot(range(525), yp.detach().numpy(), color='red')
    plt.show()

"""
    counter = 0
    ypA=[]
    for x,y in zip(dataloaderx,dataloadery):
        # 正向传播
            yp = model(x)
            ypA.append(float(yp))
            # 计算loss
            loss = lossFunc(yp.squeeze(-1),y)
            # 计算梯度
            loss.backward()
    # 反向传播        
    optimizer.step()
    # 梯度清0
    optimizer.zero_grad()


    if epoch %2 == 0:
            print(ypA)
            print('epoch: {}, loss: {}'.format(epoch, loss.data.item()),'accuracy:','acc')
"""
