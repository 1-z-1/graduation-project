import os
from torch import optim
import torch.nn as nn
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import roc_curve

os.chdir('C:/Users/24238/Desktop/毕业设计/毕业论文代码')


# 构建与之前类似的模型
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear1 = nn.Linear(4, 512, bias=True)  # 第一隐藏层
        self.Linear2 = nn.Linear(512, 256, bias=True)
        self.Linear3 = nn.Linear(256, 1, bias=True)

    def forward(self, x):
        x = torch.relu(self.Linear1(x))
        x = torch.sigmoid(self.Linear2(x))
        x = self.Linear3(x)
        return x


if __name__ == '__main__':
    totData = pd.read_excel('LogisticData.xlsx')
    xTrain = torch.tensor(np.array(totData[0:470][['OPEN', 'MAX', 'MIN', 'VOLUME']]),  # 今日OPEN，昨日MAX MIN VOLUME
                          dtype=torch.float32)
    yTrain = torch.tensor(np.array(totData[0:470]['CLOSE']), dtype=torch.float32)
    xTest = torch.tensor(np.array(totData[470:525][['OPEN', 'MAX', 'MIN', 'VOLUME']]),
                         dtype=torch.float32)
    yTest = torch.tensor(np.array(totData[470:525]['CLOSE']), dtype=torch.float32)

    model = MyModel()
    # MSE作为Loss
    lossFunc = nn.MSELoss()
    # 定义优化下降方法为Adam
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(5000):
        yPre_train = model(xTrain)
        loss = lossFunc(yPre_train.squeeze(-1), yTrain)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if epoch % 100 == 0:
            print('epoch: {}, loss: {}'.format(epoch, loss.data.item()))
    torch.save(model, 'BpPrice.pt')
    yPre_test = model(xTest)
    yPre = torch.cat((yPre_train, yPre_test), 0)
    plt.plot(range(525), totData['CLOSE'])
    plt.plot(range(470), yPre_train.detach().numpy(), color='red')
    plt.plot(range(470, 525), yPre_test.detach().numpy(), color='green')
    plt.show()


