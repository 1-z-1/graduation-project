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


# 构建模型
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear1 = nn.Linear(16, 1024)  # 第一隐藏层
        self.Linear2 = nn.Linear(1024, 512)
        self.Linear3 = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.sigmoid(self.Linear1(x))
        x = torch.relu(self.Linear2(x))
        x = torch.sigmoid(self.Linear3(x))
        return x


if __name__ == '__main__':
    # 数据准备
    data = pd.read_excel('LogisticData.xlsx')

    xData = torch.tensor(np.array(data.drop(['VOLUME', 'RESULT','date'], axis=1)),
                         dtype=torch.float32)
    yRes = torch.tensor(np.array(data['RESULT']), dtype=torch.float32)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(xData, yRes, test_size=0.2, random_state=467898)

    # 创建模型
    model = MyModel()

    # Loss函数
    lossFunc = nn.BCELoss()
    # 优化过程
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(800):
        y_train_p = model(x_train)
        # 计算loss
        loss = lossFunc(y_train_p.squeeze(-1), y_train)
        # 计算梯度
        loss.backward()
        # 反向传播
        optimizer.step()
        # 梯度清0
        optimizer.zero_grad()
        if epoch % 100 == 0:
            y_test_p = model(x_test)
            threshold = 0.5
            y_train_p = y_train_p >= threshold
            y_test_p = y_test_p >= threshold
            acc_train = float((y_train_p.squeeze(-1) == y_train).sum()) / len(y_train_p)
            acc_test = float((y_test_p.squeeze(-1) == y_test).sum()) / len(y_test_p)
            print('epoch: {}, loss: {}'.format(epoch, loss.data.item()), 'accuracy_train:', acc_train, 'accuracy_test:',
                  acc_test)
    torch.save(model, 'BP.pt')
    plt.plot(range(525), data['OPEN'])
    plt.plot(range(420), y_train_p.detach().numpy(), color='red')
    plt.show()


    def plot_roc_curve(fper, tper):
        plt.plot(fper, tper, color='red', label='ROC')
        plt.plot([0, 1], [0, 1], color='green', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()


    fper, tper, thresholds = roc_curve(y_train, y_train_p.detach().numpy())
    plot_roc_curve(fper, tper)
