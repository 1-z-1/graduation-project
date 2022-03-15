import matplotlib as plt
import torch
import numpy as np
import pandas as pd
import torch.nn as nn


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


totData = pd.read_excel('LogisticData.xlsx')
xTest = torch.tensor(np.array(totData[470:525][['OPEN', 'MAX', 'MIN', 'VOLUME']]),
                     dtype=torch.float32)

model = torch.load('BpPrice.pt')
print(xTest[0])
for i in range(55):

    print(model(xTest[i]))
