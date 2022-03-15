import tensorflow as tf
import pandas as pd
import os
from tensorflow import keras
from tensorflow.python.keras import layers, optimizers, Sequential
os.chdir('C:/Users/24238/Desktop/毕业设计/毕业论文代码')
data = pd.read_excel('LogisticData.xlsx')
train_dataset = data.sample(frac=0.8, random_state=0)
test_dataset = data.drop(train_dataset.index)
model = tf.keras.Model()
model.add(layers.Layer.)
