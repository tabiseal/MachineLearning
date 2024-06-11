import dataset
import numpy as np
import plot_utils
from keras.models import Sequential
from keras.layers import Dense

m=100
X,Y=dataset.get_beans1(m)
plot_utils.show_scatter(X,Y)

model=Sequential()
#units:神经元个数
#activation:激活函数的类型
#inupt_dim:输入数据维度（豆豆的大小）
dense=Dense(units=1,activation='sigmoid',input_dim=1)
model.add(dense)
#我们使用均方误差代价函数
#loss:损失函数（代价函数）
#optimizer:优化器（sgd:随机梯度下降算法）
#metrics:评估标准（准确度）
model.compile(loss='mean_squared_error',optimizer='sgd',metrics=['accuracy'])
#epochs:回合数（全部样本完成一次训练称为一个回合）
#batch_size:批数量（一次训练多少个样本）
model.fit(X,Y,epochs=5000,batch_size=10)

pres=model.predict(X)

plot_utils.show_scatter_curve(X,Y,pres)