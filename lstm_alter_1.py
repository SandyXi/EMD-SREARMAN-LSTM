import PyEMD.EMD as EMD
import numpy  as np
import pylab as plt
import numpy
import matplotlib.pyplot as plt
# from _pytest import python
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from pyhht.emd import EMD
from pyhht.visualization import plot_imfs
# import PyEMD.visualisation as plot_imfs
from keras import regularizers

#单独提取出客流量那一列
flowdata=pd.read_csv('D:/Event/铁道预测深度学习项目/lstm-emd-spearman算法/两路口进站工作日.csv',header=None, usecols=[1])
flowdata.to_csv('D:/Event/铁道预测深度学习项目/lstm-emd-spearman算法/imf1.csv',index=None,columns=None,header=None)
#载入到时间序列数据
data = pd.read_csv('D:/Event/铁道预测深度学习项目/lstm-emd-spearman算法/imf1.csv',header=None)
#EMD经验模态分解
decomposer = EMD(data[0])
imfs = decomposer.decompose()
#绘制分解图
plot_imfs(data[0],imfs,data.index)
#保存IMFs
arr = np.vstack((imfs,data[0]))
dataframe = pd.DataFrame(arr.T)
dataframe.to_csv('D:/Event/铁道预测深度学习项目/lstm-emd-spearman算法/imf2.csv',index=None,columns=None,header=None)

# lstm


# load the dataset导入数据：
dataframe = read_csv('D:/Event/铁道预测深度学习项目/lstm-emd-spearman算法/两路口进站工作日.csv', usecols=[1], engine='python')
dataframe_1 = read_csv('D:/Event/铁道预测深度学习项目/lstm-emd-spearman算法/imf2.csv',engine='python')
dataset = dataframe.values.transpose()
dataset_1 = dataframe_1.values
dataset_1 = dataset_1.transpose()
shift_number = dataset.shape[1]
x1 = pd.Series(list(range(19)))
spearman_data,kendall_data = [],[]
for i in range(shift_number):
    y1 = pd.Series(dataset_1[:,i])
    r = x1.corr(y1,method='spearman')
    r1 = x1.corr(y1,method='kendall')
    spearman_data.append(r)
    kendall_data.append(r1)
# 综合两个参数
regnize_data = np.array([spearman_data,kendall_data])
shift_number = regnize_data.shape[1]
weights = []
for i in range(shift_number):
    data = regnize_data[:,i]
    min_data = min(data[0],data[1])
    if min_data > 0.2:
        weights.append(1)
    else:
        weights.append(0)
weights = np.asarray(weights).reshape(-1,1)
data_1 = []
for i in range(shift_number):
    a = dataset[0][i]*weights[i][0]
    data_1.append(a)
data_set = []
for i in data_1:
    if i != 0:
        data_set.append(i)
data_1 = np.asarray(data_set)
# weights = weights.transpose()
# dataset = np.dot(dataset,weights)
data_1 = np.array(data_1).reshape(-1,1)
dataset = data_1.astype('float32')
plt.figure('IMFs')
plt.plot(dataset)
# 需要把数据做一下转化:

# 将一列变成两列，第一列是 t 月的乘客数，第二列是 t+1 列的乘客数。
# look_back 就是预测下一步所需要的 time steps：
#
# timesteps 就是 LSTM 认为每个输入数据与前多少个陆续输入的数据有联系。例如具有这样用段序列数据 “…ABCDBCEDF…”，当 timesteps 为 3 时，在模型预测中如果输入数据为“D”，那么之前接收的数据如果为“B”和“C”则此时的预测输出为 B 的概率更大，之前接收的数据如果为“C”和“E”，则此时的预测输出为 F 的概率更大。

# X is the number of passengers at a given time (t) and Y is the number of passengers at the next time (t + 1).

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

# fix random seed for reproducibility
numpy.random.seed(7)

# 当激活函数为 sigmoid 或者 tanh 时，要把数据正则话，此时 LSTM 比较敏感
# 设定 67% 是训练数据，余下的是测试数据

# normalize the dataset
# 归一成0-1之间的数
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)


# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# X=t and Y=t+1 时的数据，并且此时的维度为 [samples, features]
# use this function to prepare the train and test datasets for modeling
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# 投入到 LSTM 的 X 需要有这样的结构： [samples, time steps, features]，所以做一下变换
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# 建立 LSTM 模型：
# 输入层有 1 个input，隐藏层有 4 个神经元，输出层就是预测一个值，激活函数用 sigmoid，迭代 30 次，batch size 为 1
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=30, batch_size=1, verbose=2)

# 预测
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# 计算误差之前要先把预测数据转换成同一单位
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# 计算 mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)  # 给定数组(a)的形状和类型返回一个新的空数组
trainPredictPlot[:, :] = numpy.nan   # 全部填充为nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# 画出结果：蓝色为原数据，绿色为训练集的预测值，红色为测试集的预测值
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions
plt.figure('flow')
plt.plot(scaler.inverse_transform(dataset))  # 将降维后的数据转换成原始数据
# X=scaler.inverse_transform(X[, copy])
# 将标准化后的数据转换为原始数据。
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()