import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Input, Dense, Dropout, LayerNormalization
from keras.layers import LSTM, Flatten
from keras.models import Model
import plotly.graph_objects as go
import seaborn as sns
# 调用模型评价指标
# R2
from sklearn.metrics import r2_score
# MSE
from sklearn.metrics import mean_squared_error
# MAE
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random

# 生成随机种子37 32 31
seed = random.randint(0, 2**32 - 1)
########R2/MAE/RMSE 0.6229/0.0187/0.0294
# 保存种子到文件
seed_file_path = 'lstm_seed（323033）.txt'
with open(seed_file_path, 'w') as f:
    f.write(str(seed))

# # 设置随机种子
# random.seed(seed)
# np.random.seed(seed)
# tf.random.set_seed(seed)

# # 读取种子文件
# with open('lstm_seed（22818）.txt', 'r') as f:
#     seed = int(f.read().strip())

# 使用加载的种子设置随机数生成器
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 解决画图中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 输入的历史look_back步，和预测未来的T步，这里的look_back想用历史几天的数据就输入几天，T不用改
look_back = 5
T = 1

#读取训练集数据
train_df = pd.read_csv('../dataset/data4/b32_smoothed.csv', usecols=['IR', 'Tavg', 'Tmin', 'Tmax', 'chargetime', 'QD'])
train_X = train_df[['IR', 'Tavg', 'Tmin', 'Tmax', 'chargetime']].values
train_Y = train_df['QD'].values

#读取验证集数据
val_df = pd.read_csv('../dataset/data4/b30_smoothed.csv', usecols=['IR', 'Tavg', 'Tmin', 'Tmax', 'chargetime', 'QD'])
val_X = val_df[['IR', 'Tavg', 'Tmin', 'Tmax', 'chargetime']].values
val_Y = val_df['QD'].values

#读取测试集数据
test_df = pd.read_csv('../dataset/data4/b33_smoothed.csv', usecols=['IR', 'Tavg', 'Tmin', 'Tmax', 'chargetime', 'QD'])
test_X = test_df[['IR', 'Tavg', 'Tmin', 'Tmax', 'chargetime']].values
test_Y = test_df['QD'].values

cycle_train = pd.read_csv('../dataset/data4/b32_smoothed.csv', usecols=['cycle'])
cycle_val = pd.read_csv('../dataset/data4/b30_smoothed.csv', usecols=['cycle'])
cycle_test = pd.read_csv('../dataset/data4/b33_smoothed.csv', usecols=['cycle'])


#归一化数据
scaler1 = MinMaxScaler(feature_range=(0, 1))
scaler2 = MinMaxScaler(feature_range=(0, 1))
train_X = scaler1.fit_transform(train_X)
train_Y = scaler2.fit_transform(train_Y.reshape(-1, 1))
val_X = scaler1.fit_transform(val_X)
val_Y = scaler2.fit_transform(val_Y.reshape(-1, 1))
test_X = scaler1.transform(test_X)
test_Y = scaler2.transform(test_Y.reshape(-1, 1))

#定义输入数据，输出标签数据的格式的函数  ,同时将数据转换模型可接受的3D格式
def create_dataset(datasetX, datasetY, look_back=5, T=1):
    dataX, dataY = [], []
    for i in range(0, len(datasetX)-look_back-T, T):
        a = datasetX[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(datasetY[(i+look_back):(i+look_back+T)])
    return np.array(dataX), np.array(dataY)


# 准备训练集和测试集的数据
trainX, trainY = create_dataset(train_X, train_Y, look_back, T)
testX, testY = create_dataset(test_X, test_Y, look_back, T)
valX, valY = create_dataset(val_X, val_Y, look_back, T)

print(testX.shape)
print(testY.shape)
print(cycle_test.shape)

inputs = Input(shape=(look_back, train_X.shape[1]))
x = LSTM(64, return_sequences=True)(inputs)  # 减少LSTM神经元数量
x = Dropout(0.5)(x)  # 增加Dropout比例
x = LSTM(64)(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)  # 减少Dense层神经元数量
x = Dropout(0.5)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(T)(x)
model = Model(inputs=inputs, outputs=outputs)

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# 训练模型
history = model.fit(trainX, trainY, batch_size=32, epochs=200, validation_data=(valX, valY))

# 模型预测
test_predict = model.predict(testX)
val_predict = model.predict(valX)
#预测值和真实值
test_predict = test_predict.reshape(-1,1)
test_label = testY.reshape(-1,1)
val_predict = val_predict.reshape(-1,1)
val_label = valY.reshape(-1,1)
#测试集数据反归一化
test_predict = scaler2.inverse_transform(test_predict)
test_label = scaler2.inverse_transform(test_label)
val_predict = scaler2.inverse_transform(val_predict)
val_label = scaler2.inverse_transform(val_label)
#评估模型
# 计算模型的评价指标
test_R2 = r2_score(test_label, test_predict)
test_MAE = mean_absolute_error(test_label, test_predict)
test_RMSE = np.sqrt(mean_squared_error(test_label, test_predict))
test_MAPE = np.mean(np.abs((test_label, test_predict) / test_label))
test_MSE = mean_squared_error(test_label, test_predict)
# 打印模型的评价指标
print('Test Set Evaluation Metrics:')
print('R2:', test_R2)
print('MAE:', test_MAE)
print('RMSE:', test_RMSE)
print('MAPE:', test_MAPE)
print('MSE', test_MSE)
####
# Train set predictions
train_predict = model.predict(trainX)
train_predict = train_predict.reshape(-1, 1)
train_label = trainY.reshape(-1, 1)
train_predict = scaler2.inverse_transform(train_predict)
train_label = scaler2.inverse_transform(train_label)

# Calculate evaluation metrics for the train set
train_R2 = r2_score(train_label, train_predict)
train_MAE = mean_absolute_error(train_label, train_predict)
train_RMSE = np.sqrt(mean_squared_error(train_label, train_predict))
train_MAPE = np.mean(np.abs((train_label - train_predict) / train_label))

# Print the evaluation metrics for the train set
print('Train Set Evaluation Metrics:')
print('R2:', train_R2)
print('MAE:', train_MAE)
print('RMSE:', train_RMSE)
print('MAPE:', train_MAPE)

val_R2 = r2_score(val_label, val_predict)
val_MAE = mean_absolute_error(val_label, val_predict)
val_RMSE = np.sqrt(mean_squared_error(val_label, val_predict))
val_MAPE = np.mean(np.abs((val_label, val_predict) / val_label))

# Print the evaluation metrics for the train set
print('Val Set Evaluation Metrics:')
print('R2:', val_R2)
print('MAE:', val_MAE)
print('RMSE:', val_RMSE)
print('MAPE:', val_MAPE)

end_index = look_back + len(testX) * T
end_index1 = look_back + len(trainX)*T
end_index2 = look_back + len(valX)*T
train_cycle_adjusted = cycle_train['cycle'].values[look_back + T - 1:end_index1:T]
test_cycle_adjusted = cycle_test['cycle'].values[look_back + T - 1:end_index : 1]
# test_cycle_adjusted = cycle_test['cycle'].values[look_back - 1 : end_index : T]
val_cycle_adjusted = cycle_val['cycle'].values[look_back + T - 1:end_index2:T]

print(test_label.shape)
print(test_predict.shape)
print(test_cycle_adjusted.shape)

print(val_label.shape)
print(val_predict.shape)
print(val_cycle_adjusted.shape)

print('Length of cycle_test:', len(cycle_test['cycle']))
print('Length of testX:', len(testX))
print('Adjusted test_cycle_adjusted length:', len(test_cycle_adjusted))

sns.set_style("whitegrid")  # 设置网格背景
sns.set_palette("Set2")     # 设置调色板
plt.plot(test_cycle_adjusted, test_label.flatten(), label='Actual')
plt.plot(test_cycle_adjusted, test_predict.flatten(), label='test_Predicted')
plt.xlabel('Cycle', fontsize=14)
plt.ylabel('Discharge Capacity (QD)', fontsize=14)
plt.legend(fontsize=12, loc='upper right')
plt.show()


sns.set_style("whitegrid")  # 设置网格背景
sns.set_palette("Set2")     # 设置调色板
plt.plot(val_cycle_adjusted, val_label.flatten(), label='Actual')
plt.plot(val_cycle_adjusted, val_predict.flatten(), label='val_Predicted')
plt.xlabel('Cycle', fontsize=14)
plt.ylabel('Discharge Capacity (QD)', fontsize=14)
plt.legend(fontsize=12, loc='upper right')
plt.show()



###########
# 创建DataFrame
# train_result_df = pd.DataFrame({
#     'Cycle': train_cycle_adjusted,
#     'Actual QD': train_label.flatten(),
#     'Predicted QD': train_predict.flatten()
# })
#
# # 保存DataFrame到CSV文件
# train_result_df.to_csv('LSTM-traindata(71718).csv', index=False)

# 创建DataFrame
test_df = pd.DataFrame({
    'Cycle': test_cycle_adjusted,
    'Actual QD': test_label.flatten(),
    'Predicted QD': test_predict.flatten()
})

# 保存DataFrame到CSV文件
test_df.to_csv('LSTM-testdata(323033).csv', index=False)

# 创建DataFrame
val_df = pd.DataFrame({
    'Cycle': val_cycle_adjusted,
    'Actual QD': val_label.flatten(),
    'Predicted QD': val_predict.flatten()
})

# 保存DataFrame到CSV文件
val_df.to_csv('LSTM-valdata(323033).csv', index=False)


# # 读取CSV文件
# df = pd.read_csv('Transformer-valdata().csv')
# df['Prediction Error'] = df['Actual QD'] - df['Predicted QD']
#
# # 保存修改后的DataFrame到CSV文件
# df.to_csv("Transformer-valdata().csv", index=False)
# # 创建图表
# fig = plt.figure(figsize=(10, 8))
# gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
#
# # 创建上方的子图用于预测值和真实值曲线
# ax0 = plt.subplot(gs[0])
#
# # 绘制预测散点图
# ax0.scatter(df['Cycle'], df['Predicted QD'], color='blue', label='Predicted QD',linewidths=0.5)
#
# # 绘制真实折线图
# ax0.scatter(df['Cycle'], df['Actual QD'], color='red', label='True QD',linewidths=0.5)
#
# # 添加图例
# ax0.legend(loc='upper right')
# ax0.grid(True, linestyle='--', which='both', color='gray', linewidth=0.5)
# # 设置标签
# ax0.set_ylabel('QD')
# # 由于这个子图在上方，我们只设置X轴的刻度标签为不可见
# ax0.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
#
# # 创建下方的子图用于误差线图
# ax1 = plt.subplot(gs[1], sharex=ax0)
#
# # 绘制误差线图
# ax1.plot(df['Cycle'], df['Prediction Error'], color='black', label='Prediction Error')
#
# # 填充误差区域
# ax1.fill_between(df['Cycle'], df['Prediction Error'], color='gray', alpha=0.3)
#
# # 添加图例
# ax1.legend(loc='upper right')
#
# # 设置标签
# ax1.set_ylabel('Error')
# ax1.set_xlabel('Cycle')
# ax1.grid(True, linestyle='--', which='both', color='gray', linewidth=0.5)
# # 显示所有子图
# plt.show()