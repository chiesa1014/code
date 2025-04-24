import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 加载数据
file_path = 'lstm+transformer_index(val22817).csv'  # 请确保文件路径正确
data = pd.read_csv(file_path)

# # 计算每个样本在整个数据集所占的百分比，并添加一列位置索引记录这些百分比
# data['Position_Index'] = (data.index / len(data) * 100).round(4)

# 获取列
cycles = data['Cycle']
actual_qd = data['Actual QD']
lstm_pred = data['LSTM']
transformer_pred = data['Transformer']

# 计算预测值与真实值的差异
lstm_diff = lstm_pred - actual_qd
transformer_diff = transformer_pred - actual_qd

# 找到两个模型的预测值均高于真实值的区间
high_indices = np.where((lstm_diff > 0) & (transformer_diff > 0))[0]

# 找到两个模型的预测值均低于真实值的区间
low_indices = np.where((lstm_diff < 0) & (transformer_diff < 0))[0]

# 标记不同的区间
highlight = np.zeros(len(cycles), dtype=int)
highlight[high_indices] = 1
highlight[low_indices] = -1

# 定义一个函数来计算区间长度
def find_intervals(highlight, value):
    intervals = []
    start = None
    for i in range(len(highlight)):
        if highlight[i] == value and start is None:
            start = i
        elif highlight[i] != value and start is not None:
            intervals.append((start, i-1))
            start = None
    if start is not None:
        intervals.append((start, len(highlight)-1))
    return intervals

# 计算高于和低于真实值的区间
high_intervals = find_intervals(highlight, 1)
low_intervals = find_intervals(highlight, -1)

# 找到最长的两个区间
intervals = high_intervals + low_intervals
intervals = sorted(intervals, key=lambda x: x[1] - x[0], reverse=True)
longest_intervals = intervals[:2]

# 打印segment1和segment2各自的开始结束位置索引
print(f'Segment 1 Start: {longest_intervals[0][0]}, End: {longest_intervals[0][1]}')
print(f'Segment 2 Start: {longest_intervals[1][0]}, End: {longest_intervals[1][1]}')

# 分割数据集
segment1 = data.iloc[longest_intervals[0][0]:longest_intervals[0][1] + 1]
segment2 = data.iloc[longest_intervals[1][0]:longest_intervals[1][1] + 1]

# 拼接剩余未标记的部分，并按Cycle排序
remaining_indices = set(range(len(data))) - set(range(longest_intervals[0][0], longest_intervals[0][1] + 1)) - set(range(longest_intervals[1][0], longest_intervals[1][1] + 1))
remaining_data = data.iloc[list(remaining_indices)].sort_values(by='Cycle')

# # 保存分割后的数据集
# segment1.to_csv('segment1.csv', index=False)
# segment2.to_csv('segment2.csv', index=False)
# remaining_data.to_csv('remaining_data.csv', index=False)

# 绘制图形
plt.figure(figsize=(16, 10))
plt.plot(cycles, actual_qd, label='Actual QD', color='blue', linestyle='-',marker='o', markersize=20,linewidth=7, markevery=50)
plt.plot(cycles, lstm_pred, label='LSTM', color='orange', linestyle='--',marker='s',markersize=20,linewidth=7, markevery=50)
plt.plot(cycles, transformer_pred, label='Transformer', color='green', linestyle='-.',marker='^',markersize=20,linewidth=7, markevery=50)

# 高亮最长的两个区间并标记百分比和分割线
colors = ['red', 'purple']
for i, (start, end) in enumerate(longest_intervals):
    plt.fill_between(cycles[start:end+1], actual_qd.min(), actual_qd.max(), facecolor=colors[i], alpha=0.3)
    mid = (start + end) // 2
    text_y_position = actual_qd.min() - 0.05 * (actual_qd.max() - actual_qd.min())  # 调整文字位置到色块下方
    plt.text(cycles[mid], text_y_position, f'{(end - start + 1) / len(cycles) * 100:.2f}%', ha='center', va='bottom', fontsize=20, color=colors[i])

plt.xlabel('Cycle', fontsize=32, fontname='Times New Roman')
plt.ylabel('QD', fontsize=32, fontname='Times New Roman')
# plt.title('QD Prediction vs. Actual QD', fontsize=16)
plt.grid(True, alpha=0.3)  # 设置网格透明度
# 自定义网格线以更好地可视化
# plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.6)

# 调整坐标轴刻度的数字大小
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)

# 自定义主要和次要刻度
plt.minorticks_on()
# plt.tick_params(axis='x', which='major', length=10, width=1.5, labelsize=12)
plt.tick_params(axis='x', which='minor', length=5, width=1)
# plt.tick_params(axis='y', which='major', length=10, width=1.5, labelsize=12)
plt.tick_params(axis='y', which='minor', length=5, width=1)

# 添加图例
legend = plt.legend(loc='lower left', fontsize=32)
for text in legend.get_texts():
    text.set_fontname('Times New Roman')
# 调整布局以适应所有内容
plt.tight_layout()
plt.savefig('fig7a.pdf', format='pdf')
# 显示图形
plt.show()
