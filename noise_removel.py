import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

# 读取CSV文件
origin_pressure = pd.read_csv('pressureData.csv')
origin_pressure['time'] = pd.to_datetime(origin_pressure['time'])

# 将时间列转换为Datetime格式
origin_pressure['time'] = pd.to_datetime(origin_pressure['time'])
plot = origin_pressure.copy()           # 复制一份没有改变索引前的表格，用于画图，看去除异常值前后的效果
plot.set_index('time', inplace=True)

# 抽取日期和时间并添加到DataFrame中
origin_pressure['date'] = origin_pressure['time'].dt.date
origin_pressure['time_of_day'] = origin_pressure['time'].dt.time

# 将'time'列设置为索引
origin_pressure.set_index('time', inplace=True)
cols = list(origin_pressure.columns)
origin_pressure = origin_pressure[[cols[-2], cols[-1]] + cols[:-2]]
origin_pressure.set_index(['date', 'time_of_day'], inplace=True)       #设置双重索引，用于分组

# 定义异常值检测和插值函数
def detect_outliers_and_interpolate(series):
    mean = series.mean()
    std = series.std()
    threshold = 2 * std      # 将两倍标准差外的异常值去除作为监测值
    outliers = (series < mean - threshold) | (series > mean + threshold)

    if outliers.any():
        series[outliers] = np.nan
        series.interpolate(method='linear', inplace=True)

    return series

grouped = origin_pressure.groupby(level='time_of_day')
df_cleaned_interpolated = grouped.apply(lambda x: x.apply(detect_outliers_and_interpolate))
df_cleaned_interpolated = df_cleaned_interpolated.reset_index(level=0, drop=True)
df_reset = df_cleaned_interpolated.reset_index()

# 按照时间顺序排序，先按 date 排序，再按 time_of_day 排序
df_reset.sort_values(by=['date', 'time_of_day'], inplace=True)
df_reset['datetime'] = pd.to_datetime(df_reset['date'].astype(str) + ' ' + df_reset['time_of_day'].astype(str))
df_reset.set_index('datetime', inplace=True)
df_reset.drop(columns=['date', 'time_of_day'], inplace=True)

# 设计低通滤波器
def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# 设定采样频率和截止频率
fs = 4  # 采样频率为4 samples per hour
cutoff = 0.6 # 截止频率

# 创建一个新的 DataFrame 来存储滤波后的数据
filtered_data = pd.DataFrame(index=df_reset.index)

# 对 DataFrame 的每一列应用低通滤波
for column in df_reset.columns:
    filtered_data[column] = lowpass_filter(df_reset[column], cutoff, fs)

sensor1_original = plot.iloc[6000:7000]['CLD0082']
sensor1_cleaned = df_reset.iloc[6000:7000]['CLD0082']
sensor1_filtered = filtered_data.iloc[6000:7000]['CLD0082']

# 绘制去除前和去除后的Sensor数据
plt.figure(figsize=(14, 6))
plt.plot(sensor1_original.index, sensor1_original, label='Original Sensor1', color='blue')
plt.plot(sensor1_cleaned.index, sensor1_cleaned, label='Cleaned Sensor1', color='red')
plt.plot(sensor1_filtered.index, sensor1_filtered, label='Filtered Sensor1', color='green')
plt.title('Comparison of Sensor1 Data Before and After Removing Outliers')
plt.xlabel('Time')
plt.ylabel('Sensor1 Readings')
plt.legend()
plt.show()






