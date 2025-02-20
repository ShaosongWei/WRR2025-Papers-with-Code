import pandas as pd
import matplotlib.pyplot as plt
import pickle
from statsmodels.tsa.seasonal import STL

with open('no_noise_filtered_0.6_data_44sensors.pkl', 'rb') as file:
    pressure_data = pickle.load(file)

# 定义一个函数，对单列数据进行STL分解
def stl_decomposition(ts):
    stl = STL(ts, period=96, robust=True)  # 你可以调整seasonal参数以适应你的数据
    result = stl.fit()
    return result.trend, result.seasonal, result.resid

# 创建空的 DataFrame 来存储分解结果
trend_df = pd.DataFrame(index=pressure_data.index)
seasonal_df = pd.DataFrame(index=pressure_data.index)
resid_df = pd.DataFrame(index=pressure_data.index)

# 遍历每一列，进行STL分解并将结果存储到相应的DataFrame中
for column in pressure_data.columns:
    trend, seasonal, resid = stl_decomposition(pressure_data[column])
    trend_df[column] = trend
    seasonal_df[column] = seasonal
    resid_df[column] = resid

# 保存 seasonal_df 和 resid_df 为 .pkl 文件
with open('no_noise_filtered_0.6_data_44sensors_seasonal.pkl', 'wb') as f:
    pickle.dump(seasonal_df, f)

with open('no_noise_filtered_0.6_data_44sensors_resid.pkl', 'wb') as f:
    pickle.dump(resid_df, f)

with open('no_noise_filtered_0.6_data_44sensors_trend.pkl', 'wb') as f:
    pickle.dump(trend_df, f)

non_null_count = trend_df['CLD0076'].notna().sum()
pd.set_option('display.max_rows', 1000)
first_column = resid_df.iloc[:500, 2]

plt.figure(figsize=(12, 6))
plt.plot(first_column, label=first_column.name)
plt.xlabel('Date')
plt.ylabel('Pressure')
plt.title('Pressure Data for ' + first_column.name)
plt.legend()
plt.show()









