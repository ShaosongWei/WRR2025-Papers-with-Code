import pandas as pd
import matplotlib.pyplot as plt
import pickle
import datetime
import matplotlib.gridspec as gridspec

# 读取CSV文件
actual_df_36 = pd.read_csv(r"D:\Users\weishaosong\PycharmProjects\pressure_prediction/result_dataframe/36_6_test_rescaled.csv")
predicted_df_36 = pd.read_csv(r"D:\Users\weishaosong\PycharmProjects\pressure_prediction/result_dataframe/36_6_pred_rescaled.csv")
noise_df_36 = pd.read_csv(r"D:\Users\weishaosong\PycharmProjects\pressure_prediction/result_dataframe/36_6_test_noise_rescaled.csv")
test_trend_df_36 = pd.read_csv(r"D:\Users\weishaosong\PycharmProjects\pressure_prediction/result_dataframe/36_6_test_trend_rescaled.csv")
test_resid_df_36 = pd.read_csv(r"D:\Users\weishaosong\PycharmProjects\pressure_prediction/result_dataframe/36_6_test_resid_rescaled.csv")
pred_trend_df_36 = pd.read_csv(r"D:\Users\weishaosong\PycharmProjects\pressure_prediction/result_dataframe/36_6_pred_trend_rescaled.csv")
pred_resid_df_36 = pd.read_csv(r"D:\Users\weishaosong\PycharmProjects\pressure_prediction/result_dataframe/36_6_pred_resid_rescaled.csv")
test_seasonal_df_36 = pd.read_csv(r"D:\Users\weishaosong\PycharmProjects\pressure_prediction/result_dataframe/36_6_test_seasonal_rescaled.csv")

with open('no_noise_filtered_0.6_data_44sensors_trend.pkl', 'rb') as file:
    trend_df = pickle.load(file)
trend_df = trend_df.iloc[-1692:]

sensor_name = "CLD0034"
actual_36_CLD0082 = actual_df_36[sensor_name].values
predicted_36_CLD0082 = predicted_df_36[sensor_name].values
noise_36_CLD0082 = noise_df_36[sensor_name].values
test_trend_36_CLD0082 = test_trend_df_36[sensor_name].values
test_resid_36_CLD0082 = test_resid_df_36[sensor_name].values
pred_trend_36_CLD0082 = pred_trend_df_36[sensor_name].values
pred_resid_36_CLD0082 = pred_resid_df_36[sensor_name].values
test_seasonal_36_CLD0082 = test_seasonal_df_36[sensor_name].values

# 绘制图表
start_sample = 100
end_sample = 600
# 设置全局字体和字号
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10  # 设置全局字体大小
# 创建图形对象
fig = plt.figure(figsize=(6.83, 6.83))
gs = gridspec.GridSpec(4, 1, figure=fig, height_ratios=[1, 1,1,1])
ax1 = fig.add_subplot(gs[0, :])  # 第一行，跨所有列
ax2 = fig.add_subplot(gs[1, :])  # 第二行第一个位置
ax3 = fig.add_subplot(gs[2, :])  # 第二行第二个位置
ax4 = fig.add_subplot(gs[3, :])  # 第二行第三个位置

# 为每条线加上 label 参数，但只在一个子图中调用 legend 方法
ax1.plot(trend_df.index[start_sample:end_sample], noise_36_CLD0082[start_sample:end_sample], marker='', linestyle='-',label='Measured')
ax1.plot(trend_df.index[start_sample:end_sample], actual_36_CLD0082[start_sample:end_sample], marker='', linestyle='-',label='Noise Removed')
ax1.set_ylabel('Nodal Pressure (m)')
ax1.text(-0.05, 1.14, '(a)', transform=ax1.transAxes, fontsize=10, fontweight='bold',verticalalignment='top')
ax1.set_xticks([datetime.datetime(2020, 4, 15, 0, 0, 0),datetime.datetime(2020, 4, 17, 0, 0, 0),datetime.datetime(2020, 4, 19, 0, 0, 0)])  # 设置x轴刻度
ax1.legend(loc='upper center', bbox_to_anchor=(0.22, 1.3),
           ncol=3, fancybox=True, frameon=False,framealpha=0.5)

ax2.plot(trend_df.index[start_sample:end_sample], test_trend_36_CLD0082[start_sample:end_sample], marker='', linestyle='-',alpha=0)
ax2.plot(trend_df.index[start_sample:end_sample], test_trend_36_CLD0082[start_sample:end_sample], marker='', linestyle='-',label='Trend')
ax2.text(-0.05, 1.2, '(b)', transform=ax2.transAxes, fontsize=10, fontweight='bold',verticalalignment='top')
ax2.text(0.02, 0.92, 'Trend', transform=ax2.transAxes, fontsize=10, verticalalignment='top')
ax2.set_yticks([22,24,26,28])  # 设置y轴刻度
ax2.set_xticks([datetime.datetime(2020, 4, 15, 0, 0, 0),datetime.datetime(2020, 4, 17, 0, 0, 0),datetime.datetime(2020, 4, 19, 0, 0, 0)])

ax3.plot(trend_df.index[start_sample:end_sample], test_seasonal_36_CLD0082[start_sample:end_sample], marker='', linestyle='-',alpha=0)
ax3.plot(trend_df.index[start_sample:end_sample], test_seasonal_36_CLD0082[start_sample:end_sample], marker='', linestyle='-',label='Seasonal')
ax3.text(-0.05, 1.14, '(c)', transform=ax3.transAxes, fontsize=10, fontweight='bold',verticalalignment='top')
ax3.text(0.02, 0.92, 'Seasonality', transform=ax3.transAxes, fontsize=10, verticalalignment='top')
ax3.set_yticks([-2,0,2])  # 设置y轴刻度
ax3.set_xticks([datetime.datetime(2020, 4, 15, 0, 0, 0),datetime.datetime(2020, 4, 17, 0, 0, 0),datetime.datetime(2020, 4, 19, 0, 0, 0)])

ax4.plot(trend_df.index[start_sample:end_sample], test_resid_36_CLD0082[start_sample:end_sample], marker='', linestyle='-',alpha=0)
ax4.plot(trend_df.index[start_sample:end_sample], test_resid_36_CLD0082[start_sample:end_sample], marker='', linestyle='-',label='Residual')
ax4.set_xlabel('Date-Time')
ax4.text(-0.05, 1.2, '(d)', transform=ax4.transAxes, fontsize=10, fontweight='bold',verticalalignment='top')
ax4.text(0.02, 0.92, 'Residuals', transform=ax4.transAxes, fontsize=10, verticalalignment='top')
ax4.set_yticks([-1,-0.2,0.6,1.4])  # 设置y轴刻度
ax4.set_xticks([datetime.datetime(2020, 4, 15, 0, 0, 0),datetime.datetime(2020, 4, 17, 0, 0, 0),datetime.datetime(2020, 4, 19, 0, 0, 0)])

fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.95, hspace=0.3)
# 保存图像到文件，设置分辨率为600dpi
# fig.savefig(r"D:\Users\weishaosong\PycharmProjects\pressure_prediction/result_plotting/figure_noise_removal_timeseries_decom_2sd_small_new.tiff", dpi=600, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
# 显示图形
plt.show()










