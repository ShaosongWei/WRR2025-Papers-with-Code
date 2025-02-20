import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# 读取CSV文件
predicted_df_21 = pd.read_csv(r"D:\Users\weishaosong\PycharmProjects\pressure_prediction/result_dataframe/21_2_pred_rescaled.csv")
noise_df_21 = pd.read_csv(r"D:\Users\weishaosong\PycharmProjects\pressure_prediction/result_dataframe/21_2_test_noise_rescaled.csv")
predicted_df_24 = pd.read_csv(r"D:\Users\weishaosong\PycharmProjects\pressure_prediction/result_dataframe/24_3_pred_rescaled.csv")
noise_df_24 = pd.read_csv(r"D:\Users\weishaosong\PycharmProjects\pressure_prediction/result_dataframe/24_3_test_noise_rescaled.csv")
predicted_df_30 = pd.read_csv(r"D:\Users\weishaosong\PycharmProjects\pressure_prediction/result_dataframe/30_4_pred_rescaled.csv")
noise_df_30 = pd.read_csv(r"D:\Users\weishaosong\PycharmProjects\pressure_prediction/result_dataframe/30_4_test_noise_rescaled.csv")
predicted_df_33 = pd.read_csv(r"D:\Users\weishaosong\PycharmProjects\pressure_prediction/result_dataframe/33_5_pred_rescaled.csv")
noise_df_33 = pd.read_csv(r"D:\Users\weishaosong\PycharmProjects\pressure_prediction/result_dataframe/33_5_test_noise_rescaled.csv")
predicted_df_36 = pd.read_csv(r"D:\Users\weishaosong\PycharmProjects\pressure_prediction/result_dataframe/36_6_pred_rescaled.csv")
noise_df_36 = pd.read_csv(r"D:\Users\weishaosong\PycharmProjects\pressure_prediction/result_dataframe/36_6_test_noise_rescaled.csv")
predicted_df_54 = pd.read_csv(r"D:\Users\weishaosong\PycharmProjects\pressure_prediction/result_dataframe/54_12_pred_rescaled.csv")
noise_df_54 = pd.read_csv(r"D:\Users\weishaosong\PycharmProjects\pressure_prediction/result_dataframe/54_12_test_noise_rescaled.csv")
predicted_df_78 = pd.read_csv(r"D:\Users\weishaosong\PycharmProjects\pressure_prediction/result_dataframe/78_24_pred_rescaled.csv")
noise_df_78 = pd.read_csv(r"D:\Users\weishaosong\PycharmProjects\pressure_prediction/result_dataframe/78_24_test_noise_rescaled.csv")
predicted_df_108 = pd.read_csv(r"D:\Users\weishaosong\PycharmProjects\pressure_prediction/result_dataframe/108_48_pred_rescaled.csv")
noise_df_108 = pd.read_csv(r"D:\Users\weishaosong\PycharmProjects\pressure_prediction/result_dataframe/108_48_test_noise_rescaled.csv")
predicted_df_216 = pd.read_csv(r"D:\Users\weishaosong\PycharmProjects\pressure_prediction/result_dataframe/216_96_pred_rescaled.csv")
noise_df_216 = pd.read_csv(r"D:\Users\weishaosong\PycharmProjects\pressure_prediction/result_dataframe/216_96_test_noise_rescaled.csv")

# 误差大于阈值，则使用第一个列中的值，否则使用第二个列中的值
def choose_value_based_on_error(df1, df2, error_limit):
    result_df = pd.DataFrame(index=df1.index, columns=df1.columns)
    for col in df1.columns:
        error = abs(df1[col] - df2[col])
        result_df[col] = df1[col].where(error > error_limit, df2[col])

    return result_df

# 使用上面定义的函数处理DataFrame
transmitted_15_1 = choose_value_based_on_error(noise_df_15, predicted_df_15,0.1)
transmitted_15_2 = choose_value_based_on_error(noise_df_15, predicted_df_15,0.2)
transmitted_15_3 = choose_value_based_on_error(noise_df_15, predicted_df_15,0.3)
transmitted_15_4 = choose_value_based_on_error(noise_df_15, predicted_df_15,0.4)
transmitted_15_5 = choose_value_based_on_error(noise_df_15, predicted_df_15,0.5)

transmitted_21_1 = choose_value_based_on_error(noise_df_21, predicted_df_21,0.1)
transmitted_21_2 = choose_value_based_on_error(noise_df_21, predicted_df_21,0.2)
transmitted_21_3 = choose_value_based_on_error(noise_df_21, predicted_df_21,0.3)
transmitted_21_4 = choose_value_based_on_error(noise_df_21, predicted_df_21,0.4)
transmitted_21_5 = choose_value_based_on_error(noise_df_21, predicted_df_21,0.5)

transmitted_24_1 = choose_value_based_on_error(noise_df_24, predicted_df_24,0.1)
transmitted_24_2 = choose_value_based_on_error(noise_df_24, predicted_df_24,0.2)
transmitted_24_3 = choose_value_based_on_error(noise_df_24, predicted_df_24,0.3)
transmitted_24_4 = choose_value_based_on_error(noise_df_24, predicted_df_24,0.4)
transmitted_24_5 = choose_value_based_on_error(noise_df_24, predicted_df_24,0.5)

transmitted_30_1 = choose_value_based_on_error(noise_df_30, predicted_df_30,0.1)
transmitted_30_2 = choose_value_based_on_error(noise_df_30, predicted_df_30,0.2)
transmitted_30_3 = choose_value_based_on_error(noise_df_30, predicted_df_30,0.3)
transmitted_30_4 = choose_value_based_on_error(noise_df_30, predicted_df_30,0.4)
transmitted_30_5 = choose_value_based_on_error(noise_df_30, predicted_df_30,0.5)

transmitted_33_1 = choose_value_based_on_error(noise_df_33, predicted_df_33,0.1)
transmitted_33_2 = choose_value_based_on_error(noise_df_33, predicted_df_33,0.2)
transmitted_33_3 = choose_value_based_on_error(noise_df_33, predicted_df_33,0.3)
transmitted_33_4 = choose_value_based_on_error(noise_df_33, predicted_df_33,0.4)
transmitted_33_5 = choose_value_based_on_error(noise_df_33, predicted_df_33,0.5)

transmitted_36_1 = choose_value_based_on_error(noise_df_36, predicted_df_36,0.1)
transmitted_36_2 = choose_value_based_on_error(noise_df_36, predicted_df_36,0.2)
transmitted_36_3 = choose_value_based_on_error(noise_df_36, predicted_df_36,0.3)
transmitted_36_4 = choose_value_based_on_error(noise_df_36, predicted_df_36,0.4)
transmitted_36_5 = choose_value_based_on_error(noise_df_36, predicted_df_36,0.5)

transmitted_54_1 = choose_value_based_on_error(noise_df_54, predicted_df_54,0.1)
transmitted_54_2 = choose_value_based_on_error(noise_df_54, predicted_df_54,0.2)
transmitted_54_3 = choose_value_based_on_error(noise_df_54, predicted_df_54,0.3)
transmitted_54_4 = choose_value_based_on_error(noise_df_54, predicted_df_54,0.4)
transmitted_54_5 = choose_value_based_on_error(noise_df_54, predicted_df_54,0.5)

transmitted_78_1 = choose_value_based_on_error(noise_df_78, predicted_df_78,0.1)
transmitted_78_2 = choose_value_based_on_error(noise_df_78, predicted_df_78,0.2)
transmitted_78_3 = choose_value_based_on_error(noise_df_78, predicted_df_78,0.3)
transmitted_78_4 = choose_value_based_on_error(noise_df_78, predicted_df_78,0.4)
transmitted_78_5 = choose_value_based_on_error(noise_df_78, predicted_df_78,0.5)

transmitted_108_1 = choose_value_based_on_error(noise_df_108, predicted_df_108,0.1)
transmitted_108_2 = choose_value_based_on_error(noise_df_108, predicted_df_108,0.2)
transmitted_108_3 = choose_value_based_on_error(noise_df_108, predicted_df_108,0.3)
transmitted_108_4 = choose_value_based_on_error(noise_df_108, predicted_df_108,0.4)
transmitted_108_5 = choose_value_based_on_error(noise_df_108, predicted_df_108,0.5)

transmitted_216_1 = choose_value_based_on_error(noise_df_216, predicted_df_216,0.1)
transmitted_216_2 = choose_value_based_on_error(noise_df_216, predicted_df_216,0.2)
transmitted_216_3 = choose_value_based_on_error(noise_df_216, predicted_df_216,0.3)
transmitted_216_4 = choose_value_based_on_error(noise_df_216, predicted_df_216,0.4)
transmitted_216_5 = choose_value_based_on_error(noise_df_216, predicted_df_216,0.5)

# 计算RMSE
def calculate_rmse(actual, predicted):
    return np.sqrt(((actual - predicted) ** 2).mean())

def calculate_mae(actual, predicted):
    return mean_absolute_error(actual, predicted)

def calculate_r2(actual, predicted):
    return r2_score(actual, predicted)

rmse_series_15_1 = pd.Series({col: calculate_rmse(noise_df_15[col], transmitted_15_1[col]) for col in noise_df_15.columns})
rmse_series_15_2 = pd.Series({col: calculate_rmse(noise_df_15[col], transmitted_15_2[col]) for col in noise_df_15.columns})
rmse_series_15_3 = pd.Series({col: calculate_rmse(noise_df_15[col], transmitted_15_3[col]) for col in noise_df_15.columns})
rmse_series_15_4 = pd.Series({col: calculate_rmse(noise_df_15[col], transmitted_15_4[col]) for col in noise_df_15.columns})
rmse_series_15_5 = pd.Series({col: calculate_rmse(noise_df_15[col], transmitted_15_5[col]) for col in noise_df_15.columns})

rmse_series_21_1 = pd.Series({col: calculate_rmse(noise_df_21[col], transmitted_21_1[col]) for col in noise_df_21.columns})
rmse_series_21_2 = pd.Series({col: calculate_rmse(noise_df_21[col], transmitted_21_2[col]) for col in noise_df_21.columns})
rmse_series_21_3 = pd.Series({col: calculate_rmse(noise_df_21[col], transmitted_21_3[col]) for col in noise_df_21.columns})
rmse_series_21_4 = pd.Series({col: calculate_rmse(noise_df_21[col], transmitted_21_4[col]) for col in noise_df_21.columns})
rmse_series_21_5 = pd.Series({col: calculate_rmse(noise_df_21[col], transmitted_21_5[col]) for col in noise_df_21.columns})

rmse_series_24_1 = pd.Series({col: calculate_rmse(noise_df_24[col], transmitted_24_1[col]) for col in noise_df_24.columns})
rmse_series_24_2 = pd.Series({col: calculate_rmse(noise_df_24[col], transmitted_24_2[col]) for col in noise_df_24.columns})
rmse_series_24_3 = pd.Series({col: calculate_rmse(noise_df_24[col], transmitted_24_3[col]) for col in noise_df_24.columns})
rmse_series_24_4 = pd.Series({col: calculate_rmse(noise_df_24[col], transmitted_24_4[col]) for col in noise_df_24.columns})
rmse_series_24_5 = pd.Series({col: calculate_rmse(noise_df_24[col], transmitted_24_5[col]) for col in noise_df_24.columns})

rmse_series_30_1 = pd.Series({col: calculate_rmse(noise_df_30[col], transmitted_30_1[col]) for col in noise_df_30.columns})
rmse_series_30_2 = pd.Series({col: calculate_rmse(noise_df_30[col], transmitted_30_2[col]) for col in noise_df_30.columns})
rmse_series_30_3 = pd.Series({col: calculate_rmse(noise_df_30[col], transmitted_30_3[col]) for col in noise_df_30.columns})
rmse_series_30_4 = pd.Series({col: calculate_rmse(noise_df_30[col], transmitted_30_4[col]) for col in noise_df_30.columns})
rmse_series_30_5 = pd.Series({col: calculate_rmse(noise_df_30[col], transmitted_30_5[col]) for col in noise_df_30.columns})

rmse_series_33_1 = pd.Series({col: calculate_rmse(noise_df_33[col], transmitted_33_1[col]) for col in noise_df_33.columns})
rmse_series_33_2 = pd.Series({col: calculate_rmse(noise_df_33[col], transmitted_33_2[col]) for col in noise_df_33.columns})
rmse_series_33_3 = pd.Series({col: calculate_rmse(noise_df_33[col], transmitted_33_3[col]) for col in noise_df_33.columns})
rmse_series_33_4 = pd.Series({col: calculate_rmse(noise_df_33[col], transmitted_33_4[col]) for col in noise_df_33.columns})
rmse_series_33_5 = pd.Series({col: calculate_rmse(noise_df_33[col], transmitted_33_5[col]) for col in noise_df_33.columns})

rmse_series_36_1 = pd.Series({col: calculate_rmse(noise_df_36[col], transmitted_36_1[col]) for col in noise_df_36.columns})
rmse_series_36_2 = pd.Series({col: calculate_rmse(noise_df_36[col], transmitted_36_2[col]) for col in noise_df_36.columns})
rmse_series_36_3 = pd.Series({col: calculate_rmse(noise_df_36[col], transmitted_36_3[col]) for col in noise_df_36.columns})
rmse_series_36_4 = pd.Series({col: calculate_rmse(noise_df_36[col], transmitted_36_4[col]) for col in noise_df_36.columns})
rmse_series_36_5 = pd.Series({col: calculate_rmse(noise_df_36[col], transmitted_36_5[col]) for col in noise_df_36.columns})

rmse_series_54_1 = pd.Series({col: calculate_rmse(noise_df_54[col], transmitted_54_1[col]) for col in noise_df_54.columns})
rmse_series_54_2 = pd.Series({col: calculate_rmse(noise_df_54[col], transmitted_54_2[col]) for col in noise_df_54.columns})
rmse_series_54_3 = pd.Series({col: calculate_rmse(noise_df_54[col], transmitted_54_3[col]) for col in noise_df_54.columns})
rmse_series_54_4 = pd.Series({col: calculate_rmse(noise_df_54[col], transmitted_54_4[col]) for col in noise_df_54.columns})
rmse_series_54_5 = pd.Series({col: calculate_rmse(noise_df_54[col], transmitted_54_5[col]) for col in noise_df_54.columns})

rmse_series_78_1 = pd.Series({col: calculate_rmse(noise_df_78[col], transmitted_78_1[col]) for col in noise_df_78.columns})
rmse_series_78_2 = pd.Series({col: calculate_rmse(noise_df_78[col], transmitted_78_2[col]) for col in noise_df_78.columns})
rmse_series_78_3 = pd.Series({col: calculate_rmse(noise_df_78[col], transmitted_78_3[col]) for col in noise_df_78.columns})
rmse_series_78_4 = pd.Series({col: calculate_rmse(noise_df_78[col], transmitted_78_4[col]) for col in noise_df_78.columns})
rmse_series_78_5 = pd.Series({col: calculate_rmse(noise_df_78[col], transmitted_78_5[col]) for col in noise_df_78.columns})

rmse_series_108_1 = pd.Series({col: calculate_rmse(noise_df_108[col], transmitted_108_1[col]) for col in noise_df_108.columns})
rmse_series_108_2 = pd.Series({col: calculate_rmse(noise_df_108[col], transmitted_108_2[col]) for col in noise_df_108.columns})
rmse_series_108_3 = pd.Series({col: calculate_rmse(noise_df_108[col], transmitted_108_3[col]) for col in noise_df_108.columns})
rmse_series_108_4 = pd.Series({col: calculate_rmse(noise_df_108[col], transmitted_108_4[col]) for col in noise_df_108.columns})
rmse_series_108_5 = pd.Series({col: calculate_rmse(noise_df_108[col], transmitted_108_5[col]) for col in noise_df_108.columns})

rmse_series_216_1 = pd.Series({col: calculate_rmse(noise_df_216[col], transmitted_216_1[col]) for col in noise_df_216.columns})
rmse_series_216_2 = pd.Series({col: calculate_rmse(noise_df_216[col], transmitted_216_2[col]) for col in noise_df_216.columns})
rmse_series_216_3 = pd.Series({col: calculate_rmse(noise_df_216[col], transmitted_216_3[col]) for col in noise_df_216.columns})
rmse_series_216_4 = pd.Series({col: calculate_rmse(noise_df_216[col], transmitted_216_4[col]) for col in noise_df_216.columns})
rmse_series_216_5 = pd.Series({col: calculate_rmse(noise_df_216[col], transmitted_216_5[col]) for col in noise_df_216.columns})

# 绘制累计概率分布图
rmse_values_15_1 = rmse_series_15_1.values
rmse_values_15_2 = rmse_series_15_2.values
rmse_values_15_3 = rmse_series_15_3.values
rmse_values_15_4 = rmse_series_15_4.values
rmse_values_15_5 = rmse_series_15_5.values

rmse_values_21_1 = rmse_series_21_1.values
rmse_values_21_2 = rmse_series_21_2.values
rmse_values_21_3 = rmse_series_21_3.values
rmse_values_21_4 = rmse_series_21_4.values
rmse_values_21_5 = rmse_series_21_5.values

rmse_values_24_1 = rmse_series_24_1.values
rmse_values_24_2 = rmse_series_24_2.values
rmse_values_24_3 = rmse_series_24_3.values
rmse_values_24_4 = rmse_series_24_4.values
rmse_values_24_5 = rmse_series_24_5.values

rmse_values_30_1 = rmse_series_30_1.values
rmse_values_30_2 = rmse_series_30_2.values
rmse_values_30_3 = rmse_series_30_3.values
rmse_values_30_4 = rmse_series_30_4.values
rmse_values_30_5 = rmse_series_30_5.values

rmse_values_33_1 = rmse_series_33_1.values
rmse_values_33_2 = rmse_series_33_2.values
rmse_values_33_3 = rmse_series_33_3.values
rmse_values_33_4 = rmse_series_33_4.values
rmse_values_33_5 = rmse_series_33_5.values

rmse_values_36_1 = rmse_series_36_1.values
rmse_values_36_2 = rmse_series_36_2.values
rmse_values_36_3 = rmse_series_36_3.values
rmse_values_36_4 = rmse_series_36_4.values
rmse_values_36_5 = rmse_series_36_5.values

rmse_values_54_1 = rmse_series_54_1.values
rmse_values_54_2 = rmse_series_54_2.values
rmse_values_54_3 = rmse_series_54_3.values
rmse_values_54_4 = rmse_series_54_4.values
rmse_values_54_5 = rmse_series_54_5.values

rmse_values_78_1 = rmse_series_78_1.values
rmse_values_78_2 = rmse_series_78_2.values
rmse_values_78_3 = rmse_series_78_3.values
rmse_values_78_4 = rmse_series_78_4.values
rmse_values_78_5 = rmse_series_78_5.values

rmse_values_108_1 = rmse_series_108_1.values
rmse_values_108_2 = rmse_series_108_2.values
rmse_values_108_3 = rmse_series_108_3.values
rmse_values_108_4 = rmse_series_108_4.values
rmse_values_108_5 = rmse_series_108_5.values

rmse_values_216_1 = rmse_series_216_1.values
rmse_values_216_2 = rmse_series_216_2.values
rmse_values_216_3 = rmse_series_216_3.values
rmse_values_216_4 = rmse_series_216_4.values
rmse_values_216_5 = rmse_series_216_5.values

sorted_rmse_15_1 = np.sort(rmse_values_15_1)
sorted_rmse_15_2 = np.sort(rmse_values_15_2)
sorted_rmse_15_3 = np.sort(rmse_values_15_3)
sorted_rmse_15_4 = np.sort(rmse_values_15_4)
sorted_rmse_15_5 = np.sort(rmse_values_15_5)

sorted_rmse_21_1 = np.sort(rmse_values_21_1)
sorted_rmse_21_2 = np.sort(rmse_values_21_2)
sorted_rmse_21_3 = np.sort(rmse_values_21_3)
sorted_rmse_21_4 = np.sort(rmse_values_21_4)
sorted_rmse_21_5 = np.sort(rmse_values_21_5)

sorted_rmse_24_1 = np.sort(rmse_values_24_1)
sorted_rmse_24_2 = np.sort(rmse_values_24_2)
sorted_rmse_24_3 = np.sort(rmse_values_24_3)
sorted_rmse_24_4 = np.sort(rmse_values_24_4)
sorted_rmse_24_5 = np.sort(rmse_values_24_5)

sorted_rmse_30_1 = np.sort(rmse_values_30_1)
sorted_rmse_30_2 = np.sort(rmse_values_30_2)
sorted_rmse_30_3 = np.sort(rmse_values_30_3)
sorted_rmse_30_4 = np.sort(rmse_values_30_4)
sorted_rmse_30_5 = np.sort(rmse_values_30_5)

sorted_rmse_33_1 = np.sort(rmse_values_33_1)
sorted_rmse_33_2 = np.sort(rmse_values_33_2)
sorted_rmse_33_3 = np.sort(rmse_values_33_3)
sorted_rmse_33_4 = np.sort(rmse_values_33_4)
sorted_rmse_33_5 = np.sort(rmse_values_33_5)

sorted_rmse_36_1 = np.sort(rmse_values_36_1)
sorted_rmse_36_2 = np.sort(rmse_values_36_2)
sorted_rmse_36_3 = np.sort(rmse_values_36_3)
sorted_rmse_36_4 = np.sort(rmse_values_36_4)
sorted_rmse_36_5 = np.sort(rmse_values_36_5)

sorted_rmse_54_1 = np.sort(rmse_values_54_1)
sorted_rmse_54_2 = np.sort(rmse_values_54_2)
sorted_rmse_54_3 = np.sort(rmse_values_54_3)
sorted_rmse_54_4 = np.sort(rmse_values_54_4)
sorted_rmse_54_5 = np.sort(rmse_values_54_5)

sorted_rmse_78_1 = np.sort(rmse_values_78_1)
sorted_rmse_78_2 = np.sort(rmse_values_78_2)
sorted_rmse_78_3 = np.sort(rmse_values_78_3)
sorted_rmse_78_4 = np.sort(rmse_values_78_4)
sorted_rmse_78_5 = np.sort(rmse_values_78_5)

sorted_rmse_108_1 = np.sort(rmse_values_108_1)
sorted_rmse_108_2 = np.sort(rmse_values_108_2)
sorted_rmse_108_3 = np.sort(rmse_values_108_3)
sorted_rmse_108_4 = np.sort(rmse_values_108_4)
sorted_rmse_108_5 = np.sort(rmse_values_108_5)

sorted_rmse_216_1 = np.sort(rmse_values_216_1)
sorted_rmse_216_2 = np.sort(rmse_values_216_2)
sorted_rmse_216_3 = np.sort(rmse_values_216_3)
sorted_rmse_216_4 = np.sort(rmse_values_216_4)
sorted_rmse_216_5 = np.sort(rmse_values_216_5)

yrmse_15_1 = np.arange(len(sorted_rmse_15_1)) / float(len(sorted_rmse_15_1) - 1)
yrmse_15_2 = np.arange(len(sorted_rmse_15_2)) / float(len(sorted_rmse_15_2) - 1)
yrmse_15_3 = np.arange(len(sorted_rmse_15_3)) / float(len(sorted_rmse_15_3) - 1)
yrmse_15_4 = np.arange(len(sorted_rmse_15_4)) / float(len(sorted_rmse_15_4) - 1)
yrmse_15_5 = np.arange(len(sorted_rmse_15_5)) / float(len(sorted_rmse_15_5) - 1)

yrmse_21_1 = np.arange(len(sorted_rmse_21_1)) / float(len(sorted_rmse_21_1) - 1)
yrmse_21_2 = np.arange(len(sorted_rmse_21_2)) / float(len(sorted_rmse_21_2) - 1)
yrmse_21_3 = np.arange(len(sorted_rmse_21_3)) / float(len(sorted_rmse_21_3) - 1)
yrmse_21_4 = np.arange(len(sorted_rmse_21_4)) / float(len(sorted_rmse_21_4) - 1)
yrmse_21_5 = np.arange(len(sorted_rmse_21_5)) / float(len(sorted_rmse_21_5) - 1)

yrmse_24_1 = np.arange(len(sorted_rmse_24_1)) / float(len(sorted_rmse_24_1) - 1)
yrmse_24_2 = np.arange(len(sorted_rmse_24_2)) / float(len(sorted_rmse_24_2) - 1)
yrmse_24_3 = np.arange(len(sorted_rmse_24_3)) / float(len(sorted_rmse_24_3) - 1)
yrmse_24_4 = np.arange(len(sorted_rmse_24_4)) / float(len(sorted_rmse_24_4) - 1)
yrmse_24_5 = np.arange(len(sorted_rmse_24_5)) / float(len(sorted_rmse_24_5) - 1)

yrmse_30_1 = np.arange(len(sorted_rmse_30_1)) / float(len(sorted_rmse_30_1) - 1)
yrmse_30_2 = np.arange(len(sorted_rmse_30_2)) / float(len(sorted_rmse_30_2) - 1)
yrmse_30_3 = np.arange(len(sorted_rmse_30_3)) / float(len(sorted_rmse_30_3) - 1)
yrmse_30_4 = np.arange(len(sorted_rmse_30_4)) / float(len(sorted_rmse_30_4) - 1)
yrmse_30_5 = np.arange(len(sorted_rmse_30_5)) / float(len(sorted_rmse_30_5) - 1)

yrmse_33_1 = np.arange(len(sorted_rmse_33_1)) / float(len(sorted_rmse_33_1) - 1)
yrmse_33_2 = np.arange(len(sorted_rmse_33_2)) / float(len(sorted_rmse_33_2) - 1)
yrmse_33_3 = np.arange(len(sorted_rmse_33_3)) / float(len(sorted_rmse_33_3) - 1)
yrmse_33_4 = np.arange(len(sorted_rmse_33_4)) / float(len(sorted_rmse_33_4) - 1)
yrmse_33_5 = np.arange(len(sorted_rmse_33_5)) / float(len(sorted_rmse_33_5) - 1)

yrmse_36_1 = np.arange(len(sorted_rmse_36_1)) / float(len(sorted_rmse_36_1) - 1)
yrmse_36_2 = np.arange(len(sorted_rmse_36_2)) / float(len(sorted_rmse_36_2) - 1)
yrmse_36_3 = np.arange(len(sorted_rmse_36_3)) / float(len(sorted_rmse_36_3) - 1)
yrmse_36_4 = np.arange(len(sorted_rmse_36_4)) / float(len(sorted_rmse_36_4) - 1)
yrmse_36_5 = np.arange(len(sorted_rmse_36_5)) / float(len(sorted_rmse_36_5) - 1)

yrmse_54_1 = np.arange(len(sorted_rmse_54_1)) / float(len(sorted_rmse_54_1) - 1)
yrmse_54_2 = np.arange(len(sorted_rmse_54_2)) / float(len(sorted_rmse_54_2) - 1)
yrmse_54_3 = np.arange(len(sorted_rmse_54_3)) / float(len(sorted_rmse_54_3) - 1)
yrmse_54_4 = np.arange(len(sorted_rmse_54_4)) / float(len(sorted_rmse_54_4) - 1)
yrmse_54_5 = np.arange(len(sorted_rmse_54_5)) / float(len(sorted_rmse_54_5) - 1)

yrmse_78_1 = np.arange(len(sorted_rmse_78_1)) / float(len(sorted_rmse_78_1) - 1)
yrmse_78_2 = np.arange(len(sorted_rmse_78_2)) / float(len(sorted_rmse_78_2) - 1)
yrmse_78_3 = np.arange(len(sorted_rmse_78_3)) / float(len(sorted_rmse_78_3) - 1)
yrmse_78_4 = np.arange(len(sorted_rmse_78_4)) / float(len(sorted_rmse_78_4) - 1)
yrmse_78_5 = np.arange(len(sorted_rmse_78_5)) / float(len(sorted_rmse_78_5) - 1)

yrmse_108_1 = np.arange(len(sorted_rmse_108_1)) / float(len(sorted_rmse_108_1) - 1)
yrmse_108_2 = np.arange(len(sorted_rmse_108_2)) / float(len(sorted_rmse_108_2) - 1)
yrmse_108_3 = np.arange(len(sorted_rmse_108_3)) / float(len(sorted_rmse_108_3) - 1)
yrmse_108_4 = np.arange(len(sorted_rmse_108_4)) / float(len(sorted_rmse_108_4) - 1)
yrmse_108_5 = np.arange(len(sorted_rmse_108_5)) / float(len(sorted_rmse_108_5) - 1)

yrmse_216_1 = np.arange(len(sorted_rmse_216_1)) / float(len(sorted_rmse_216_1) - 1)
yrmse_216_2 = np.arange(len(sorted_rmse_216_2)) / float(len(sorted_rmse_216_2) - 1)
yrmse_216_3 = np.arange(len(sorted_rmse_216_3)) / float(len(sorted_rmse_216_3) - 1)
yrmse_216_4 = np.arange(len(sorted_rmse_216_4)) / float(len(sorted_rmse_216_4) - 1)
yrmse_216_5 = np.arange(len(sorted_rmse_216_5)) / float(len(sorted_rmse_216_5) - 1)

# 设置全局字体和字号
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10  # 设置全局字体大小
# 创建一个2行3列的子图网格
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(nrows=3, ncols=3, figsize=(6.83, 6.45))
lines = []
labels = []

ax1.plot(rmse_values_21_1, yrmse_21_1, marker='o', linestyle='', markersize=4, label='Threshold 0.1 (m)')
ax1.plot(rmse_values_21_2, yrmse_21_2, marker='o', linestyle='', markersize=4, label='Threshold 0.2 (m)')
ax1.plot(rmse_values_21_3, yrmse_21_3, marker='o', linestyle='', markersize=4, label='Threshold 0.3 (m)')
ax1.plot(rmse_values_21_4, yrmse_21_4, marker='o', linestyle='',markersize=4, label='Threshold 0.4 (m)')
ax1.plot(rmse_values_21_5, yrmse_21_5, marker='o', linestyle='',markersize=4, label='Threshold 0.5 (m)')
ax1.set_xlabel('RMSE')
ax1.set_ylabel('Cumulative Probability')
ax1.text(-0.1, 1.15, '(a)', transform=ax1.transAxes, fontsize=10, fontweight='bold',verticalalignment='top')
ax1.set_yticks([0,0.25,0.5,0.75,1])  # 设置y轴刻度
ax1.set_xticks([0,0.06,0.12,0.18,0.24])  # 设置x轴刻度

ax2.plot(rmse_values_24_1, yrmse_24_1, marker='o', linestyle='', markersize=4, label='Threshold 0.1 (m)')
ax2.plot(rmse_values_24_2, yrmse_24_2, marker='o', linestyle='', markersize=4, label='Threshold 0.2 (m)')
ax2.plot(rmse_values_24_3, yrmse_24_3, marker='o', linestyle='', markersize=4, label='Threshold 0.3 (m)')
ax2.plot(rmse_values_24_4, yrmse_24_4, marker='o', linestyle='',markersize=4, label='Threshold 0.4 (m)')
ax2.plot(rmse_values_24_5, yrmse_24_5, marker='o', linestyle='',markersize=4, label='Threshold 0.5 (m)')
ax2.set_xlabel('RMSE')
ax2.text(-0.1, 1.15, '(b)', transform=ax2.transAxes, fontsize=10, fontweight='bold',verticalalignment='top')
ax2.set_yticks([0,0.25,0.5,0.75,1])  # 设置y轴刻度
ax2.set_xticks([0,0.06,0.12,0.18,0.24])  # 设置x轴刻度

ax3.plot(rmse_values_30_1, yrmse_30_1, marker='o', linestyle='', markersize=4, label='Threshold 0.1 (m)')
ax3.plot(rmse_values_30_2, yrmse_30_2, marker='o', linestyle='', markersize=4, label='Threshold 0.2 (m)')
ax3.plot(rmse_values_30_3, yrmse_30_3, marker='o', linestyle='', markersize=4, label='Threshold 0.3 (m)')
ax3.plot(rmse_values_30_4, yrmse_30_4, marker='o', linestyle='',markersize=4, label='Threshold 0.4 (m)')
ax3.plot(rmse_values_30_5, yrmse_30_5, marker='o', linestyle='',markersize=4, label='Threshold 0.5 (m)')
ax3.set_xlabel('RMSE')
ax3.text(-0.1, 1.15, '(c)', transform=ax3.transAxes, fontsize=10, fontweight='bold',verticalalignment='top')
ax3.set_yticks([0,0.25,0.5,0.75,1])  # 设置y轴刻度
ax3.set_xticks([0,0.06,0.12,0.18,0.24])  # 设置x轴刻度

ax4.plot(rmse_values_33_1, yrmse_33_1, marker='o', linestyle='', markersize=4, label='Threshold 0.1 (m)')
ax4.plot(rmse_values_33_2, yrmse_33_2, marker='o', linestyle='', markersize=4, label='Threshold 0.2 (m)')
ax4.plot(rmse_values_33_3, yrmse_33_3, marker='o', linestyle='', markersize=4, label='Threshold 0.3 (m)')
ax4.plot(rmse_values_33_4, yrmse_33_4, marker='o', linestyle='',markersize=4, label='Threshold 0.4 (m)')
ax4.plot(rmse_values_33_5, yrmse_33_5, marker='o', linestyle='',markersize=4, label='Threshold 0.5 (m)')
ax4.set_xlabel('RMSE')
ax4.set_ylabel('Cumulative Probability')
ax4.text(-0.1, 1.15, '(d)', transform=ax4.transAxes, fontsize=10, fontweight='bold',verticalalignment='top')
ax4.set_yticks([0,0.25,0.5,0.75,1])  # 设置y轴刻度
ax4.set_xticks([0,0.06,0.12,0.18,0.24])  # 设置x轴刻度

lines += ax5.plot(rmse_values_36_1, yrmse_36_1, marker='o', linestyle='', markersize=4, label=r"$E_{r} = 0.1 \, \text{m}$")
lines += ax5.plot(rmse_values_36_2, yrmse_36_2, marker='o', linestyle='', markersize=4, label=r"$E_{r} = 0.2 \, \text{m}$")
lines += ax5.plot(rmse_values_36_3, yrmse_36_3, marker='o', linestyle='', markersize=4, label=r"$E_{r} = 0.3 \, \text{m}$")
lines += ax5.plot(rmse_values_36_4, yrmse_36_4, marker='o', linestyle='',markersize=4, label=r"$E_{r} = 0.4 \, \text{m}$")
lines += ax5.plot(rmse_values_36_5, yrmse_36_5, marker='o', linestyle='',markersize=4, label=r"$E_{r} = 0.5 \, \text{m}$")
ax5.set_xlabel('RMSE')
ax5.text(-0.1, 1.15, '(e)', transform=ax5.transAxes, fontsize=10, fontweight='bold',verticalalignment='top')
ax5.set_yticks([0,0.25,0.5,0.75,1])  # 设置y轴刻度
ax5.set_xticks([0,0.06,0.12,0.18,0.24])  # 设置x轴刻度

ax6.plot(rmse_values_54_1, yrmse_54_1, marker='o', linestyle='', markersize=4, label='Threshold 0.1 (m)')
ax6.plot(rmse_values_54_2, yrmse_54_2, marker='o', linestyle='', markersize=4, label='Threshold 0.2 (m)')
ax6.plot(rmse_values_54_3, yrmse_54_3, marker='o', linestyle='', markersize=4, label='Threshold 0.3 (m)')
ax6.plot(rmse_values_54_4, yrmse_54_4, marker='o', linestyle='',markersize=4, label='Threshold 0.4 (m)')
ax6.plot(rmse_values_54_5, yrmse_54_5, marker='o', linestyle='',markersize=4, label='Threshold 0.5 (m)')
ax6.set_xlabel('RMSE')
ax6.text(-0.1, 1.15, '(f)', transform=ax6.transAxes, fontsize=10, fontweight='bold',verticalalignment='top')
ax6.set_yticks([0,0.25,0.5,0.75,1])  # 设置y轴刻度
ax6.set_xticks([0,0.06,0.12,0.18,0.24])  # 设置x轴刻度

ax7.plot(rmse_values_78_1, yrmse_78_1, marker='o', linestyle='', markersize=4, label='Threshold 0.1 (m)')
ax7.plot(rmse_values_78_2, yrmse_78_2, marker='o', linestyle='', markersize=4, label='Threshold 0.2 (m)')
ax7.plot(rmse_values_78_3, yrmse_78_3, marker='o', linestyle='', markersize=4, label='Threshold 0.3 (m)')
ax7.plot(rmse_values_78_4, yrmse_78_4, marker='o', linestyle='',markersize=4, label='Threshold 0.4 (m)')
ax7.plot(rmse_values_78_5, yrmse_78_5, marker='o', linestyle='',markersize=4, label='Threshold 0.5 (m)')
ax7.set_xlabel('RMSE')
ax7.set_ylabel('Cumulative Probability')
ax7.text(-0.1, 1.15, '(g)', transform=ax7.transAxes, fontsize=10, fontweight='bold',verticalalignment='top')
ax7.set_yticks([0,0.25,0.5,0.75,1])  # 设置y轴刻度
ax7.set_xticks([0,0.06,0.12,0.18,0.24])  # 设置x轴刻度

ax8.plot(rmse_values_108_1, yrmse_108_1, marker='o', linestyle='', markersize=4, label='Threshold 0.1 (m)')
ax8.plot(rmse_values_108_2, yrmse_108_2, marker='o', linestyle='', markersize=4, label='Threshold 0.2 (m)')
ax8.plot(rmse_values_108_3, yrmse_108_3, marker='o', linestyle='', markersize=4, label='Threshold 0.3 (m)')
ax8.plot(rmse_values_108_4, yrmse_108_4, marker='o', linestyle='',markersize=4, label='Threshold 0.4 (m)')
ax8.plot(rmse_values_108_5, yrmse_108_5, marker='o', linestyle='',markersize=4, label='Threshold 0.5 (m)')
ax8.set_xlabel('RMSE')
ax8.text(-0.1, 1.15, '(h)', transform=ax8.transAxes, fontsize=10, fontweight='bold',verticalalignment='top')
ax8.set_yticks([0,0.25,0.5,0.75,1])  # 设置y轴刻度
ax8.set_xticks([0,0.06,0.12,0.18,0.24])  # 设置x轴刻度

ax9.plot(rmse_values_216_1, yrmse_216_1, marker='o', linestyle='', markersize=4, label='Threshold 0.1 (m)')
ax9.plot(rmse_values_216_2, yrmse_216_2, marker='o', linestyle='', markersize=4, label='Threshold 0.2 (m)')
ax9.plot(rmse_values_216_3, yrmse_216_3, marker='o', linestyle='', markersize=4, label='Threshold 0.3 (m)')
ax9.plot(rmse_values_216_4, yrmse_216_4, marker='o', linestyle='',markersize=4, label='Threshold 0.4 (m)')
ax9.plot(rmse_values_216_5, yrmse_216_5, marker='o', linestyle='',markersize=4, label='Threshold 0.5 (m)')
ax9.set_xlabel('RMSE')
ax9.text(-0.1, 1.15, '(i)', transform=ax9.transAxes, fontsize=10, fontweight='bold',verticalalignment='top')
ax9.set_yticks([0,0.25,0.5,0.75,1])  # 设置y轴刻度
ax9.set_xticks([0,0.06,0.12,0.18,0.24])  # 设置x轴刻度

labels = [line.get_label() for line in lines]

# 在图的右上角添加图例
fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1),
           ncol=5, fancybox=True,frameon=False)

# 调整子图间距
plt.tight_layout(rect=[0, 0, 1, 1],h_pad=-0)  # 调整图形的整体布局
plt.subplots_adjust(top=0.9)
# 保存图像到文件，设置分辨率为600dpi
# fig.savefig(r"D:\Users\weishaosong\PycharmProjects\pressure_prediction/result_plotting/figure_transmitted_RMSE_2sd_45scenario.tiff", dpi=600,format="tiff", pil_kwargs={"compression": "tiff_lzw"})
# 显示图形
plt.show()




