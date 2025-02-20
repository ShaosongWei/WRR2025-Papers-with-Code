import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def calculate_EC(actual, predicted, error_limit, horizon):
    errors = np.abs(actual - predicted)
    below_limit = errors > error_limit
    ratio_over_limit = np.sum(below_limit) / len(actual)
    TT_prediciton = 96/horizon + (96-96/horizon)*ratio_over_limit
    EC = TT_prediciton/96
    return EC

EC_series_15_1 = pd.Series({col: calculate_EC(noise_df_15[col], predicted_df_15[col],0.1,1) for col in noise_df_15.columns})
EC_series_15_2 = pd.Series({col: calculate_EC(noise_df_15[col], predicted_df_15[col],0.2,1) for col in noise_df_15.columns})
EC_series_15_3 = pd.Series({col: calculate_EC(noise_df_15[col], predicted_df_15[col],0.3,1) for col in noise_df_15.columns})
EC_series_15_4 = pd.Series({col: calculate_EC(noise_df_15[col], predicted_df_15[col],0.4,1) for col in noise_df_15.columns})
EC_series_15_5 = pd.Series({col: calculate_EC(noise_df_15[col], predicted_df_15[col],0.5,1) for col in noise_df_15.columns})

EC_series_21_1 = pd.Series({col: calculate_EC(noise_df_21[col], predicted_df_21[col],0.1,2) for col in noise_df_21.columns})
EC_series_21_2 = pd.Series({col: calculate_EC(noise_df_21[col], predicted_df_21[col],0.2,2) for col in noise_df_21.columns})
EC_series_21_3 = pd.Series({col: calculate_EC(noise_df_21[col], predicted_df_21[col],0.3,2) for col in noise_df_21.columns})
EC_series_21_4 = pd.Series({col: calculate_EC(noise_df_21[col], predicted_df_21[col],0.4,2) for col in noise_df_21.columns})
EC_series_21_5 = pd.Series({col: calculate_EC(noise_df_21[col], predicted_df_21[col],0.5,2) for col in noise_df_21.columns})

EC_series_24_1 = pd.Series({col: calculate_EC(noise_df_24[col], predicted_df_24[col],0.1,3) for col in noise_df_24.columns})
EC_series_24_2 = pd.Series({col: calculate_EC(noise_df_24[col], predicted_df_24[col],0.2,3) for col in noise_df_24.columns})
EC_series_24_3 = pd.Series({col: calculate_EC(noise_df_24[col], predicted_df_24[col],0.3,3) for col in noise_df_24.columns})
EC_series_24_4 = pd.Series({col: calculate_EC(noise_df_24[col], predicted_df_24[col],0.4,3) for col in noise_df_24.columns})
EC_series_24_5 = pd.Series({col: calculate_EC(noise_df_24[col], predicted_df_24[col],0.5,3) for col in noise_df_24.columns})

EC_series_30_1 = pd.Series({col: calculate_EC(noise_df_30[col], predicted_df_30[col],0.1,4) for col in noise_df_30.columns})
EC_series_30_2 = pd.Series({col: calculate_EC(noise_df_30[col], predicted_df_30[col],0.2,4) for col in noise_df_30.columns})
EC_series_30_3 = pd.Series({col: calculate_EC(noise_df_30[col], predicted_df_30[col],0.3,4) for col in noise_df_30.columns})
EC_series_30_4 = pd.Series({col: calculate_EC(noise_df_30[col], predicted_df_30[col],0.4,4) for col in noise_df_30.columns})
EC_series_30_5 = pd.Series({col: calculate_EC(noise_df_30[col], predicted_df_30[col],0.5,4) for col in noise_df_30.columns})

EC_series_33_1 = pd.Series({col: calculate_EC(noise_df_33[col], predicted_df_33[col],0.1,5) for col in noise_df_33.columns})
EC_series_33_2 = pd.Series({col: calculate_EC(noise_df_33[col], predicted_df_33[col],0.2,5) for col in noise_df_33.columns})
EC_series_33_3 = pd.Series({col: calculate_EC(noise_df_33[col], predicted_df_33[col],0.3,5) for col in noise_df_33.columns})
EC_series_33_4 = pd.Series({col: calculate_EC(noise_df_33[col], predicted_df_33[col],0.4,5) for col in noise_df_33.columns})
EC_series_33_5 = pd.Series({col: calculate_EC(noise_df_33[col], predicted_df_33[col],0.5,5) for col in noise_df_33.columns})

EC_series_36_1 = pd.Series({col: calculate_EC(noise_df_36[col], predicted_df_36[col],0.1,6) for col in noise_df_36.columns})
EC_series_36_2 = pd.Series({col: calculate_EC(noise_df_36[col], predicted_df_36[col],0.2,6) for col in noise_df_36.columns})
EC_series_36_3 = pd.Series({col: calculate_EC(noise_df_36[col], predicted_df_36[col],0.3,6) for col in noise_df_36.columns})
EC_series_36_4 = pd.Series({col: calculate_EC(noise_df_36[col], predicted_df_36[col],0.4,6) for col in noise_df_36.columns})
EC_series_36_5 = pd.Series({col: calculate_EC(noise_df_36[col], predicted_df_36[col],0.5,6) for col in noise_df_36.columns})

EC_series_54_1 = pd.Series({col: calculate_EC(noise_df_54[col], predicted_df_54[col],0.1,12) for col in noise_df_54.columns})
EC_series_54_2 = pd.Series({col: calculate_EC(noise_df_54[col], predicted_df_54[col],0.2,12) for col in noise_df_54.columns})
EC_series_54_3 = pd.Series({col: calculate_EC(noise_df_54[col], predicted_df_54[col],0.3,12) for col in noise_df_54.columns})
EC_series_54_4 = pd.Series({col: calculate_EC(noise_df_54[col], predicted_df_54[col],0.4,12) for col in noise_df_54.columns})
EC_series_54_5 = pd.Series({col: calculate_EC(noise_df_54[col], predicted_df_54[col],0.5,12) for col in noise_df_54.columns})

EC_series_78_1 = pd.Series({col: calculate_EC(noise_df_78[col], predicted_df_78[col],0.1,24) for col in noise_df_78.columns})
EC_series_78_2 = pd.Series({col: calculate_EC(noise_df_78[col], predicted_df_78[col],0.2,24) for col in noise_df_78.columns})
EC_series_78_3 = pd.Series({col: calculate_EC(noise_df_78[col], predicted_df_78[col],0.3,24) for col in noise_df_78.columns})
EC_series_78_4 = pd.Series({col: calculate_EC(noise_df_78[col], predicted_df_78[col],0.4,24) for col in noise_df_78.columns})
EC_series_78_5 = pd.Series({col: calculate_EC(noise_df_78[col], predicted_df_78[col],0.5,24) for col in noise_df_78.columns})

EC_series_108_1 = pd.Series({col: calculate_EC(noise_df_108[col], predicted_df_108[col],0.1,48) for col in noise_df_108.columns})
EC_series_108_2 = pd.Series({col: calculate_EC(noise_df_108[col], predicted_df_108[col],0.2,48) for col in noise_df_108.columns})
EC_series_108_3 = pd.Series({col: calculate_EC(noise_df_108[col], predicted_df_108[col],0.3,48) for col in noise_df_108.columns})
EC_series_108_4 = pd.Series({col: calculate_EC(noise_df_108[col], predicted_df_108[col],0.4,48) for col in noise_df_108.columns})
EC_series_108_5 = pd.Series({col: calculate_EC(noise_df_108[col], predicted_df_108[col],0.5,48) for col in noise_df_108.columns})

EC_series_216_1 = pd.Series({col: calculate_EC(noise_df_216[col], predicted_df_216[col],0.1,96) for col in noise_df_216.columns})
EC_series_216_2 = pd.Series({col: calculate_EC(noise_df_216[col], predicted_df_216[col],0.2,96) for col in noise_df_216.columns})
EC_series_216_3 = pd.Series({col: calculate_EC(noise_df_216[col], predicted_df_216[col],0.3,96) for col in noise_df_216.columns})
EC_series_216_4 = pd.Series({col: calculate_EC(noise_df_216[col], predicted_df_216[col],0.4,96) for col in noise_df_216.columns})
EC_series_216_5 = pd.Series({col: calculate_EC(noise_df_216[col], predicted_df_216[col],0.5,96) for col in noise_df_216.columns})

# 绘制累计概率分布图
EC_series_15_1 = EC_series_15_1.values
EC_series_15_2 = EC_series_15_2.values
EC_series_15_3 = EC_series_15_3.values
EC_series_15_4 = EC_series_15_4.values
EC_series_15_5 = EC_series_15_5.values
sorted_EC_15_1 = np.sort(EC_series_15_1)
sorted_EC_15_2 = np.sort(EC_series_15_2)
sorted_EC_15_3 = np.sort(EC_series_15_3)
sorted_EC_15_4 = np.sort(EC_series_15_4)
sorted_EC_15_5 = np.sort(EC_series_15_5)
EC_15_1 = np.arange(len(sorted_EC_15_1)) / float(len(sorted_EC_15_1) - 1)
EC_15_2 = np.arange(len(sorted_EC_15_2)) / float(len(sorted_EC_15_2) - 1)
EC_15_3 = np.arange(len(sorted_EC_15_3)) / float(len(sorted_EC_15_3) - 1)
EC_15_4 = np.arange(len(sorted_EC_15_4)) / float(len(sorted_EC_15_4) - 1)
EC_15_5 = np.arange(len(sorted_EC_15_5)) / float(len(sorted_EC_15_5) - 1)

EC_series_21_1 = EC_series_21_1.values
EC_series_21_2 = EC_series_21_2.values
EC_series_21_3 = EC_series_21_3.values
EC_series_21_4 = EC_series_21_4.values
EC_series_21_5 = EC_series_21_5.values
sorted_EC_21_1 = np.sort(EC_series_21_1)
sorted_EC_21_2 = np.sort(EC_series_21_2)
sorted_EC_21_3 = np.sort(EC_series_21_3)
sorted_EC_21_4 = np.sort(EC_series_21_4)
sorted_EC_21_5 = np.sort(EC_series_21_5)
EC_21_1 = np.arange(len(sorted_EC_21_1)) / float(len(sorted_EC_21_1) - 1)
EC_21_2 = np.arange(len(sorted_EC_21_2)) / float(len(sorted_EC_21_2) - 1)
EC_21_3 = np.arange(len(sorted_EC_21_3)) / float(len(sorted_EC_21_3) - 1)
EC_21_4 = np.arange(len(sorted_EC_21_4)) / float(len(sorted_EC_21_4) - 1)
EC_21_5 = np.arange(len(sorted_EC_21_5)) / float(len(sorted_EC_21_5) - 1)

EC_series_24_1 = EC_series_24_1.values
EC_series_24_2 = EC_series_24_2.values
EC_series_24_3 = EC_series_24_3.values
EC_series_24_4 = EC_series_24_4.values
EC_series_24_5 = EC_series_24_5.values
sorted_EC_24_1 = np.sort(EC_series_24_1)
sorted_EC_24_2 = np.sort(EC_series_24_2)
sorted_EC_24_3 = np.sort(EC_series_24_3)
sorted_EC_24_4 = np.sort(EC_series_24_4)
sorted_EC_24_5 = np.sort(EC_series_24_5)
EC_24_1 = np.arange(len(sorted_EC_24_1)) / float(len(sorted_EC_24_1) - 1)
EC_24_2 = np.arange(len(sorted_EC_24_2)) / float(len(sorted_EC_24_2) - 1)
EC_24_3 = np.arange(len(sorted_EC_24_3)) / float(len(sorted_EC_24_3) - 1)
EC_24_4 = np.arange(len(sorted_EC_24_4)) / float(len(sorted_EC_24_4) - 1)
EC_24_5 = np.arange(len(sorted_EC_24_5)) / float(len(sorted_EC_24_5) - 1)

EC_series_30_1 = EC_series_30_1.values
EC_series_30_2 = EC_series_30_2.values
EC_series_30_3 = EC_series_30_3.values
EC_series_30_4 = EC_series_30_4.values
EC_series_30_5 = EC_series_30_5.values
sorted_EC_30_1 = np.sort(EC_series_30_1)
sorted_EC_30_2 = np.sort(EC_series_30_2)
sorted_EC_30_3 = np.sort(EC_series_30_3)
sorted_EC_30_4 = np.sort(EC_series_30_4)
sorted_EC_30_5 = np.sort(EC_series_30_5)
EC_30_1 = np.arange(len(sorted_EC_30_1)) / float(len(sorted_EC_30_1) - 1)
EC_30_2 = np.arange(len(sorted_EC_30_2)) / float(len(sorted_EC_30_2) - 1)
EC_30_3 = np.arange(len(sorted_EC_30_3)) / float(len(sorted_EC_30_3) - 1)
EC_30_4 = np.arange(len(sorted_EC_30_4)) / float(len(sorted_EC_30_4) - 1)
EC_30_5 = np.arange(len(sorted_EC_30_5)) / float(len(sorted_EC_30_5) - 1)

EC_series_33_1 = EC_series_33_1.values
EC_series_33_2 = EC_series_33_2.values
EC_series_33_3 = EC_series_33_3.values
EC_series_33_4 = EC_series_33_4.values
EC_series_33_5 = EC_series_33_5.values
sorted_EC_33_1 = np.sort(EC_series_33_1)
sorted_EC_33_2 = np.sort(EC_series_33_2)
sorted_EC_33_3 = np.sort(EC_series_33_3)
sorted_EC_33_4 = np.sort(EC_series_33_4)
sorted_EC_33_5 = np.sort(EC_series_33_5)
EC_33_1 = np.arange(len(sorted_EC_33_1)) / float(len(sorted_EC_33_1) - 1)
EC_33_2 = np.arange(len(sorted_EC_33_2)) / float(len(sorted_EC_33_2) - 1)
EC_33_3 = np.arange(len(sorted_EC_33_3)) / float(len(sorted_EC_33_3) - 1)
EC_33_4 = np.arange(len(sorted_EC_33_4)) / float(len(sorted_EC_33_4) - 1)
EC_33_5 = np.arange(len(sorted_EC_33_5)) / float(len(sorted_EC_33_5) - 1)

EC_series_36_1 = EC_series_36_1.values
EC_series_36_2 = EC_series_36_2.values
EC_series_36_3 = EC_series_36_3.values
EC_series_36_4 = EC_series_36_4.values
EC_series_36_5 = EC_series_36_5.values
sorted_EC_36_1 = np.sort(EC_series_36_1)
sorted_EC_36_2 = np.sort(EC_series_36_2)
sorted_EC_36_3 = np.sort(EC_series_36_3)
sorted_EC_36_4 = np.sort(EC_series_36_4)
sorted_EC_36_5 = np.sort(EC_series_36_5)
EC_36_1 = np.arange(len(sorted_EC_36_1)) / float(len(sorted_EC_36_1) - 1)
EC_36_2 = np.arange(len(sorted_EC_36_2)) / float(len(sorted_EC_36_2) - 1)
EC_36_3 = np.arange(len(sorted_EC_36_3)) / float(len(sorted_EC_36_3) - 1)
EC_36_4 = np.arange(len(sorted_EC_36_4)) / float(len(sorted_EC_36_4) - 1)
EC_36_5 = np.arange(len(sorted_EC_36_5)) / float(len(sorted_EC_36_5) - 1)

EC_series_54_1 = EC_series_54_1.values
EC_series_54_2 = EC_series_54_2.values
EC_series_54_3 = EC_series_54_3.values
EC_series_54_4 = EC_series_54_4.values
EC_series_54_5 = EC_series_54_5.values
sorted_EC_54_1 = np.sort(EC_series_54_1)
sorted_EC_54_2 = np.sort(EC_series_54_2)
sorted_EC_54_3 = np.sort(EC_series_54_3)
sorted_EC_54_4 = np.sort(EC_series_54_4)
sorted_EC_54_5 = np.sort(EC_series_54_5)
EC_54_1 = np.arange(len(sorted_EC_54_1)) / float(len(sorted_EC_54_1) - 1)
EC_54_2 = np.arange(len(sorted_EC_54_2)) / float(len(sorted_EC_54_2) - 1)
EC_54_3 = np.arange(len(sorted_EC_54_3)) / float(len(sorted_EC_54_3) - 1)
EC_54_4 = np.arange(len(sorted_EC_54_4)) / float(len(sorted_EC_54_4) - 1)
EC_54_5 = np.arange(len(sorted_EC_54_5)) / float(len(sorted_EC_54_5) - 1)

EC_series_78_1 = EC_series_78_1.values
EC_series_78_2 = EC_series_78_2.values
EC_series_78_3 = EC_series_78_3.values
EC_series_78_4 = EC_series_78_4.values
EC_series_78_5 = EC_series_78_5.values
sorted_EC_78_1 = np.sort(EC_series_78_1)
sorted_EC_78_2 = np.sort(EC_series_78_2)
sorted_EC_78_3 = np.sort(EC_series_78_3)
sorted_EC_78_4 = np.sort(EC_series_78_4)
sorted_EC_78_5 = np.sort(EC_series_78_5)
EC_78_1 = np.arange(len(sorted_EC_78_1)) / float(len(sorted_EC_78_1) - 1)
EC_78_2 = np.arange(len(sorted_EC_78_2)) / float(len(sorted_EC_78_2) - 1)
EC_78_3 = np.arange(len(sorted_EC_78_3)) / float(len(sorted_EC_78_3) - 1)
EC_78_4 = np.arange(len(sorted_EC_78_4)) / float(len(sorted_EC_78_4) - 1)
EC_78_5 = np.arange(len(sorted_EC_78_5)) / float(len(sorted_EC_78_5) - 1)

EC_series_108_1 = EC_series_108_1.values
EC_series_108_2 = EC_series_108_2.values
EC_series_108_3 = EC_series_108_3.values
EC_series_108_4 = EC_series_108_4.values
EC_series_108_5 = EC_series_108_5.values
sorted_EC_108_1 = np.sort(EC_series_108_1)
sorted_EC_108_2 = np.sort(EC_series_108_2)
sorted_EC_108_3 = np.sort(EC_series_108_3)
sorted_EC_108_4 = np.sort(EC_series_108_4)
sorted_EC_108_5 = np.sort(EC_series_108_5)
EC_108_1 = np.arange(len(sorted_EC_108_1)) / float(len(sorted_EC_108_1) - 1)
EC_108_2 = np.arange(len(sorted_EC_108_2)) / float(len(sorted_EC_108_2) - 1)
EC_108_3 = np.arange(len(sorted_EC_108_3)) / float(len(sorted_EC_108_3) - 1)
EC_108_4 = np.arange(len(sorted_EC_108_4)) / float(len(sorted_EC_108_4) - 1)
EC_108_5 = np.arange(len(sorted_EC_108_5)) / float(len(sorted_EC_108_5) - 1)


EC_series_216_1 = EC_series_216_1.values
EC_series_216_2 = EC_series_216_2.values
EC_series_216_3 = EC_series_216_3.values
EC_series_216_4 = EC_series_216_4.values
EC_series_216_5 = EC_series_216_5.values
sorted_EC_216_1 = np.sort(EC_series_216_1)
sorted_EC_216_2 = np.sort(EC_series_216_2)
sorted_EC_216_3 = np.sort(EC_series_216_3)
sorted_EC_216_4 = np.sort(EC_series_216_4)
sorted_EC_216_5 = np.sort(EC_series_216_5)
EC_216_1 = np.arange(len(sorted_EC_216_1)) / float(len(sorted_EC_216_1) - 1)
EC_216_2 = np.arange(len(sorted_EC_216_2)) / float(len(sorted_EC_216_2) - 1)
EC_216_3 = np.arange(len(sorted_EC_216_3)) / float(len(sorted_EC_216_3) - 1)
EC_216_4 = np.arange(len(sorted_EC_216_4)) / float(len(sorted_EC_216_4) - 1)
EC_216_5 = np.arange(len(sorted_EC_216_5)) / float(len(sorted_EC_216_5) - 1)

# 计算能量节约倍数的均值和方差
mean_15_1 = np.mean(EC_series_15_1)
print("mean_15_1: {}".format(mean_15_1))
mean_15_2 = np.mean(EC_series_15_2)
print("mean_15_2: {}".format(mean_15_2))
mean_15_3 = np.mean(EC_series_15_3)
print("mean_15_3: {}".format(mean_15_3))
mean_15_4 = np.mean(EC_series_15_4)
print("mean_15_4: {}".format(mean_15_4))
mean_15_5 = np.mean(EC_series_15_5)
print("mean_15_5: {}".format(mean_15_5))

mean_21_1 = np.mean(EC_series_21_1)
print("mean_21_1: {}".format(mean_21_1))
mean_21_2 = np.mean(EC_series_21_2)
print("mean_21_2: {}".format(mean_21_2))
mean_21_3 = np.mean(EC_series_21_3)
print("mean_21_3: {}".format(mean_21_3))
mean_21_4 = np.mean(EC_series_21_4)
print("mean_21_4: {}".format(mean_21_4))
mean_21_5 = np.mean(EC_series_21_5)
print("mean_21_5: {}".format(mean_21_5))

mean_24_1 = np.mean(EC_series_24_1)
print("mean_24_1: {}".format(mean_24_1))
mean_24_2 = np.mean(EC_series_24_2)
print("mean_24_2: {}".format(mean_24_2))
mean_24_3 = np.mean(EC_series_24_3)
print("mean_24_3: {}".format(mean_24_3))
mean_24_4 = np.mean(EC_series_24_4)
print("mean_24_4: {}".format(mean_24_4))
mean_24_5 = np.mean(EC_series_24_5)
print("mean_24_5: {}".format(mean_24_5))

mean_30_1 = np.mean(EC_series_30_1)
print("mean_30_1: {}".format(mean_30_1))
mean_30_2 = np.mean(EC_series_30_2)
print("mean_30_2: {}".format(mean_30_2))
mean_30_3 = np.mean(EC_series_30_3)
print("mean_30_3: {}".format(mean_30_3))
mean_30_4 = np.mean(EC_series_30_4)
print("mean_30_4: {}".format(mean_30_4))
mean_30_5 = np.mean(EC_series_30_5)
print("mean_30_5: {}".format(mean_30_5))

mean_33_1 = np.mean(EC_series_33_1)
print("mean_33_1: {}".format(mean_33_1))
mean_33_2 = np.mean(EC_series_33_2)
print("mean_33_2: {}".format(mean_33_2))
mean_33_3 = np.mean(EC_series_33_3)
print("mean_33_3: {}".format(mean_33_3))
mean_33_4 = np.mean(EC_series_33_4)
print("mean_33_4: {}".format(mean_33_4))
mean_33_5 = np.mean(EC_series_33_5)
print("mean_33_5: {}".format(mean_33_5))

mean_36_1 = np.mean(EC_series_36_1)
print("mean_36_1: {}".format(mean_36_1))
mean_36_2 = np.mean(EC_series_36_2)
print("mean_36_2: {}".format(mean_36_2))
mean_36_3 = np.mean(EC_series_36_3)
print("mean_36_3: {}".format(mean_36_3))
mean_36_4 = np.mean(EC_series_36_4)
print("mean_36_4: {}".format(mean_36_4))
mean_36_5 = np.mean(EC_series_36_5)
print("mean_36_5: {}".format(mean_36_5))

mean_54_1 = np.mean(EC_series_54_1)
print("mean_54_1: {}".format(mean_54_1))
mean_54_2 = np.mean(EC_series_54_2)
print("mean_54_2: {}".format(mean_54_2))
mean_54_3 = np.mean(EC_series_54_3)
print("mean_54_3: {}".format(mean_54_3))
mean_54_4 = np.mean(EC_series_54_4)
print("mean_54_4: {}".format(mean_54_4))
mean_54_5 = np.mean(EC_series_54_5)
print("mean_54_5: {}".format(mean_54_5))

mean_78_1 = np.mean(EC_series_78_1)
print("mean_78_1: {}".format(mean_78_1))
mean_78_2 = np.mean(EC_series_78_2)
print("mean_78_2: {}".format(mean_78_2))
mean_78_3 = np.mean(EC_series_78_3)
print("mean_78_3: {}".format(mean_78_3))
mean_78_4 = np.mean(EC_series_78_4)
print("mean_78_4: {}".format(mean_78_4))
mean_78_5 = np.mean(EC_series_78_5)
print("mean_78_5: {}".format(mean_78_5))

mean_108_1 = np.mean(EC_series_108_1)
print("mean_108_1: {}".format(mean_108_1))
mean_108_2 = np.mean(EC_series_108_2)
print("mean_108_2: {}".format(mean_108_2))
mean_108_3 = np.mean(EC_series_108_3)
print("mean_108_3: {}".format(mean_108_3))
mean_108_4 = np.mean(EC_series_108_4)
print("mean_108_4: {}".format(mean_108_4))
mean_108_5 = np.mean(EC_series_108_5)
print("mean_108_5: {}".format(mean_108_5))

mean_216_1 = np.mean(EC_series_216_1)
print("mean_216_1: {}".format(mean_216_1))
mean_216_2 = np.mean(EC_series_216_2)
print("mean_216_2: {}".format(mean_216_2))
mean_216_3 = np.mean(EC_series_216_3)
print("mean_216_3: {}".format(mean_216_3))
mean_216_4 = np.mean(EC_series_216_4)
print("mean_216_4: {}".format(mean_216_4))
mean_216_5 = np.mean(EC_series_216_5)
print("mean_216_5: {}".format(mean_216_5))

var_15_1 = np.var(EC_series_15_1)
print("var_15_1: {}".format(var_15_1))
var_15_2 = np.var(EC_series_15_2)
print("var_15_2: {}".format(var_15_2))
var_15_3 = np.var(EC_series_15_3)
print("var_15_3: {}".format(var_15_3))
var_15_4 = np.var(EC_series_15_4)
print("var_15_4: {}".format(var_15_4))
var_15_5 = np.var(EC_series_15_5)
print("var_15_5: {}".format(var_15_5))

var_21_1 = np.var(EC_series_21_1)
print("var_21_1: {}".format(var_21_1))
var_21_2 = np.var(EC_series_21_2)
print("var_21_2: {}".format(var_21_2))
var_21_3 = np.var(EC_series_21_3)
print("var_21_3: {}".format(var_21_3))
var_21_4 = np.var(EC_series_21_4)
print("var_21_4: {}".format(var_21_4))
var_21_5 = np.var(EC_series_21_5)
print("var_21_5: {}".format(var_21_5))

var_24_1 = np.var(EC_series_24_1)
print("var_24_1: {}".format(var_24_1))
var_24_2 = np.var(EC_series_24_2)
print("var_24_2: {}".format(var_24_2))
var_24_3 = np.var(EC_series_24_3)
print("var_24_3: {}".format(var_24_3))
var_24_4 = np.var(EC_series_24_4)
print("var_24_4: {}".format(var_24_4))
var_24_5 = np.var(EC_series_24_5)
print("var_24_5: {}".format(var_24_5))

var_30_1 = np.var(EC_series_30_1)
print("var_30_1: {}".format(var_30_1))
var_30_2 = np.var(EC_series_30_2)
print("var_30_2: {}".format(var_30_2))
var_30_3 = np.var(EC_series_30_3)
print("var_30_3: {}".format(var_30_3))
var_30_4 = np.var(EC_series_30_4)
print("var_30_4: {}".format(var_30_4))
var_30_5 = np.var(EC_series_30_5)
print("var_30_5: {}".format(var_30_5))

var_33_1 = np.var(EC_series_33_1)
print("var_33_1: {}".format(var_33_1))
var_33_2 = np.var(EC_series_33_2)
print("var_33_2: {}".format(var_33_2))
var_33_3 = np.var(EC_series_33_3)
print("var_33_3: {}".format(var_33_3))
var_33_4 = np.var(EC_series_33_4)
print("var_33_4: {}".format(var_33_4))
var_33_5 = np.var(EC_series_33_5)
print("var_33_5: {}".format(var_33_5))

var_36_1 = np.var(EC_series_36_1)
print("var_36_1: {}".format(var_36_1))
var_36_2 = np.var(EC_series_36_2)
print("var_36_2: {}".format(var_36_2))
var_36_3 = np.var(EC_series_36_3)
print("var_36_3: {}".format(var_36_3))
var_36_4 = np.var(EC_series_36_4)
print("var_36_4: {}".format(var_36_4))
var_36_5 = np.var(EC_series_36_5)
print("var_36_5: {}".format(var_36_5))

var_54_1 = np.var(EC_series_54_1)
print("var_54_1: {}".format(var_54_1))
var_54_2 = np.var(EC_series_54_2)
print("var_54_2: {}".format(var_54_2))
var_54_3 = np.var(EC_series_54_3)
print("var_54_3: {}".format(var_54_3))
var_54_4 = np.var(EC_series_54_4)
print("var_54_4: {}".format(var_54_4))
var_54_5 = np.var(EC_series_54_5)
print("var_54_5: {}".format(var_54_5))

var_78_1 = np.var(EC_series_78_1)
print("var_78_1: {}".format(var_78_1))
var_78_2 = np.var(EC_series_78_2)
print("var_78_2: {}".format(var_78_2))
var_78_3 = np.var(EC_series_78_3)
print("var_78_3: {}".format(var_78_3))
var_78_4 = np.var(EC_series_78_4)
print("var_78_4: {}".format(var_78_4))
var_78_5 = np.var(EC_series_78_5)
print("var_78_5: {}".format(var_78_5))

var_108_1 = np.var(EC_series_108_1)
print("var_108_1: {}".format(var_108_1))
var_108_2 = np.var(EC_series_108_2)
print("var_108_2: {}".format(var_108_2))
var_108_3 = np.var(EC_series_108_3)
print("var_108_3: {}".format(var_108_3))
var_108_4 = np.var(EC_series_108_4)
print("var_108_4: {}".format(var_108_4))
var_108_5 = np.var(EC_series_108_5)
print("var_108_5: {}".format(var_108_5))

var_216_1 = np.var(EC_series_216_1)
print("var_216_1: {}".format(var_216_1))
var_216_2 = np.var(EC_series_216_2)
print("var_216_2: {}".format(var_216_2))
var_216_3 = np.var(EC_series_216_3)
print("var_216_3: {}".format(var_216_3))
var_216_4 = np.var(EC_series_216_4)
print("var_216_4: {}".format(var_216_4))
var_216_5 = np.var(EC_series_216_5)
print("var_216_5: {}".format(var_216_5))

# 设置全局字体和字号
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10  # 设置全局字体大小
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(nrows=3, ncols=3, figsize=(6.83, 6.45))

# 为每条线加上 label 参数，但只在一个子图中调用 legend 方法
lines = []
labels = []

ax1.plot(sorted_EC_21_1, EC_21_1, marker='o', linestyle='', markersize=4, label='Threshold 0.1 (m)')
ax1.plot(sorted_EC_21_2, EC_21_2, marker='o', linestyle='', markersize=4, label='Threshold 0.2 (m)')
ax1.plot(sorted_EC_21_3, EC_21_3, marker='o', linestyle='', markersize=4, label='Threshold 0.3 (m)')
ax1.plot(sorted_EC_21_4, EC_21_4, marker='o', linestyle='',markersize=4, label='Threshold 0.4 (m)')
ax1.plot(sorted_EC_21_5, EC_21_5, marker='o', linestyle='',markersize=4, label='Threshold 0.5 (m)')
ax1.set_xlabel('ESR')
ax1.set_ylabel('Cumulative Probability')
ax1.text(-0.1, 1.15, '(a)', transform=ax1.transAxes, fontsize=10, fontweight='bold',verticalalignment='top')
ax1.set_yticks([0,0.25,0.5,0.75,1])  # 设置y轴刻度
ax1.set_xticks([0.2,0.4,0.6,0.8])  # 设置x轴刻度

ax2.plot(sorted_EC_24_1, EC_24_1, marker='o', linestyle='', markersize=4, label='Threshold 0.1 (m)')
ax2.plot(sorted_EC_24_2, EC_24_2, marker='o', linestyle='', markersize=4, label='Threshold 0.2 (m)')
ax2.plot(sorted_EC_24_3, EC_24_3, marker='o', linestyle='', markersize=4, label='Threshold 0.3 (m)')
ax2.plot(sorted_EC_24_4, EC_24_4, marker='o', linestyle='',markersize=4, label='Threshold 0.4 (m)')
ax2.plot(sorted_EC_24_5, EC_24_5, marker='o', linestyle='',markersize=4, label='Threshold 0.5 (m)')
ax2.set_xlabel('ESR')
ax2.text(-0.1, 1.15, '(b)', transform=ax2.transAxes, fontsize=10, fontweight='bold',verticalalignment='top')
ax2.set_yticks([0,0.25,0.5,0.75,1])  # 设置y轴刻度
ax2.set_xticks([0.2,0.4,0.6,0.8])  # 设置x轴刻度

ax3.plot(sorted_EC_30_1, EC_30_1, marker='o', linestyle='', markersize=4, label='Threshold 0.1 (m)')
ax3.plot(sorted_EC_30_2, EC_30_2, marker='o', linestyle='', markersize=4, label='Threshold 0.2 (m)')
ax3.plot(sorted_EC_30_3, EC_30_3, marker='o', linestyle='', markersize=4, label='Threshold 0.3 (m)')
ax3.plot(sorted_EC_30_4, EC_30_4, marker='o', linestyle='',markersize=4, label='Threshold 0.4 (m)')
ax3.plot(sorted_EC_30_5, EC_30_5, marker='o', linestyle='',markersize=4, label='Threshold 0.5 (m)')
ax3.set_xlabel('ESR')
ax3.text(-0.1, 1.15, '(c)', transform=ax3.transAxes, fontsize=10, fontweight='bold',verticalalignment='top')
ax3.set_yticks([0,0.25,0.5,0.75,1])  # 设置y轴刻度
ax3.set_xticks([0.2,0.4,0.6,0.8])  # 设置x轴刻度

ax4.plot(sorted_EC_33_1, EC_33_1, marker='o', linestyle='', markersize=4, label='Threshold 0.1 (m)')
ax4.plot(sorted_EC_33_2, EC_33_2, marker='o', linestyle='', markersize=4, label='Threshold 0.2 (m)')
ax4.plot(sorted_EC_33_3, EC_33_3, marker='o', linestyle='', markersize=4, label='Threshold 0.3 (m)')
ax4.plot(sorted_EC_33_4, EC_33_4, marker='o', linestyle='',markersize=4, label='Threshold 0.4 (m)')
ax4.plot(sorted_EC_33_5, EC_33_5, marker='o', linestyle='',markersize=4, label='Threshold 0.5 (m)')
ax4.set_xlabel('ESR')
ax4.set_ylabel('Cumulative Probability')
ax4.text(-0.1, 1.15, '(d)', transform=ax4.transAxes, fontsize=10, fontweight='bold',verticalalignment='top')
ax4.set_yticks([0,0.25,0.5,0.75,1])  # 设置y轴刻度
ax4.set_xticks([0.2,0.4,0.6,0.8])  # 设置x轴刻度


lines += ax5.plot(sorted_EC_36_1, EC_36_1, marker='o', linestyle='', markersize=4, label=r"$E_{r} = 0.1 \, \text{m}$")
lines += ax5.plot(sorted_EC_36_2, EC_36_2, marker='o', linestyle='', markersize=4, label=r"$E_{r} = 0.2 \, \text{m}$")
lines += ax5.plot(sorted_EC_36_3, EC_36_3, marker='o', linestyle='', markersize=4, label=r"$E_{r} = 0.3 \, \text{m}$")
lines += ax5.plot(sorted_EC_36_4, EC_36_4, marker='o', linestyle='',markersize=4, label=r"$E_{r} = 0.4 \, \text{m}$")
lines += ax5.plot(sorted_EC_36_5, EC_36_5, marker='o', linestyle='',markersize=4, label=r"$E_{r} = 0.5 \, \text{m}$")
ax5.set_xlabel('ESR')
ax5.text(-0.1, 1.15, '(e)', transform=ax5.transAxes, fontsize=10, fontweight='bold',verticalalignment='top')
ax5.set_yticks([0,0.25,0.5,0.75,1])  # 设置y轴刻度
ax5.set_xticks([0.2,0.4,0.6,0.8])    # 设置x轴刻度


ax6.plot(sorted_EC_54_1, EC_54_1, marker='o', linestyle='', markersize=4, label='Threshold 0.1 (m)')
ax6.plot(sorted_EC_54_2, EC_54_2, marker='o', linestyle='', markersize=4, label='Threshold 0.2 (m)')
ax6.plot(sorted_EC_54_3, EC_54_3, marker='o', linestyle='', markersize=4, label='Threshold 0.3 (m)')
ax6.plot(sorted_EC_54_4, EC_54_4, marker='o', linestyle='',markersize=4, label='Threshold 0.4 (m)')
ax6.plot(sorted_EC_54_5, EC_54_5, marker='o', linestyle='',markersize=4, label='Threshold 0.5 (m)')
ax6.set_xlabel('ESR')
ax6.text(-0.1, 1.15, '(f)', transform=ax6.transAxes, fontsize=10, fontweight='bold',verticalalignment='top')
ax6.set_yticks([0,0.25,0.5,0.75,1])  # 设置y轴刻度
ax6.set_xticks([0.2,0.4,0.6,0.8])  # 设置x轴刻度

ax7.plot(sorted_EC_78_1, EC_78_1, marker='o', linestyle='', markersize=4, label='Threshold 0.1 (m)')
ax7.plot(sorted_EC_78_2, EC_78_2, marker='o', linestyle='', markersize=4, label='Threshold 0.2 (m)')
ax7.plot(sorted_EC_78_3, EC_78_3, marker='o', linestyle='', markersize=4, label='Threshold 0.3 (m)')
ax7.plot(sorted_EC_78_4, EC_78_4, marker='o', linestyle='',markersize=4, label='Threshold 0.4 (m)')
ax7.plot(sorted_EC_78_5, EC_78_5, marker='o', linestyle='',markersize=4, label='Threshold 0.5 (m)')
ax7.set_xlabel('ESR')
ax7.set_ylabel('Cumulative Probability')
ax7.text(-0.1, 1.15, '(g)', transform=ax7.transAxes, fontsize=10, fontweight='bold',verticalalignment='top')
ax7.set_yticks([0,0.25,0.5,0.75,1])  # 设置y轴刻度
ax7.set_xticks([0.2,0.4,0.6,0.8])  # 设置x轴刻度

ax8.plot(sorted_EC_108_1, EC_108_1, marker='o', linestyle='', markersize=4, label='Threshold 0.1 (m)')
ax8.plot(sorted_EC_108_2, EC_108_2, marker='o', linestyle='', markersize=4, label='Threshold 0.2 (m)')
ax8.plot(sorted_EC_108_3, EC_108_3, marker='o', linestyle='', markersize=4, label='Threshold 0.3 (m)')
ax8.plot(sorted_EC_108_4, EC_108_4, marker='o', linestyle='',markersize=4, label='Threshold 0.4 (m)')
ax8.plot(sorted_EC_108_5, EC_108_5, marker='o', linestyle='',markersize=4, label='Threshold 0.5 (m)')
ax8.set_xlabel('ESR')
ax8.text(-0.1, 1.15, '(h)', transform=ax8.transAxes, fontsize=10, fontweight='bold',verticalalignment='top')
ax8.set_yticks([0,0.25,0.5,0.75,1])  # 设置y轴刻度
ax8.set_xticks([0.2,0.4,0.6,0.8])  # 设置x轴刻度

ax9.plot(sorted_EC_216_1, EC_216_1, marker='o', linestyle='', markersize=4, label='Threshold 0.1 (m)')
ax9.plot(sorted_EC_216_2, EC_216_2, marker='o', linestyle='', markersize=4, label='Threshold 0.2 (m)')
ax9.plot(sorted_EC_216_3, EC_216_3, marker='o', linestyle='', markersize=4, label='Threshold 0.3 (m)')
ax9.plot(sorted_EC_216_4, EC_216_4, marker='o', linestyle='',markersize=4, label='Threshold 0.4 (m)')
ax9.plot(sorted_EC_216_5, EC_216_5, marker='o', linestyle='',markersize=4, label='Threshold 0.5 (m)')
ax9.set_xlabel('ESR')
ax9.text(-0.1, 1.15, '(i)', transform=ax9.transAxes, fontsize=10, fontweight='bold',verticalalignment='top')
ax9.set_yticks([0,0.25,0.5,0.75,1])  # 设置y轴刻度
ax9.set_xticks([0.2,0.4,0.6,0.8])  # 设置x轴刻度

# 获取所有图例项
labels = [line.get_label() for line in lines]
# 在图的右上角添加图例
fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1),
           ncol=5, fancybox=True,frameon=False,)
# 调整子图间距
plt.tight_layout(rect=[0, 0, 1, 1],h_pad=-0)  # 调整图形的整体布局
plt.subplots_adjust(top=0.9)
# 保存图像到文件，设置分辨率为600dpi
fig.savefig(r"D:\Users\weishaosong\PycharmProjects\pressure_prediction/result_plotting/figure_energy_saving_different_threshold_2sd_45scenario.tiff", dpi=600,format="tiff", pil_kwargs={"compression": "tiff_lzw"})
# 显示图形
plt.show()
























