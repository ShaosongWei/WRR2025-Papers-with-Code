import pandas as pd
import numpy as np
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils.metric import distance_metric, type_metric


# 读取湖州管网中用于校准的压力数据
origin_pressure_cal = pd.read_csv('pressureData_1.csv')
# 将time列转换为datetime格式
origin_pressure_cal['time'] = pd.to_datetime(origin_pressure_cal['time'])
# 将'time'列设置为索引
origin_pressure_cal.set_index('time', inplace=True)
# 读取湖州管网中用于验证的压力数据
origin_pressure_val = pd.read_csv('pressureData_2.csv')
# 将time列转换为datetime格式
origin_pressure_val['time'] = pd.to_datetime(origin_pressure_val['time'])
# 将'time'列设置为索引
origin_pressure_val.set_index('time', inplace=True)
# 将校准传感器和验证传感器按列进行拼接
origin_pressure = pd.concat([origin_pressure_cal, origin_pressure_val], axis=1)
# 读取上述传感器中有爆管发生的传感器id
sensors_to_delete = pd.read_csv('burst_sensor_id.csv')
# 将有爆管发生的传感器id删除
sensors_to_delete_list = sensors_to_delete['sensor_name'].tolist()
origin_pressure.drop(columns=sensors_to_delete_list, inplace=True)

# 设置传感器聚类的数目
cluster_num = 4
# 选择使用前多少步长的数据进行聚类
MOD_measure = origin_pressure.iloc[:672,:]
# 提取传感器的名称列表
sensor_name = MOD_measure.columns.values.tolist()  # 监测点名称列表

# 提取监测数据的均值和方差
means = MOD_measure.mean(axis=0)
data_mean = means.tolist()
var = MOD_measure.var(axis=0)
data_var = var.tolist()
data_var = [round(i, 5) for i in data_var]
data_mean = [round(i, 5) for i in data_mean]
# 将均值和方差组合成一个特征矩阵
MOD_meas_vm = pd.DataFrame({'mean': data_mean, 'var': data_var}, index=sensor_name)  # 用于进行距离矩阵计算的矩阵信息
input_data = MOD_meas_vm.values.tolist()  # 导入数据为监测数据的平均值和方差
# 计算两个传感器之间的相关系数
def corr_distance_id(x, y):
    x_name = MOD_meas_vm[(MOD_meas_vm['mean'] == x[0]) & (MOD_meas_vm['var'] == x[1])].index.tolist()
    y_name = MOD_meas_vm[(MOD_meas_vm['mean'] == y[0]) & (MOD_meas_vm['var'] == y[1])].index.tolist()
    x_name = x_name[0]
    y_name = y_name[0]
    N = MOD_measure.shape[0]
    x_list = MOD_measure[x_name].values.tolist()
    y_list = MOD_measure[y_name].values.tolist()
    xy = [a * b for a, b in zip(x_list, y_list)]
    mean_xy = np.mean(xy)  # E(xy)
    part1 = mean_xy - (MOD_meas_vm.at[x_name, 'mean'] * MOD_meas_vm.at[y_name, 'mean'])  # COV(X,Y)=E(XY)-E(X)E(Y)
    part2 = (MOD_meas_vm.at[x_name, 'var'] * MOD_meas_vm.at[y_name, 'var']) ** 0.5  # 分母=sqrt[D(X)D(Y)]
    corr = part1 / part2  # CORR = COV(X,Y)/sqrt[D(X)D(Y)]
    return 1 - corr


# 定义聚类的距离为自定义的相关系数
metric = distance_metric(type_metric.USER_DEFINED, func=corr_distance_id)
per_class_size = []
initial_centers = kmeans_plusplus_initializer(input_data, cluster_num).initialize(return_index=True)  # 初始化形心
kmedoids_instance = kmedoids(input_data, initial_centers, metric=metric)  # 实例化kmedoids类
kmedoids_instance.process()  # 训练
clusters = kmedoids_instance.get_clusters()  # 归类
medoids = kmedoids_instance.get_medoids()  # 返回形心


# 将聚类结果保存到 DataFrame 中
clustered_sensors = pd.DataFrame(index=MOD_measure.columns, columns=['Cluster'])

for cluster_id, cluster in enumerate(clusters):
    for sensor_index in cluster:
        clustered_sensors.iloc[sensor_index] = cluster_id

# 保存结果到 CSV 文件
clustered_sensors.to_csv('clustered_sensors.csv')

# 打印聚类中心（medoids）及其对应的传感器名称
medoid_names = [MOD_measure.columns[medoid] for medoid in medoids]
print("Medoids (cluster centers) and their corresponding sensor names:")
for medoid, name in zip(medoids, medoid_names):
    print(f"Medoid index: {medoid}, Sensor name: {name}")

# 提取标签为0的所有传感器名称
sensors_with_label_0 = clustered_sensors[clustered_sensors['Cluster'] == 0].index.tolist()
print("Sensors with label 0:", sensors_with_label_0)

# 提取标签为1的所有传感器名称
sensors_with_label_1 = clustered_sensors[clustered_sensors['Cluster'] == 1].index.tolist()
print("Sensors with label 1:", sensors_with_label_1)

# 提取标签为2的所有传感器名称
sensors_with_label_2 = clustered_sensors[clustered_sensors['Cluster'] == 2].index.tolist()
print("Sensors with label 2:", sensors_with_label_2)

# 提取标签为3的所有传感器名称
sensors_with_label_3 = clustered_sensors[clustered_sensors['Cluster'] == 3].index.tolist()
print("Sensors with label 3:", sensors_with_label_3)


