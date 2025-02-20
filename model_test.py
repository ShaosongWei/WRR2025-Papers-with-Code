import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class DSTP_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(DSTP_RNN, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention1 = nn.MultiheadAttention(hidden_size, num_heads=1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size * 2, hidden_size, num_layers, batch_first=True)
        self.attention2 = nn.MultiheadAttention(hidden_size, num_heads=1, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        lstm_out1, _ = self.lstm1(x)
        attention_out1, _ = self.attention1(lstm_out1, lstm_out1, lstm_out1)
        combined_out1 = torch.cat((lstm_out1, attention_out1), dim=-1)

        lstm_out2, _ = self.lstm2(combined_out1)
        attention_out2, _ = self.attention2(lstm_out2, lstm_out2, lstm_out2)
        combined_out2 = torch.cat((lstm_out2, attention_out2), dim=-1)

        output = self.fc(combined_out2[:, -1, :])
        return output

model_trend = torch.load(r"D:\Users\weishaosong\PycharmProjects\pressure_prediction/model_saved/trend/36_6_1_4_CLD0021.pth")
model_resid = torch.load(r"D:\Users\weishaosong\PycharmProjects\pressure_prediction/model_saved/resid/36_6_1_4_CLD0021.pth")
model_trend.eval()
model_resid.eval()

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('no_noise_filtered_0.6_data_44sensors_trend.pkl', 'rb') as file:
    trend_df = pickle.load(file)
with open('no_noise_filtered_0.6_data_44sensors_seasonal.pkl', 'rb') as file:
    seasonal_df = pickle.load(file)
with open('no_noise_filtered_0.6_data_44sensors_resid.pkl', 'rb') as file:
    resid_df = pickle.load(file)
with open('noise_data_44sensors.pkl', 'rb') as file:
    noise_df = pickle.load(file)

cluster_sensor_name = ['CLD0082', 'CLD0041', 'CLD0034', 'CLD0021']
predicted_sensor_name = cluster_sensor_name[3]     # 代表预测的是几号传感器
cluster_number = "1_3"   # 前一个数字代表传感器类别，后一个数字代表传感器数量
trend_data = trend_df[cluster_sensor_name]
seasonal_data = seasonal_df[cluster_sensor_name]
resid_data = resid_df[cluster_sensor_name]
noise_data = noise_df[cluster_sensor_name]
# 归一化数据
trend_scaler = MinMaxScaler()
seasonal_scaler = MinMaxScaler()
resid_scaler = MinMaxScaler()
noise_scaler = MinMaxScaler()
scaled_trend_data = trend_scaler.fit_transform(trend_data)
scaled_seasonal_data = seasonal_scaler.fit_transform(seasonal_data)
scaled_resid_data = resid_scaler.fit_transform(resid_data)
scaled_noise_data = noise_scaler.fit_transform(noise_data)

# 创建训练集和测试集
train_size = int(len(scaled_trend_data) * 0.7)
val_size = int(len(scaled_trend_data) * 0.1)
test_size = len(scaled_trend_data) - train_size - val_size

train_trend_data, test_trend_data = scaled_trend_data[:train_size], scaled_trend_data[train_size + val_size:]
train_seasonal_data, test_seasonal_data = scaled_seasonal_data[:train_size], scaled_seasonal_data[train_size + val_size:]
train_resid_data, test_resid_data = scaled_resid_data[:train_size], scaled_resid_data[train_size + val_size:]
train_noise_data, test_noise_data = scaled_noise_data[:train_size], scaled_noise_data[train_size + val_size:]

# train_trend_data, test_trend_data = scaled_trend_data[:train_size], scaled_trend_data[train_size:train_size + val_size]
# train_seasonal_data, test_seasonal_data = scaled_seasonal_data[:train_size], scaled_seasonal_data[train_size:train_size + val_size]
# train_resid_data, test_resid_data = scaled_resid_data[:train_size], scaled_resid_data[train_size:train_size + val_size]
# train_noise_data, test_noise_data = scaled_noise_data[:train_size], scaled_noise_data[train_size:train_size + val_size]

# 创建输入输出
def create_dataset(data, time_steps, prediction_steps):
    X, Y = [], []
    for i in range(len(data) - time_steps - prediction_steps + 1):
        X.append(data[i:(i + time_steps), :])
        Y.append(data[(i + time_steps):(i + time_steps + prediction_steps), -1])  # -1代表预测4号传感器的数据
    return np.array(X), np.array(Y)

prediction_steps = 6   # 预测未来多少个时间步
time_steps = 36        # 使用过去多少个时间步的数据
learning_rate = 0.0005
batch_size = 32

X_train_trend, y_train_trend = create_dataset(train_trend_data, time_steps, prediction_steps)
X_test_trend, y_test_trend = create_dataset(test_trend_data, time_steps, prediction_steps)
X_train_seasonal, y_train_seasonal = create_dataset(train_seasonal_data, time_steps, prediction_steps)
X_test_seasonal, y_test_seasonal = create_dataset(test_seasonal_data, time_steps, prediction_steps)
X_train_resid, y_train_resid = create_dataset(train_resid_data, time_steps, prediction_steps)
X_test_resid, y_test_resid = create_dataset(test_resid_data, time_steps, prediction_steps)
X_train_noise, y_train_noise = create_dataset(train_noise_data, time_steps, prediction_steps)
X_test_noise, y_test_noise = create_dataset(test_noise_data, time_steps, prediction_steps)

# 转换为 PyTorch 张量
X_train_trend_tensor = torch.tensor(X_train_trend, dtype=torch.float32).to(device)
y_train_trend_tensor = torch.tensor(y_train_trend, dtype=torch.float32).to(device)
X_test_trend_tensor = torch.tensor(X_test_trend, dtype=torch.float32).to(device)
y_test_trend_tensor = torch.tensor(y_test_trend, dtype=torch.float32).to(device)
X_train_resid_tensor = torch.tensor(X_train_resid, dtype=torch.float32).to(device)
y_train_resid_tensor = torch.tensor(y_train_resid, dtype=torch.float32).to(device)
X_test_resid_tensor = torch.tensor(X_test_resid, dtype=torch.float32).to(device)
y_test_resid_tensor = torch.tensor(y_test_resid, dtype=torch.float32).to(device)

# 创建 DataLoader
train_trend_dataset = TensorDataset(X_train_trend_tensor, y_train_trend_tensor)
test_trend_dataset = TensorDataset(X_test_trend_tensor, y_test_trend_tensor)
train_resid_dataset = TensorDataset(X_train_resid_tensor, y_train_resid_tensor)
test_resid_dataset = TensorDataset(X_test_resid_tensor, y_test_resid_tensor)

train_trend_loader = DataLoader(train_trend_dataset, batch_size=batch_size, shuffle=True)
test_trend_loader = DataLoader(test_trend_dataset, batch_size=batch_size, shuffle=False)
train_resid_loader = DataLoader(train_resid_dataset, batch_size=batch_size, shuffle=True)
test_resid_loader = DataLoader(test_resid_dataset, batch_size=batch_size, shuffle=False)
criterion = nn.MSELoss()

model_trend.eval()
with torch.no_grad():
    test_loss = 0
    for X_batch, y_batch in test_trend_loader:
        outputs = model_trend(X_batch)
        test_loss += criterion(outputs, y_batch).item()
    test_loss /= len(test_trend_loader)

print(f'Test Loss: {test_loss:.4f}')

model_resid.eval()
with torch.no_grad():
    test_loss = 0
    for X_batch, y_batch in test_resid_loader:
        outputs = model_resid(X_batch)
        test_loss += criterion(outputs, y_batch).item()
    test_loss /= len(test_resid_loader)

print(f'Test Loss: {test_loss:.4f}')

# 预测
model_trend.eval()
with torch.no_grad():
    y_trend_pred = []
    for X_batch, _ in test_trend_loader:
        outputs = model_trend(X_batch)
        y_trend_pred.append(outputs)

    y_trend_pred = torch.cat(y_trend_pred, dim=0).cpu().numpy()

model_resid.eval()
with torch.no_grad():
    y_resid_pred = []
    for X_batch, _ in test_resid_loader:
        outputs = model_resid(X_batch)
        y_resid_pred.append(outputs)

    y_resid_pred = torch.cat(y_resid_pred, dim=0).cpu().numpy()

# 反归一化预测结果
selected_rows_trend_test =y_test_trend[::prediction_steps]
y_test_trend = selected_rows_trend_test.ravel()
selected_rows_seasonal_test =y_test_seasonal[::prediction_steps]
y_test_seasonal = selected_rows_seasonal_test.ravel()
selected_rows_resid_test =y_test_resid[::prediction_steps]
y_test_resid = selected_rows_resid_test.ravel()

selected_rows_trend_pred =y_trend_pred[::prediction_steps]
y_pred_trend = selected_rows_trend_pred.ravel()
selected_rows_resid_pred =y_resid_pred[::prediction_steps]
y_pred_resid = selected_rows_resid_pred.ravel()

noise_y_test = y_test_noise[::prediction_steps]
y_test_noise = noise_y_test.ravel()

y_test_trend_rescaled = trend_scaler.inverse_transform(np.concatenate((np.zeros((len(y_test_trend), 3)), y_test_trend.reshape(-1, 1)), axis=1))[:, -1]
y_test_seasonal_rescaled = seasonal_scaler.inverse_transform(np.concatenate((np.zeros((len(y_test_seasonal), 3)), y_test_seasonal.reshape(-1, 1)), axis=1))[:, -1]
y_test_resid_rescaled = resid_scaler.inverse_transform(np.concatenate((np.zeros((len(y_test_resid), 3)), y_test_resid.reshape(-1, 1)), axis=1))[:, -1]
y_pred_trend_rescaled = trend_scaler.inverse_transform(np.concatenate((np.zeros((len(y_pred_trend), 3)), y_pred_trend.reshape(-1, 1)), axis=1))[:, -1]
y_pred_resid_rescaled = resid_scaler.inverse_transform(np.concatenate((np.zeros((len(y_pred_resid), 3)), y_pred_resid.reshape(-1, 1)), axis=1))[:, -1]
y_test_noise_rescaled = noise_scaler.inverse_transform(np.concatenate((np.zeros((len(y_test_noise), 3)), y_test_noise.reshape(-1, 1)), axis=1))[:, -1]

y_test_rescaled = y_test_trend_rescaled + y_test_seasonal_rescaled + y_test_resid_rescaled
y_pred_rescaled = y_pred_trend_rescaled + y_test_seasonal_rescaled + y_pred_resid_rescaled

mae_resid = mean_absolute_error(y_test_resid_rescaled, y_pred_resid_rescaled)
print("Mean Absolute Error (MAE):", mae_resid)

# 打印结果
print('传感器数量44，连续5个步长预测结果，输入采用历史数据')
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
print("Mean Squared Error (MSE):", mse)
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
print("Mean Absolute Error (MAE):", mae)
r2 = r2_score(y_test_rescaled, y_pred_rescaled)
print("R² Score:", r2)

print('传感器数量44，连续5个步长预测结果，输入采用历史数据')
noise_mse = mean_squared_error(y_pred_rescaled, y_test_noise_rescaled)
print("Mean Squared Error (MSE):", noise_mse)
noise_mae = mean_absolute_error(y_pred_rescaled, y_test_noise_rescaled)
print("Mean Absolute Error (MAE):", noise_mae)
noise_r2 = r2_score(y_pred_rescaled, y_test_noise_rescaled)
print("R² Score:", noise_r2)

# 指定误差限值
error_limit = 0.1
errors = np.abs(y_test_rescaled - y_pred_rescaled)
below_limit = errors < error_limit
ratio_below_limit = np.sum(below_limit)/len(y_test_rescaled)
print("误差小于0.1的数据占比:",ratio_below_limit)

error_limit = 0.1
errors = np.abs(y_pred_rescaled - y_test_noise_rescaled)
below_limit = errors < error_limit
ratio_below_limit = np.sum(below_limit)/len(y_test_rescaled)
print("噪声误差小于0.1的数据占比:",ratio_below_limit)

error_limit = 0.2
errors = np.abs(y_test_rescaled - y_pred_rescaled)
below_limit = errors < error_limit
ratio_below_limit = np.sum(below_limit)/len(y_test_rescaled)
print("误差小于0.2的数据占比:",ratio_below_limit)

error_limit = 0.2
errors = np.abs(y_pred_rescaled - y_test_noise_rescaled)
below_limit = errors < error_limit
ratio_below_limit = np.sum(below_limit)/len(y_test_rescaled)
print("噪声误差小于0.2的数据占比:",ratio_below_limit)

error_limit = 0.3
errors = np.abs(y_test_rescaled - y_pred_rescaled)
below_limit = errors < error_limit
ratio_below_limit = np.sum(below_limit)/len(y_test_rescaled)
print("误差小于0.3的数据占比:",ratio_below_limit)

error_limit = 0.3
errors = np.abs(y_pred_rescaled - y_test_noise_rescaled)
below_limit = errors < error_limit
ratio_below_limit = np.sum(below_limit)/len(y_test_rescaled)
print("噪声误差小于0.3的数据占比:",ratio_below_limit)

error_limit = 0.4
errors = np.abs(y_test_rescaled - y_pred_rescaled)
below_limit = errors < error_limit
ratio_below_limit = np.sum(below_limit)/len(y_test_rescaled)
print("误差小于0.4的数据占比:",ratio_below_limit)

error_limit = 0.4
errors = np.abs(y_pred_rescaled - y_test_noise_rescaled)
below_limit = errors < error_limit
ratio_below_limit = np.sum(below_limit)/len(y_test_rescaled)
print("噪声误差小于0.4的数据占比:",ratio_below_limit)

error_limit = 0.5
errors = np.abs(y_test_rescaled - y_pred_rescaled)
below_limit = errors < error_limit
ratio_below_limit = np.sum(below_limit)/len(y_test_rescaled)
print("误差小于0.5的数据占比:",ratio_below_limit)

error_limit = 0.5
errors = np.abs(y_pred_rescaled - y_test_noise_rescaled)
below_limit = errors < error_limit
ratio_below_limit = np.sum(below_limit)/len(y_test_rescaled)
print("噪声误差小于0.5的数据占比:",ratio_below_limit)

