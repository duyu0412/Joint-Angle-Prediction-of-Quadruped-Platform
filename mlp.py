import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载配置文件
def load_config(config_path='config.yml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

# 加载配置
config = load_config()

# 文件路径配置
input_file = config['files']['input_file']
output_file = config['files']['output_file']
model_path = config['files']['pretrain_model_path']

# 连杆参数
r0 = config['linkage']['r0']
r3 = config['linkage']['r3']
r6 = config['linkage']['r6']
r8 = config['linkage']['r8']
r9 = config['linkage']['r9']
theta0 = config['linkage']['theta0_deg'] * np.pi / 180
theta3_6 = config['linkage']['theta3_6_deg'] * np.pi / 180
theta8_9 = config['linkage']['theta8_9_deg'] * np.pi / 180

# 模型参数
input_shape = config['model']['input_shape']
output_shape = config['model']['output_shape']

# 训练参数
lr = config['training']['learning_rate']
epochs = config['training']['epochs']
# 2. 定义模型
class LinkageNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinkageNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )
    def forward(self, x):
        return self.model(x)

def load_data(input_file, output_file):
    # 1. 读取数据
    x = pd.read_csv(input_file, header=None).values.astype(np.float32)
    y = pd.read_csv(output_file, header=None).values.astype(np.float32)

    # 数据归一化
    x_min = x.min(axis=0)
    x_max = x.max(axis=0)
    y_min = y.min(axis=0)
    y_max = y.max(axis=0)
    x_norm = (x - x_min) / (x_max - x_min)
    y_norm = (y - y_min) / (y_max - y_min)
    return x_norm, y_norm, x_min, x_max, y_min, y_max
def compute_end_effector_position(y_pred):
    theta3_true = theta0 + y_pred[:,0]
    theta6_true = theta3_true + theta3_6
    theta8_true = theta6_true + y_pred[:,4]
    theta9_true = theta8_true + theta8_9
    x_9_true = r0 * np.cos(theta0) + r3 * np.cos(theta3_true) + r6 * np.cos(theta6_true) + r9 * np.cos(theta9_true)
    y_9_true = r0 * np.sin(theta0) + r3 * np.sin(theta3_true) + r6 * np.sin(theta6_true) + r9 * np.sin(theta9_true)
    return x_9_true, y_9_true
def main():
    x_norm, y_norm, _, _, y_min, y_max = load_data(input_file, output_file)

    # 转换为tensor并放入GPU
    x_train = torch.tensor(x_norm, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_norm, dtype=torch.float32).to(device)

    # model = LinkageNet(x_train.shape[1], y_train.shape[1]).to(device)
    model_path = "linkage_net_overfit_gpu.pth"
    # 如果存在预训练模型，则加载
    if os.path.exists(model_path):
        model = LinkageNet(input_shape, output_shape).to(device)
        model.load_state_dict(torch.load(model_path))
        print("Loaded pre-trained model.")
    else:
        model = LinkageNet(input_shape, output_shape).to(device)
    # 3. 定义训练参数
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4. 训练
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {loss.item():.8f}")

    # 5. 预测与反归一化
    model.eval()
    with torch.no_grad():
        y_pred_norm = model(x_train).detach().cpu().numpy()  # 注意：取出时要转回CPU

    y_pred = y_pred_norm * (y_max - y_min) + y_min
    y_true = y_train.detach().cpu().numpy() * (y_max - y_min) + y_min

    # 6. 可视化
    plt.figure(figsize=(10,6))
    plt.plot(y_true[:,0], label='True Link1')
    plt.plot(y_pred[:,0], '--', label='Pred Link1')
    plt.legend()
    plt.title("Overfit Check: Link1 Angle Prediction (GPU)")
    plt.show()

    # 7. 保存模型
    torch.save(model.state_dict(), "linkage_net_overfit_gpu_best.pth")
    print("Model saved successfully.")

if __name__ == "__main__":
    main()
