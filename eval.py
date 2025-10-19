import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from mlp import LinkageNet, load_data, compute_end_effector_position, load_config

# 加载配置
config = load_config()

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 文件路径配置
input_file = config['files']['input_file']
output_file = config['files']['output_file']
model_path = config['files']['best_model_path']

# 模型参数
input_shape = config['model']['input_shape']
output_shape = config['model']['output_shape']


def plot_position_trajectories(y_pred, y_true):
    """绘制末端执行器位置轨迹对比图"""
    x_9_pre, y_9_pre = compute_end_effector_position(y_pred)
    x_9_true, y_9_true = compute_end_effector_position(y_true)
    error = np.mean(np.sqrt((x_9_pre - x_9_true)**2 + (y_9_pre - y_9_true)**2))
    plt.figure(figsize=(10,6))
    plt.plot(x_9_true, y_9_true, 'o-', label='True Trajectory')
    plt.plot(x_9_pre, y_9_pre, 'x--', label='Predicted Trajectory')
    plt.title(f"End Effector Trajectory, Mean Position Error: {error:.4f}")
    plt.ylabel("Y Position")
    plt.xlabel("X Position")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


def main():
    # 构建模型并加载权重
    net = LinkageNet(input_shape, output_shape).to(device)
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded pre-trained model.")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # 加载数据并做归一化
    x_norm, y_norm, _, _, y_min, y_max = load_data(input_file, output_file)
    x_tensor = torch.tensor(x_norm, dtype=torch.float32).to(device)

    net.eval()
    with torch.no_grad():
        y_pred_norm = net(x_tensor).detach().cpu().numpy()

    # 反归一化
    y_pred = y_pred_norm * (y_max - y_min) + y_min
    y_true = y_norm * (y_max - y_min) + y_min

    plot_position_trajectories(y_pred, y_true)


if __name__ == "__main__":
    main()
