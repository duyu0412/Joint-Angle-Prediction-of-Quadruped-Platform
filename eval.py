import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
from mlp import LinkageNet, load_data, compute_end_effector_position, load_config
import yaml

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# 加载配置
config = load_config()

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
    error= np.mean(np.sqrt((x_9_pre - x_9_true)**2 + (y_9_pre - y_9_true)**2))
    plt.figure(figsize=(10,6))
    plt.plot(x_9_true, y_9_true, 'o')
    plt.plot(x_9_pre, y_9_pre, 'x')
    plt.title("End Effector Trajectory, Mean Position Error: {:.4f}".format(error))
    plt.ylabel("Y Position")
    plt.xlabel("X Position")
    plt.legend(['True Trajectory', 'Predicted Trajectory'])
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def main():
    if os.path.exists(model_path):
        model = LinkageNet(input_shape, output_shape).to(device)
        model.load_state_dict(torch.load(model_path))
        print("Loaded pre-trained model.")

    x_norm, y_norm, _, _, y_min, y_max = load_data(input_file, output_file)
    x_tensor = torch.tensor(x_norm, dtype=torch.float32).to(device)
    y_pred_norm = model(x_tensor).detach().cpu().numpy()
    y_pred = y_pred_norm * (y_max - y_min) + y_min
    y_true = y_norm * (y_max - y_min) + y_min
    plot_position_trajectories(y_pred, y_true)

if __name__ == "__main__":
    main()

