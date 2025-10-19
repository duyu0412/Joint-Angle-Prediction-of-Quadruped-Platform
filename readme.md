# 四足单元连杆机构神经网络预测系统

## 项目简介

本项目用于建立四足平台单个曲柄对应四个髋关节、膝关节的角度映射关系。通过深度学习神经网络模型，实现从输入角度到多关节角度的精确预测，为四足机器人的运动控制提供智能化解决方案。

## 功能特点

- 🔧 **连杆机构建模**: 基于机械原理的四足机器人连杆系统数学建模
- 🧠 **深度学习预测**: 使用多层感知器(MLP)神经网络进行角度映射
# 四足机器人连杆机构神经网络预测系统

## 项目简介

本项目用于建立四足平台单个曲柄对应四个髋关节、膝关节的角度映射关系。通过深度学习神经网络模型，实现从输入角度到多关节角度的精确预测，为四足机器人的运动控制提供智能化解决方案。

## 功能特点

- 🔧 **连杆机构建模**: 基于机械原理的四足机器人连杆系统数学建模
- 🧠 **深度学习预测**: 使用多层感知器(MLP)神经网络进行角度映射
- ⚡ **GPU加速训练**: 支持CUDA加速，提升训练效率
- 📊 **数据可视化**: 实时显示训练结果和预测精度
- ⚙️ **配置化管理**: 使用YAML配置文件管理所有参数
- 💾 **模型持久化**: 支持模型保存和加载，便于部署使用

## 项目结构

```
tanma_predict/
├── config.yml                          # 配置文件
├── mlp.py                              # 主程序文件
├── eval.py                             # 模型评估程序
├── input.csv                           # 输入数据
├── output.csv                          # 输出标签数据
├── linkage_net_overfit_gpu.pth         # 训练模型权重
├── linkage_net_overfit_gpu_best.pth    # 最佳模型权重
└── readme.md                           # 项目说明文档
```
# 四足机器人连杆机构神经网络预测系统

## 项目简介

本项目用于建立四足平台单个曲柄对应四个髋关节、膝关节的角度映射关系。通过深度学习神经网络模型，实现从输入角度到多关节角度的精确预测，为四足机器人的运动控制提供智能化解决方案。

## 功能特点

- 🔧 **连杆机构建模**: 基于机械原理的四足机器人连杆系统数学建模
- 🧠 **深度学习预测**: 使用多层感知器(MLP)神经网络进行角度映射
- ⚡ **GPU加速训练**: 支持CUDA加速，提升训练效率
- 📊 **数据可视化**: 实时显示训练结果和预测精度
- ⚙️ **配置化管理**: 使用YAML配置文件管理所有参数
- 💾 **模型持久化**: 支持模型保存和加载，便于部署使用

## 项目结构

```
tanma_predict/
├── config.yml                          # 配置文件
├── mlp.py                              # 主程序文件
├── eval.py                             # 模型评估程序
├── input.csv                           # 输入数据
├── output.csv                          # 输出标签数据
├── linkage_net_overfit_gpu.pth         # 训练模型权重
├── linkage_net_overfit_gpu_best.pth    # 最佳模型权重
└── readme.md                           # 项目说明文档
```

## 安装依赖

确保您的环境中安装了以下依赖包：

```bash
pip install torch torchvision numpy pandas matplotlib pyyaml
```

### 系统要求

- Python 3.9+
- PyTorch 2.0+
- CUDA支持（可选，用于GPU加速）

## 配置说明

项目使用`config.yml`配置文件管理所有参数：

### 文件路径配置
```yaml
files:
  input_file: 'input.csv'      # 输入数据文件
  output_file: 'output.csv'    # 输出标签文件  
  model_path: "linkage_net_overfit_gpu_best.pth"  # 模型权重文件
```

### 连杆机构参数
```yaml
linkage:
  r0: 84          # 连杆0长度
  r3: 60          # 连杆3长度
  r6: 50.53       # 连杆6长度
  r8: 53.7        # 连杆8长度
  r9: 140         # 连杆9长度
  theta0_deg: 35  # 初始角度θ0 (度)
  theta3_6_deg: 15 # 角度偏移θ3-6 (度)
  theta8_9_deg: 30 # 角度偏移θ8-9 (度)
```

### 模型参数
```yaml
model:
  input_shape: 1   # 输入维度
  output_shape: 8  # 输出维度（8个关节角度）
```

## 使用方法

### 1. 数据准备

确保`input.csv`和`output.csv`文件在项目根目录：
- `input.csv`: 包含输入角度数据（单列）
- `output.csv`: 包含对应的8个关节角度标签（8列）
- 0、4列表示第一条前腿腿的髋关节、膝关节旋转角度，1、5列是同侧后腿，2、6列是对侧前腿，3、7列是对侧后腿

### 2. 训练模型

```bash
python mlp.py
```

程序将：
- 自动检测并使用GPU（如果可用）
- 加载配置参数和数据
- 执行数据归一化预处理
- 训练深度神经网络模型
- 保存训练好的模型权重
- 显示训练过程可视化结果

### 3. 模型评估，用于比较足端轨迹的误差

```bash
python eval.py
```

## 模型架构

### LinkageNet神经网络

```
输入层 (1) → 线性层 (128) → ReLU → 
线性层 (256) → ReLU → 线性层 (256) → ReLU → 
线性层 (128) → ReLU → 输出层 (8)
```

- **激活函数**: ReLU
- **损失函数**: MSE (均方误差)
- **优化器**: Adam (学习率: 5e-5)
- **训练轮数**: 200,000 epochs

## 核心算法

### 末端执行器位置计算

系统通过连杆机构的几何关系计算末端执行器位置：

```python
def compute_end_effector_position(y_pred):
    theta3_true = theta0 + y_pred[:,0]
    theta6_true = theta3_true + theta3_6  
    theta8_true = theta6_true + y_pred[:,4]
    theta9_true = theta8_true + theta8_9
    
    x_9_true = r0*cos(theta0) + r3*cos(theta3_true) + 
               r6*cos(theta6_true) + r9*cos(theta9_true)
    y_9_true = r0*sin(theta0) + r3*sin(theta3_true) + 
               r6*sin(theta6_true) + r9*sin(theta9_true)
    return x_9_true, y_9_true
```

## 训练参数

- **批次大小**: 全量数据训练
- **学习率**: 5e-8 (自适应调整)
- **训练轮数**: 20,000
- **数据归一化**: Min-Max归一化 [0,1]
- **设备**: 自动检测GPU/CPU
- **预训练模型**: pretrain_model_path: "linkage_net_overfit_gpu.pth"

## 输出结果

训练完成后，程序将：

1. **保存模型**: `linkage_net_overfit_gpu_best.pth`
2. **显示可视化**: 真实值vs预测值对比图
3. **打印训练日志**: 每1000轮显示损失值
4. **计算精度指标**: MSE损失和预测准确度


## 贡献指南

欢迎提交Issue和Pull Request来改进项目。请确保：

1. 代码符合PEP8规范
2. 添加必要的注释和文档
3. 测试新功能的兼容性

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 联系方式

如有技术问题或建议，请通过以下方式联系：

- 创建GitHub Issue
- 发送邮件至项目维护者

---

*最后更新: 2025年10月19日*