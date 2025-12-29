# 模型训练和使用指南

## 训练模型

### IEMOCAP数据集训练
```bash
cd /path/to/CGGM
python run/iemo_run.py --data_path /path/to/iemocap/data --model_save_path checkpoints/iemo_best_model.pth
```

### MOSI数据集训练
```bash
cd /path/to/CGGM
python run/mosi_run.py --data_path /path/to/mosi/data --model_save_path checkpoints/mosi_best_model.pth
```

## 使用训练好的模型进行推理

### IEMOCAP推理
```bash
python run/inference.py --model_path checkpoints/iemo_best_model.pth --dataset iemo --data_path /path/to/iemocap/data
```

### MOSI推理
```bash
python run/inference.py --model_path checkpoints/mosi_best_model.pth --dataset mosi --data_path /path/to/mosi/data
```

## 主要修改

1. **模型保存功能**: 训练过程中自动保存最佳模型
2. **模型加载工具**: 提供便捷的模型加载函数
3. **推理脚本**: 可用于测试训练好的模型
4. **Checkpoints目录**: 自动创建用于保存模型

## 模型保存内容

保存的模型文件包含：
- `model_state_dict`: 主模型参数
- `optimizer_state_dict`: 优化器状态
- `classifier_state_dict`: 分类器参数（如果使用CGGM）
- `cls_optimizer_state_dict`: 分类器优化器状态（如果使用CGGM）
- `epoch`: 训练轮数
- `best_acc`: 最佳性能
- `hyp_params`: 所有超参数配置

## 参数说明

### 通用参数
- `--model_save_path`: 模型保存路径（默认：checkpoints/dataset_best_model.pth）
- `--data_path`: 数据集路径
- `--batch_size`: 批次大小
- `--num_epochs`: 训练轮数
- `--lr`: 学习率
- `--modulation`: 调制策略（none/cggm）

### 数据集特定参数

#### IEMOCAP
- 输出维度：4（情感类别）
- 损失函数：CrossEntropyLoss
- 默认模型：checkpoints/iemo_best_model.pth

#### MOSI  
- 输出维度：1（情感分数）
- 损失函数：L1Loss
- 默认模型：checkpoints/mosi_best_model.pth