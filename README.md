# GMAP: Generalized Manipulation of Articulated Objects Using Pre-trained Model

> 复现论文: *GMAP: Generalized Manipulation of Articulated Objects in Robotic Using Pre-trained Model* (AAAI 2025)

## 概述

GMAP 系统性地整合了从感知到操控的完整流程：

1. **预训练阶段** — 基于多尺度点云特征提取器 (MSFE) 的 VQ-VAE 掩码预训练
2. **感知阶段** — 部件分割 (Seg-Net)、关节参数估计 (Para-Net)、可操作性预测 (Afford-Net)
3. **操控阶段** — 轨迹规划 + SAPIEN 仿真评估

## 项目结构

```
gmap/
├── configs/                     # YAML 配置文件
│   ├── pretrain.yaml           # VQ-VAE 预训练
│   ├── segnet.yaml             # Seg-Net 微调
│   ├── paranet.yaml            # Para-Net 微调
│   ├── affordnet.yaml          # Afford-Net 微调
│   └── simulation.yaml         # SAPIEN 仿真
├── gmap/
│   ├── models/                  # 模型定义
│   │   ├── msfe.py             # 多尺度特征提取器 (3 尺度 ViT)
│   │   ├── dvae.py             # dVAE 离散化 Tokenizer
│   │   ├── pfe.py              # 点级特征传播
│   │   ├── pretrain.py         # VQ-VAE 预训练模型
│   │   ├── segnet.py           # 部件分割 + 可动性预测
│   │   ├── paranet.py          # 关节参数估计
│   │   ├── affordnet.py        # 可操作性预测
│   │   ├── transformer.py      # ViT Encoder 模块
│   │   └── pointnet2_utils.py  # FPS, KNN, 多尺度分组
│   ├── data/                    # 数据集
│   │   ├── shapenet_dataset.py # ShapeNet55 (预训练)
│   │   ├── partnet_dataset.py  # PartNet-Mobility (下游任务)
│   │   └── transforms.py       # 点云数据增强
│   ├── train/                   # 训练脚本
│   │   ├── train_pretrain.py   # VQ-VAE 预训练
│   │   ├── train_segnet.py     # Seg-Net 训练
│   │   ├── train_paranet.py    # Para-Net 训练
│   │   └── train_affordnet.py  # Afford-Net 训练
│   ├── eval/                    # 评估
│   │   └── metrics.py          # mIoU, 轴误差, 位置误差
│   ├── planner/                 # 轨迹规划
│   │   └── trajectory.py       # 旋转/平移轨迹生成
│   ├── simulation/              # SAPIEN 仿真
│   │   ├── env.py              # 仿真环境
│   │   ├── robot.py            # Panda 机器人控制
│   │   └── evaluate_sim.py     # 端到端评估
│   └── utils/                   # 工具函数
│       ├── logger.py
│       ├── checkpoint.py
│       └── pc_utils.py
├── tests/                       # 单元测试 (33 个)
├── setup.py
└── requirements.txt
```

## 安装

```bash
# 基础依赖
pip install -r requirements.txt

# (可选) PointNet++ CUDA 算子加速
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

# (可选) SAPIEN 仿真器 (仿真评估时需要)
pip install sapien

# 安装项目
pip install -e .
```

> 未安装 pointnet2_ops 时，自动使用纯 PyTorch 实现的 FPS/KNN (速度较慢但功能等价)。

## 数据准备

### ShapeNet55 (预训练)

将 ShapeNet55 数据集处理为 h5 格式放入 `data/ShapeNet55/`:

```
data/ShapeNet55/
├── train.h5    # keys: "data" (N, 8192, 3), "label" (N,)
└── test.h5
```

### PartNet-Mobility (下游任务)

将 PartNet-Mobility 数据放入 `data/PartNetMobility/`:

```
data/PartNetMobility/
├── train.txt           # 训练集物体 ID 列表
├── val.txt             # 验证集物体 ID 列表
├── test.txt            # 测试集物体 ID 列表
└── <object_id>/        # 每个物体一个目录
    ├── point_cloud.npy # (8192, 3) 点云
    ├── seg_label.npy   # (8192,) 部件分割标签
    ├── movable_label.npy # (8192,) 可动性标签
    └── joint_params.json # 关节参数
```

## 训练

### 阶段 1: VQ-VAE 预训练 (ShapeNet, 300 epochs)

```bash
python -m gmap.train.train_pretrain --config configs/pretrain.yaml
```

### 阶段 2: 下游任务微调 (PartNet-Mobility, 各 100 epochs)

```bash
# Seg-Net: 部件分割 + 可动性预测
python -m gmap.train.train_segnet --config configs/segnet.yaml

# Para-Net: 关节参数估计
python -m gmap.train.train_paranet --config configs/paranet.yaml

# Afford-Net: 可操作性预测
python -m gmap.train.train_affordnet --config configs/affordnet.yaml
```

### 阶段 3: SAPIEN 仿真评估

```bash
python -m gmap.simulation.evaluate_sim \
    --config configs/simulation.yaml \
    --segnet_ckpt checkpoints/segnet/epoch_100.pth \
    --paranet_ckpt checkpoints/paranet/epoch_100.pth \
    --affordnet_ckpt checkpoints/affordnet/epoch_100.pth
```

## 模型架构

### MSFE (多尺度特征提取器)

| 尺度 | FPS 中心数 | KNN 邻居数 | ViT 配置 |
|------|-----------|-----------|----------|
| Scale 1 | 512 | 32 | 6 层, 384 维, 6 头 |
| Scale 2 | 256 | 8 | 6 层, 384 维, 6 头 |
| Scale 3 | 64 | 8 | 6 层, 384 维, 6 头 |

### 预训练超参数

| 参数 | 值 |
|------|-----|
| 输入点数 | 8192 |
| 掩码比例 | 60% |
| Codebook 大小 | 8192 |
| Codebook 维度 | 256 |
| 训练轮数 | 300 |
| 学习率 | 1e-3 (cosine + 10 epoch warmup) |
| 优化器 | AdamW (weight_decay=0.05) |

### 评估指标

- **Seg-Net**: mIoU (部件分割), Accuracy (可动性)
- **Para-Net**: 关节类型准确率, 方向误差 (°), 位置误差 (cm), 状态误差
- **仿真**: 7 类物体 (Laptop, Box, Drawer, Door, Faucet, Kettle, Switch) 操控成功率

## 测试

```bash
# 运行全部 33 个单元测试
python -m pytest tests/ -v
```

## 技术栈

- PyTorch >= 1.12
- timm (ViT 实现参考)
- pointnet2_ops (可选 CUDA 加速)
- SAPIEN >= 2.0 (仿真评估)
- scipy, h5py, open3d, einops, tensorboard

## 引用

```bibtex
@inproceedings{zeng2025gmap,
  title={GMAP: Generalized Manipulation of Articulated Objects in Robotic Using Pre-trained Model},
  author={Zeng, H. and Zhang, P. and Li, F. and Yi, Q. and Ye, T. and Wang, J.},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={14},
  pages={14736--14744},
  year={2025}
}
```
