import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(dataset_name='PROTEINS', batch_size=32):
    """
    加载图数据集并划分训练/验证/测试集
    参数:
        dataset_name: TUDataset数据集名称 (默认'PROTEINS')
        batch_size: 批次大小 (默认32)
    返回:
        train_loader, val_loader, test_loader, num_features, num_classes
    """
    # 加载数据集
    dataset = TUDataset(root=f'data/{dataset_name}', name=dataset_name)
    
    # 数据集统计信息
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    
    # 划分训练/验证/测试集 (60/20/20)
    indices = np.arange(len(dataset))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.25, random_state=42)  # 0.25*0.8=0.2
    
    # 创建数据集子集
    train_dataset = dataset[train_idx.tolist()]
    val_dataset = dataset[val_idx.tolist()]
    test_dataset = dataset[test_idx.tolist()]
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader, num_features, num_classes