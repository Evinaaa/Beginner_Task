import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from sklearn.metrics import accuracy_score
import time
import argparse
from data_load import load_data
from model import GNN

def train(model, loader, optimizer):
    """
    训练模型
    参数:
        model: 模型实例
        loader: 数据加载器
        optimizer: 优化器
    返回:
        平均训练损失
    """
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        
    return total_loss / len(loader.dataset)

def test(model, loader):
    """
    测试模型
    参数:
        model: 模型实例
        loader: 数据加载器
    返回:
        准确率
    """
    model.eval()
    correct = 0
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
    
    return correct / len(loader.dataset)

def run_experiment(gnn_type, pooling_type, dataset_name='PROTEINS'):
    """
    运行完整实验
    参数:
        gnn_type: GNN类型 ('gcn', 'gat', 'sage', 'gin')
        pooling_type: 池化类型 ('avg', 'max', 'min')
        dataset_name: 数据集名称 (默认'PROTEINS')
    返回:
        测试准确率
    """
    # 加载数据
    train_loader, val_loader, test_loader, num_features, num_classes = load_data(dataset_name)
    
    # 初始化模型
    model = GNN(
        gnn_type=gnn_type,
        pooling_type=pooling_type,
        in_channels=num_features,
        hidden_channels=64,
        out_channels=num_classes,
        num_layers=3
    ).to(device)
    
    # 优化器
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    best_val_acc = 0
    test_acc = 0
    
    for epoch in range(1, 101):
        train_loss = train(model, train_loader, optimizer)
        val_acc = test(model, val_loader)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = test(model, test_loader)
        
        if epoch % 20 == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
                  f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
    
    return test_acc

if __name__ == "__main__":
    # 参数解析
    parser = argparse.ArgumentParser(description='图分类实验')
    parser.add_argument('--dataset', type=str, default='PROTEINS', help='数据集名称')
    parser.add_argument('--gnn', type=str, required=True, choices=['gcn', 'gat', 'sage', 'gin'], help='GNN类型')
    parser.add_argument('--pooling', type=str, required=True, choices=['avg', 'max', 'min'], help='池化类型')
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 运行实验
    start_time = time.time()
    test_acc = run_experiment(args.gnn, args.pooling, args.dataset)
    elapsed = time.time() - start_time
    
    print(f'\n结果: {args.gnn.upper()} with {args.pooling} pooling')
    print(f'测试准确率: {test_acc:.4f}')
    print(f'总时间: {elapsed:.2f}秒')