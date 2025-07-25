import os
import torch
import torch.optim as optim
from torch_geometric.loader import LinkNeighborLoader
from model import SAGELinkPred
from data_load import load_data
from utils import plot_results, test

# 创建结果目录
os.makedirs('./GCN/results', exist_ok=True)

# 训练参数
EPOCHS = 100
HIDDEN_DIM = 128
OUT_DIM = 64
LR = 0.01
BATCH_SIZE = 256
NEIGHBORS = [20, 10]
MODES = ['full', 'sampling']

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def train_full_graph(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    
    pos_pred = model(data.x, data.train_pos_edge_index, data.train_pos_edge_index)
    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1)
    )
    neg_pred = model(data.x, data.train_pos_edge_index, neg_edge_index)
    
    pos_loss = F.binary_cross_entropy_with_logits(pos_pred, torch.ones_like(pos_pred))
    neg_loss = F.binary_cross_entropy_with_logits(neg_pred, torch.zeros_like(neg_pred))
    loss = pos_loss + neg_loss
    loss.backward()
    optimizer.step()
    return loss.item()

def train_sampling(model, loader, optimizer):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        
        pos_edges = batch.edge_label_index[:, batch.edge_label == 1]
        neg_edges = batch.edge_label_index[:, batch.edge_label == 0]
        
        if pos_edges.size(1) == 0 or neg_edges.size(1) == 0:
            continue
            
        pos_pred = model(batch.x, batch.edge_index, pos_edges)
        neg_pred = model(batch.x, batch.edge_index, neg_edges)
        
        pos_loss = F.binary_cross_entropy_with_logits(pos_pred, torch.ones_like(pos_pred))
        neg_loss = F.binary_cross_entropy_with_logits(neg_pred, torch.zeros_like(neg_pred))
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

def main():
    data = load_data().to(device)
    print("Cora dataset loaded:")
    print(f"Nodes: {data.num_nodes}, Features: {data.num_features}")
    print(f"Training edges: {data.train_pos_edge_index.size(1)}")
    
    for mode in MODES:
        print(f"\n===== Training GCN in {mode.upper()} mode =====")
        model = SAGELinkPred(data.num_features, HIDDEN_DIM, OUT_DIM).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        
        train_losses = []
        val_aucs = []
        val_aps = []
        best_val_auc = 0
        
        # 采样模式设置
        if mode == 'sampling':
            loader = LinkNeighborLoader(
                data,
                num_neighbors=NEIGHBORS,
                batch_size=BATCH_SIZE,
                edge_label_index=data.train_pos_edge_index,
                edge_label=torch.ones(data.train_pos_edge_index.size(1)),
                negative_sampling_ratio=1.0,
                shuffle=True
            )
        
        for epoch in range(1, EPOCHS + 1):
            start_time = time.time()
            
            if mode == 'full':
                loss = train_full_graph(model, data, optimizer)
            else:
                loss = train_sampling(model, loader, optimizer)
            
            train_losses.append(loss)
            
            # 验证集评估
            val_edges = torch.cat([data.val_pos_edge_index, data.val_neg_edge_index], dim=1)
            val_labels = torch.cat([torch.ones(data.val_pos_edge_index.size(1)), 
                                  torch.zeros(data.val_neg_edge_index.size(1))])
            val_auc, val_ap = test(model, data, val_edges, val_labels)
            val_aucs.append(val_auc)
            val_aps.append(val_ap)
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), f'./GCN/results/best_{mode}_model.pt')
            
            epoch_time = time.time() - start_time
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}, Time: {epoch_time:.2f}s')
        
        # 测试集评估
        model.load_state_dict(torch.load(f'./GCN/results/best_{mode}_model.pt'))
        test_edges = torch.cat([data.test_pos_edge_index, data.test_neg_edge_index], dim=1)
        test_labels = torch.cat([torch.ones(data.test_pos_edge_index.size(1)), 
                               torch.zeros(data.test_neg_edge_index.size(1))])
        test_auc, test_ap = test(model, data, test_edges, test_labels)
        print(f"\n=== GCN {mode.upper()} Mode Test Results ===")
        print(f"Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}")
        
        # 绘制结果
        plot_results(train_losses, val_aucs, val_aps, 'GCN', mode)

if __name__ == "__main__":
    main()