import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_min_pool
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, ModuleList

class GNN(torch.nn.Module):
    def __init__(self, gnn_type, pooling_type, in_channels, hidden_channels, out_channels, num_layers=3):
        """
        图神经网络模型
        
        参数:
            gnn_type: GNN类型 ('gcn', 'gat', 'sage', 'gin')
            pooling_type: 池化类型 ('avg', 'max', 'min')
            in_channels: 输入特征维度
            hidden_channels: 隐藏层维度
            out_channels: 输出类别数
            num_layers: GNN层数 (默认3)
        """
        super().__init__()
        self.gnn_type = gnn_type
        self.pooling_type = pooling_type
        self.convs = ModuleList()
        self.bns = ModuleList()
        
        # 创建GNN层
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            out_dim = hidden_channels if i < num_layers - 1 else hidden_channels
            
            if gnn_type == 'gcn':
                conv = GCNConv(in_dim, out_dim)
            elif gnn_type == 'gat':
                conv = GATConv(in_dim, out_dim, heads=4, concat=False)
            elif gnn_type == 'sage':
                conv = SAGEConv(in_dim, out_dim)
            elif gnn_type == 'gin':
                nn = Sequential(Linear(in_dim, out_dim), BatchNorm1d(out_dim), ReLU())
                conv = GINConv(nn, train_eps=True)
            else:
                raise ValueError(f"不支持的GNN类型: {gnn_type}")
            
            self.convs.append(conv)
            self.bns.append(BatchNorm1d(out_dim))
        
        # 池化函数映射
        self.pooling_fns = {
            'avg': global_mean_pool,
            'max': global_max_pool,
            'min': global_min_pool
        }
        
        # 分类器
        self.classifier = Linear(hidden_channels, out_channels)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GNN消息传递
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        
        # 图级池化
        if self.pooling_type not in self.pooling_fns:
            raise ValueError(f"不支持的池化类型: {self.pooling_type}")
        
        x_pool = self.pooling_fns[self.pooling_type](x, batch)
        return self.classifier(x_pool)