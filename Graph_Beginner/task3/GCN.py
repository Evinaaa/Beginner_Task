import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_sort_pool, GCNConv
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import train_test_split
import numpy as np

# Load PROTEINS dataset
dataset = TUDataset(root='data/TUDataset', name='PROTEINS')

class DGCNN(nn.Module):
    def __init__(self, hidden_channels=32, num_layers=4, k=35):
        super(DGCNN, self).__init__()
        self.k = k
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(-1, hidden_channels))

        # Corrected Conv1d layers
        self.conv1 = nn.Conv1d(hidden_channels * num_layers, 16, kernel_size=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1)

        # Calculate the correct input size for the linear layer
        # After sort pooling: [batch_size, k * hidden_channels * num_layers]
        # After conv1: [batch_size, 16, k]
        # After pool: [batch_size, 16, k//2]
        # After conv2: [batch_size, 32, (k//2)-4]
        linear_input_size = 32 * ((k // 2) - 4)

        self.fc1 = nn.Linear(linear_input_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, edge_index, batch):
        xs = []
        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index))
            xs.append(x)

        x = torch.cat(xs, dim=1)
        x = global_sort_pool(x, batch, self.k)  # [batch_size, k * hidden_channels * num_layers]

        # Reshape for Conv1d: [batch_size, channels, sequence_length]
        batch_size = len(torch.unique(batch))
        x = x.view(batch_size, self.k, -1)  # [batch_size, k, hidden_channels * num_layers]
        x = x.permute(0, 2, 1)  # [batch_size, hidden_channels * num_layers, k]

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = x.view(batch_size, -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x
    
model = DGCNN(hidden_channels=32, num_layers=4, k=35)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.BCELoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

train_idx, test_idx = train_test_split(
    range(len(dataset)), 
    test_size=0.3, 
    stratify=[data.y.item() for data in dataset],
    random_state=42
)

train_dataset = dataset[train_idx]
test_dataset = dataset[test_idx]

train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

train_idx, test_idx = train_test_split(
    range(len(dataset)), 
    test_size=0.3, 
    stratify=[data.y.item() for data in dataset],
    random_state=42
)

train_dataset = dataset[train_idx]
test_dataset = dataset[test_idx]

train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)