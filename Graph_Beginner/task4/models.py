class KGE_Model(nn.Module):
    def __init__(self, n_entities, n_relations, dim):
        super().__init__()
        self.ent_emb = nn.Embedding(n_entities, dim)
        self.rel_emb = nn.Embedding(n_relations, dim)
    
    def forward(self, h_idx, r_idx, t_idx):
        raise NotImplementedError
        
    def score(self, h, r, t):
        raise NotImplementedError
    
# TransE 实现
class TransE(KGE_Model):
    def __init__(self, n_entities, n_relations, dim, p_norm=1):
        super().__init__(n_entities, n_relations, dim)
        self.p_norm = p_norm
        
    def score(self, h, r, t):
        return -torch.norm(h + r - t, p=self.p_norm, dim=-1)

# RotatE 实现（需复数支持）
class RotatE(KGE_Model):
    def __init__(self, n_entities, n_relations, dim):
        super().__init__(n_entities, n_relations, dim//2)  # 复数维度减半
        self.rel_emb = nn.Embedding(n_relations, dim//2)
        
    def score(self, h, r, t):
        # 将实向量转为复数
        h_re, h_im = torch.chunk(h, 2, dim=-1)
        r_re, r_im = torch.chunk(r, 2, dim=-1)
        t_re, t_im = torch.chunk(t, 2, dim=-1)
        
        # 复数乘法：h ◦ r
        rot_h_re = h_re * r_re - h_im * r_im
        rot_h_im = h_re * r_im + h_im * r_re
        
        # 计算距离
        return -torch.norm(torch.cat([rot_h_re, rot_h_im], dim=-1) - 
                         torch.cat([t_re, t_im], dim=-1), dim=-1)

# ConvE 实现
class ConvE(KGE_Model):
    def __init__(self, n_entities, n_relations, dim, emb_dim1=10):
        super().__init__(n_entities, n_relations, dim)
        self.emb_dim1 = emb_dim1
        self.conv = nn.Conv2d(1, 32, (3, 3))
        self.fc = nn.Linear(32 * (emb_dim1*2 - 2) * (emb_dim1 - 2), dim)
        
    def forward(self, h_idx, r_idx, t_idx):
        h = self.ent_emb(h_idx).view(-1, 1, self.emb_dim1*2, self.emb_dim1)
        r = self.rel_emb(r_idx).view(-1, 1, self.emb_dim1*2, self.emb_dim1)
        x = torch.cat([h, r], dim=2)  # 拼接头实体和关系
        
        x = F.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.sigmoid(x @ self.ent_emb.weight.T)