import torch
import torch.nn as nn
import torch.nn.functional as F
 
# 定义CNN类，继承自PyTorch中的nn.Module类，表示这是一个模型类
class CNN_model(nn.Module):
    def __init__(self, len_feature, len_words, longest, len_kernel=50, typenum=5, weight=None, drop_out=0.5):
        super(CNN_model, self).__init__()
        self.len_feature = len_feature
        self.len_words = len_words
        self.len_kernel = len_kernel
        self.longest = longest
        self.dropout = nn.Dropout(drop_out)
        # 将输入的文本序列转换为词向量
        if weight is None:
            x = nn.init.xavier_normal_(torch.Tensor(len_words, len_feature))
            self.embedding = nn.Embedding(num_embeddings=len_words, embedding_dim=len_feature, _weight=x).cuda()
        else:
            self.embedding = nn.Embedding(num_embeddings=len_words, embedding_dim=len_feature, _weight=weight).cuda()
        # 定义了四个不同长度的卷积层，卷积核大小分别为2、3、4、5
        # longest表示在文本序列的长度方向进行padding，padding表示在词向量维度上进行padding
        # ReLU激活函数作为卷积层的非线性函数
        self.conv1 = nn.Sequential(nn.Conv2d(1, longest, (2, len_feature), padding=(1, 0)), nn.ReLU()).cuda()
        self.conv2 = nn.Sequential(nn.Conv2d(1, longest, (3, len_feature), padding=(1, 0)), nn.ReLU()).cuda()
        self.conv3 = nn.Sequential(nn.Conv2d(1, longest, (4, len_feature), padding=(2, 0)), nn.ReLU()).cuda()
        self.conv4 = nn.Sequential(nn.Conv2d(1, longest, (5, len_feature), padding=(2, 0)), nn.ReLU()).cuda()
        # 全连接层
        self.fc = nn.Linear(4 * longest, typenum).cuda()
 
        # 表示在最终的输出层之后再添加一个 softmax 层。
        # 但由于在训练过程中使用交叉熵损失函数，softmax 层的作用被交叉熵损失函数替代了。因此，该层是多余的，可以注释掉。
        # self.act = nn.Softmax(dim=1)
 
    def forward(self, x):
        x = torch.LongTensor(x).cuda()
        # 将输入 x 通过 embedding 层转换成张量，并对张量的形状进行重构，使其能够传递给卷积层
        out_put = self.embedding(x).view(x.shape[0], 1, x.shape[1], self.len_feature)
        out_put = self.dropout(out_put)
 
        conv1 = self.conv1(out_put).squeeze(3)
        # 使用 max_pool1d 对卷积后的输出进行最大池化，以减少输出的维度
        pool1 = F.max_pool1d(conv1, conv1.shape[2])
 
        conv2 = self.conv2(out_put).squeeze(3)
        pool2 = F.max_pool1d(conv2, conv2.shape[2])
 
        conv3 = self.conv3(out_put).squeeze(3)
        pool3 = F.max_pool1d(conv3, conv3.shape[2])
 
        conv4 = self.conv4(out_put).squeeze(3)
        pool4 = F.max_pool1d(conv4, conv4.shape[2])
 
        pool = torch.cat([pool1, pool2, pool3, pool4], 1).squeeze(2)
        # 将4个卷积层的输出连接在一起，然后通过全连接层fc进行线性变换
        # 最终的输出是一个形状为 (batch_size, typenum) 的张量
        out_put = self.fc(pool)
        # out_put = self.act(out_put)
        return out_put


# 定义RNN类，继承自PyTorch中的nn.Module类，表示这是一个模型类
class RNN_model(nn.Module):
    """
    len_feature：特征向量的长度
    len_hidden：隐藏层的长度
    len_words：词嵌入矩阵的行数，即词汇表的大小
    typenum：分类问题中的类别数目，默认为5
    weight：词嵌入矩阵的权重，如果为None，则随机初始化，否则使用给定的权重
    layer：RNN的层数，默认为1
    nonlinearity：RNN中的非线性激活函数，默认为tanh
    batch_first：是否将batch放在第一维，默认为True。
    drop_out：dropout的概率，默认为0.5
    """
    def __init__(self, len_feature, len_hidden, len_words, typenum=5, weight=None, layer=1, nonlinearity='tanh',
                 batch_first=True, drop_out=0.5):
        super(RNN_model, self).__init__()
        self.len_feature = len_feature
        self.len_hidden = len_hidden
        self.len_words = len_words
        self.layer = layer
        self.dropout = nn.Dropout(drop_out)
        if weight is None:
            # Xavier初始化方法，即从一个均匀分布中随机采样得到权重
            x = nn.init.xavier_normal_(torch.Tensor(len_words, len_feature))
            # 将词汇表中的每个单词映射到一个 len_feature 维的向量空间中
            self.embedding = nn.Embedding(num_embeddings=len_words, embedding_dim=len_feature, _weight=x).cuda()
        else:
            self.embedding = nn.Embedding(num_embeddings=len_words, embedding_dim=len_feature, _weight=weight).cuda()
        self.rnn = nn.RNN(input_size=len_feature, hidden_size=len_hidden, num_layers=layer, nonlinearity=nonlinearity,
                          batch_first=batch_first, dropout=drop_out).cuda()
        # 全连接层
        # 输入大小为len_hidden，输出大小为typenum，用于将RNN的输出转换为分类问题的输出
        self.fc = nn.Linear(len_hidden, typenum).cuda()
        # An extra softmax layer may be redundant
        # self.act = nn.Softmax(dim=1)
 
    def forward(self, x):
        x = torch.LongTensor(x).cuda()
        batch_size = x.size(0)
        # 将输入x映射为词嵌入向量，然后通过dropout层进行正则化
        out_put = self.embedding(x)
        out_put=self.dropout(out_put)
        # h0 = torch.randn(self.layer, batch_size, self.len_hidden).cuda()
 
        # 初始化隐藏状态 h0 为全零向量，并通过 RNN 层 rnn 得到最终隐藏状态 hn
        h0 = torch.autograd.Variable(torch.zeros(self.layer, batch_size, self.len_hidden)).cuda()
        # 将 RNN 层的输出状态 hn 赋值给变量 hn，而将 RNN 层的输入 out_put 丢弃
        # 因为在这里只需要输出状态，而不需要每一个时间步的输出，使用 _ 来代替不需要保留的输出
        _, hn = self.rnn(out_put, h0)
        # 将最终隐藏状态hn通过全连接层fc得到输出，然后通过squeeze(0)函数去除batch维度，得到形状为(typenum,)的输出向量
        out_put = self.fc(hn).squeeze(0)
        # out_put = self.act(out_put)
        return out_put
