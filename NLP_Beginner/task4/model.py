import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torchcrf import CRF  # pytorch-crf包提供了一个CRF层的PyTorch版本实现
 
 
"""定义神经网络模型类 LSTM_CRF"""
class LSTM_CRF(nn.Module):
 
    # vocab_size: 词汇表的大小（即词汇量）。
    # tag_to_index: 一个字典，将标签映射到索引。
    # embedding_size: 嵌入层的维数。
    # hidden_size: 隐藏层的大小。
    # max_length: 句子的最大长度。
    # vectors: 预训练词向量（默认为None）。
 
    def __init__(self, vocab_size, tag_to_index, embedding_size, hidden_size, max_length, vectors=None):
        # 调用父类的初始化函数
        super(LSTM_CRF, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        # 标签到索引的映射字典
        self.tag_to_index = tag_to_index
        # 标签数量
        self.target_size = len(tag_to_index)
        # 初始化嵌入层，如果提供了预训练词向量，则使用预训练词向量进行初始化
        if vectors is None:
            self.embedding = nn.Embedding(vocab_size, embedding_size)
        else:
            self.embedding = nn.Embedding.from_pretrained(vectors)
        # 初始化BiLSTM模型
        # hidden_size // 2 是整除运算符，表示将 hidden_size 的值除以 2 并向下取整，得到的结果作为新的 hidden_size 的值。
        # 这个操作的目的是将 LSTM 层的 hidden_size 拆成两个部分，以便实现双向 LSTM。
        # 如果 hidden_size 的值为 100，则 hidden_size // 2 的值为 50，即每个方向上的 LSTM 的 hidden_size 值都是 50。
        self.lstm = nn.LSTM(embedding_size, hidden_size // 2, bidirectional=True)
        # 定义一个全连接层，将隐藏层的输出映射到标签的数量
        self.hidden_to_tag = nn.Linear(hidden_size, self.target_size)
        # 定义条件随机场层
        self.crf = CRF(self.target_size, batch_first=True)
        # 存储最大句子长度
        self.max_length = max_length
 
    # 定义函数，根据给定的句子长度列表生成一个掩码
    def get_mask(self, length_list):
        # 函数根据输入的句子长度列表生成一个掩码张量，掩码张量用于屏蔽输入句子中的填充元素
        mask = []
        # 零填充: 根据长度列表生成掩码张量，其中长度小于最大长度的位置用0填充，否则用1填充。
        for length in length_list:
            mask.append([1 for i in range(length)] + [0 for j in range(self.max_length - length)])
        return torch.tensor(mask, dtype=torch.bool)
 
    # 定义LSTM层的前向传递函数
    def LSTM_Layer(self, sentences, length_list):
        # 将输入序列嵌入到低维空间中
        embeds = self.embedding(sentences)
        # 使用pack_padded_sequence函数将嵌入序列打包
        packed_sentences = pack_padded_sequence(embeds, lengths=length_list, batch_first=True, enforce_sorted=False)
        # 使用LSTM层处理打包后的序列
        lstm_out, _ = self.lstm(packed_sentences)
        # 将打包后的序列解包
        result, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=self.max_length)
        # 将结果传递到全连接层中进行标记预测
        feature = self.hidden_to_tag(result)
        return feature
 
    # 计算给定排放分数的标签序列的条件对数似然性
    def CRF_layer(self, input, targets, length_list):
        """input：发射得分张量，大小为(seq_length, batch_size, num_tags)或(batch_size, seq_length, num_tags)，取决于batch_first参数是否为 True。
        targets：标记序列张量，大小为(seq_length, batch_size)或(batch_size, seq_length)，取决于batch_first参数是否为True。
        length_list：每个句子的实际长度列表。
        该函数调用了self.crf，它是一个torchcrf库中的CRF层。它接受3个参数：
        emissions：发射得分张量，大小为(seq_length, batch_size, num_tags)或(batch_size, seq_length, num_tags)，取决于batch_first参数是否为True。
        tags：标记序列张量，大小为(seq_length, batch_size)或(batch_size, seq_length)，取决于batch_first参数是否为True。
        mask：掩码张量，大小为(seq_length, batch_size)或(batch_size, seq_length)，取决于batch_first参数是否为True。
        """
        return self.crf(input, targets, self.get_mask(length_list))
 
    def forward(self, sentences, length_list, targets):
        # length_list 包含了每个句子的实际长度；targets 包含了每个句子中每个词对应的标签
        # 调用 LSTM_Layer 方法对输入序列进行处理得到 x。
        x = self.LSTM_Layer(sentences, length_list)
        # 将 x 和 targets 传递给 CRF_layer 方法，用于计算条件对数似然
        x = self.CRF_layer(x, targets, length_list)
        return x
 
    def predict(self, sentences, length_list):
        out = self.LSTM_Layer(sentences, length_list)
        mask = self.get_mask(length_list)
        # 将 LSTM_Layer 的输出 out 和 mask 传递给 decode 方法来预测每个词对应的标签序列，然后将预测得到的标签序列返回。
        return self.crf.decode(out, mask)
