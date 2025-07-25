import os
import numpy as np
import time
import torch
import matplotlib.pyplot as plt
from torchtext.vocab import Vectors
from utils import read_data
from utils import get_dataloader
from utils import pre_processing
from utils import compute_f1
from model import LSTM_CRF
 
 
# 参数设置
n_classes = 5  # 分类个数
batch_size = 250
embedding_size = 50  # 每个词向量有几维（几个特征）
hidden_size = 50
epochs = 20
vectors = Vectors('data/glove.6B.50d.txt')
 
 
def train(model, vocab_size, tag2idx, embedding_size, hidden_size, max_length, vectors=None):
    # model: 要训练的模型，类型为 LSTMCRF。
    # vocab_size: 词汇表大小，即训练集中不同词汇的数量。
    # tag2idx: 标签到索引的映射字典。
    # embedding_size: 嵌入层的维度大小。
    # hidden_size: LSTM 隐藏层的维度大小。
    # max_length: 训练集中句子的最大长度。
    # vectors: 预训练的词向量矩阵。
    model = model(vocab_size, tag2idx, embedding_size, hidden_size, max_length, vectors=vectors)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # loss_history 记录了每个 epoch 的平均损失，f1_history 记录了每个 epoch 的平均 F1 分数
    loss_history = []
    # 训练集数据加载器的长度，等于训练集中的样本数量除以batch size
    print("dataloader length: ", len(train_dataloader))
    model.train()
    f1_history = []
    # 循环次数为epochs，epoch次数 = 迭代次数 / batch size；迭代次数 = 样本数量 / batch size
    for epoch in range(epochs):
        total_loss = 0.
        f1 = 0
        for idx, (inputs, targets, length_list) in enumerate(train_dataloader):
            # 梯度清零，以免梯度累积
            model.zero_grad()
            # 计算每个样本的损失，并将损失加到总损失中。损失是模型的负对数似然损失
            loss = (-1) * model(inputs, length_list, targets)
            total_loss += loss.item()
            # 这两行代码计算每个样本的预测结果，并将F1分数加到总F1分数中
            pred = model.predict(inputs, length_list)
            f1 += compute_f1(pred, targets, length_list)
            # 计算模型的梯度、对梯度进行剪裁，然后使用优化器
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            # 每 10 个批次输出一次当前 epoch 的平均损失和 F1 分数
            if (idx + 1) % 10 == 0 and idx:
                cur_loss = total_loss
                loss_history.append(cur_loss / (idx+1))
                f1_history.append(f1 / (idx+1))
                total_loss = 0
                # batch指一次性处理的一组训练样本，将其分为多个小批次，每个小批次包含的样本数量为batch size
                print("epochs : {}, batch : {}, loss : {}, f1 : {}".format(epoch+1, idx*batch_size,
                                                                           cur_loss / (idx * batch_size), f1 / (idx+1)))
    # 绘制损失图
    plt.plot(np.arange(len(loss_history)), np.array(loss_history))
    plt.xlabel('Iterations')
    plt.ylabel('Training Loss')
    plt.title('LSTM+CRF model')
    plt.show()
    # 绘制f1得分图
    plt.plot(np.arange(len(f1_history)), np.array(f1_history))
    plt.title('train f1 scores')
    plt.show()
 
    # 将模型设置为评估模式，这意味着在模型的前向传播过程中，不会更新权重，也不会计算梯度，以加快模型的执行速度
    model.eval()
    f1 = 0
    f1_history = []
    s = 0
    with torch.no_grad():
        # 迭代测试数据集的每个batch，其中inputs是输入序列，targets是对应的标签序列，length_list是每个输入序列的实际长度。
        for idx, (inputs, targets, length_list) in enumerate(test_dataloader):
            loss = (-1) * model(inputs, length_list, targets)
            total_loss += loss.item()
            # 使用模型进行预测，并返回预测的标签序列
            pred = model.predict(inputs, length_list)
            # 计算预测标签序列和真实标签序列之间的F1值，并将结果累加到f1中
            f1 += compute_f1(pred, targets, length_list) * 250
    print("f1 score : {}, test size = {}".format(f1/3200, 3200))
 
 
if __name__ == '__main__':
    x_train, y_train = read_data("data/train.txt", 14000)
    x_test, y_test = read_data("data/test.txt", 3200)
    word2idx, tag2idx, vocab_size = pre_processing()
    train_dataloader, train_max_length = get_dataloader(x_train, y_train, batch_size)
    test_dataloader, test_max_length = get_dataloader(x_test, y_test, 250)
    train(LSTM_CRF, vocab_size, tag2idx, embedding_size, hidden_size, max_length=train_max_length, vectors=None)
