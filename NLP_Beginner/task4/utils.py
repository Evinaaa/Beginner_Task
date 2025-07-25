import torch
from torch.utils.data import DataLoader, Dataset
 
 
""" 读取数据 """
def read_data(path, length):  # length 限制读取的句子数量
    sentences_list = []         # 每一个元素是一整个句子
    sentences_list_labels = []  # 每个元素是一整个句子的标签
    with open(path, 'r', encoding='UTF-8') as f:
        sentence_labels = []    # 每个元素是这个句子的每个单词的标签
        sentence = []           # 每个元素是这个句子的每个单词
 
        for line in f:
            line = line.strip()  # 对于文件中的每一行，删除字符串前后的空白字符
            if not line:        # 如果遇到了空白行
                if sentence:    # 如果上一个句子不是空句子（防止空白行连续多个，导致出现空白的句子）
                    sentences_list.append(' '.join(sentence))  # 将单词合并为句子，并将该句子加入到列表sentences_list中
                    sentences_list_labels.append(' '.join(sentence_labels))  # 将标签合并为标签序列，并将该标签序列加入到列表sentences_list_labels中
 
                    sentence = []
                    sentence_labels = []  # 重置，开始处理下一个句子的单词和标签
            else:
                res = line.split()  # 将一行字符串按空格划分为单词、空格、标签、空格四个部分
                assert len(res) == 4  # 断言每一行都必须划分为4个部分
                if res[0] == '-DOCSTART-':  #如果该行为起始标志，忽略该行，开始处理下一行
                    continue
                sentence.append(res[0])  # 将单词加入到sentence列表中
                sentence_labels.append(res[3])  # 将标签加入到sentence_labels列表中
 
        if sentence:            # 处理最后一个句子，防止最后一个句子没有空白行
            sentences_list.append(' '.join(sentence))
            sentences_list_labels.append(' '.join(sentence_labels))
    return sentences_list[:length], sentences_list_labels[:length]  # 返回处理好的句子及其对应的标签序列，length指定了返回的句子数量
 
 
""" 构建词典（分词）"""
def build_vocab(sentences_list):  # sentences_list 包含多个句子的列表
    vocab = []
    # 使用列表解析式将 sentences 拆分成单个单词，并返回一个由这些单词组成的列表。
    for sentences in sentences_list:
        vocab += [word for word in sentences.split()]
    # 首先使用 set 函数将列表中的元素去重，然后将去重后的元素转换为列表
    return list(set(vocab))
 
 
""" 自定义数据集 """
class ClsDataset(Dataset):  # 用于将输入的数据和标签转换为可迭代的数据集对象
    def __init__(self, x: torch.Tensor, y: torch.Tensor, length_list):
        self.x = x
        self.y = y
        self.length_list = length_list
    def __getitem__(self, index):  # 返回给定索引的数据项
        data = self.x[index]  # 使用给定索引从输入特征张量中获取相应的输入数据
        labels = self.y[index]  # 使用给定索引从目标变量张量中获取相应的标签数据
        length = self.length_list[index]  # 使用给定索引从输入序列长度列表中获取相应的序列长度
        return data, labels, length
    def __len__(self):  # 返回数据集的长度
        return len(self.x)
 
 
""" 返回单词在字典中的索引 """
def get_idx(word, d):
    # 判断字典 d 中是否包含单词 word 的索引，如果包含则返回该索引
    if d[word] is not None:
        return d[word]
    # 如果字典 d 中不包含单词 word 的索引，则返回字典中预先定义好的 '<unknown>' 对应的索引
    else:
        return d['<unknown>']
 
 
""" 将句子转换为由词汇表中的单词索引组成的向量 """
def sentence2vector(sentence, d):  # d为词汇表，由单词索引组成的字典
    # 使用列表推导式将句子分割成单词，对于每个单词调用get_idx函数获得它在字典中的索引，然后将所有的单词索引组成的列表返回
    return [get_idx(word, d) for word in sentence.split()]
 
 
"""用指定值填充序列"""
def padding(x, max_length, d):
    length = 0
    # 确定填充长度后，将 <pad> 对应的值标记添加到 x 的末尾，进行填充
    for i in range(max_length - len(x)):
        x.append(d['<pad>'])
    return x
 
 
""" 将原始文本数据集 x 和 y 转换为 PyTorch 的数据加载器 """
def get_dataloader(x, y, batch_size):
    word2idx, tag2idx, vocab_size = pre_processing()  # 预处理数据并建立词表和标签表
    inputs = [sentence2vector(s, word2idx) for s in x]  # 每一个句子都转化成vector
    targets = [sentence2vector(s, tag2idx) for s in y]
    # 计算每个句子的长度
    length_list = [len(sentence) for sentence in inputs]
    # 找到最长的句子的长度并将其截断为124
    max_length = max(max(length_list), 124)
    # 使用padding将每个句子填充为最大长度
    inputs = torch.tensor([padding(sentence, max_length, word2idx) for sentence in inputs])
    targets = torch.tensor([padding(sentence, max_length, tag2idx) for sentence in targets], dtype=torch.long)
    # 创建数据集并使用DataLoader加载数据
    dataset = ClsDataset(inputs, targets, length_list)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)
    # 返回数据加载器和最大长度
    return dataloader, max_length
 
 
""" 数据预处理 """
def pre_processing():
    # 调用 read_data 函数读取训练集和测试集数据，返回两个元组，每个元组包含两个列表，分别是输入数据和标签数据
    x_train, y_train = read_data("data/train.txt", 14000)
    x_test, y_test = read_data("data/test.txt", 3200)
    # 调用 build_vocab 函数，分别对输入和标签数据建立词汇表。这里将训练集和测试集合并后一起建立词汇表。
    d_x = build_vocab(x_train+x_test)
    d_y = build_vocab(y_train+y_test)
    # 将每个词汇/标签映射到一个唯一的整数，用字典存储。字典的键是词汇/标签，值是整数。
    word2idx = {d_x[i]: i for i in range(len(d_x))}
    tag2idx = {d_y[i]: i for i in range(len(d_y))}
    # 为起始标签和终止标签分别添加索引值。这些标签通常用于序列标注任务中。
    tag2idx["<START>"] = 9
    tag2idx["<STOP>"] = 10
    # 为填充标记添加索引。将填充标记添加到词汇表和标签字典的末尾。
    pad_idx = len(word2idx)
    word2idx['<pad>'] = pad_idx
    tag2idx['<pad>'] = len(tag2idx)
    # 计算词汇表的大小，建立标签到索引的反向映射字典。输出标签到索引的字典。
    vocab_size = len(word2idx)
    # idx2tag = {value: key for key, value in tag2idx.items()}
    print(tag2idx)
    # 返回词汇表、标签字典和词汇表大小
    return word2idx, tag2idx, vocab_size
 
 
""" 计算F1-score """
def compute_f1(pred, targets, length_list):
    # 初始化 TP, FN 和 FP
    tp, fn, fp = [], [], []
    # 共有15个标签
    for i in range(15):
        tp.append(0)
        fn.append(0)
        fp.append(0)
    # 遍历每个句子的标签预测结果和真实标签，更新计数。
    for i, length in enumerate(length_list):
        for j in range(length):
            # 获取预测的标签和真实的标签。
            a, b = pred[i][j], targets[i][j]
            # 若预测的标签和真实的标签一致，则增加标签的 TP 计数。
            if (a == b):
                tp[a] += 1
            else:
                fp[a] += 1
                fn[b] += 1
    # 计算所有有效标签的TP/FP/FN
    tps = 0
    fps = 0
    fns = 0
    for i in range(9):
        tps += tp[i]
        fps += fp[i]
        fns += fn[i]
    # 计算 Precision 和 Recall
    p = tps / (tps + fps)
    r = tps / (tps + fns)
    # 计算 F1 分数并返回
    return 2 * p * r / (p + r)
