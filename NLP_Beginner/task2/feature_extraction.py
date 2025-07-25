import random
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
 
# 将数据按照一定的比例分割为训练集和测试集
def data_split(data,test_rate=0.3):
    train = list()
    test = list()
    for datum in data:
        if random.random() > test_rate:
            train.append(datum)
        else:
            test.append(datum)
    return train,test
 
 
# 随机初始化
class Random_embedding():
    def __init__(self,data,test_rate=0.3):
        self.dict_words = dict()   # 单词->ID的映射
        data.sort(key=lambda x:len(x[2].split())) # 按照句子长度排序，短着在前，这样做可以避免后面一个batch内句子长短不一，导致padding过度
        self.data = data
        self.len_words = 0  # 单词数目（包括padding的ID：0）
        self.train,self.test = data_split(data,test_rate=test_rate) # 训练集测试集划分
        self.train_y = [int(term[3]) for term in self.train]  # 训练集类别
        self.test_y = [int(term[3]) for term in self.test] # 测试集类别
        self.train_matrix = list()  # 训练集的单词ID列表，叠成一个矩阵
        self.test_matrix = list()  # 测试集的单词ID列表，叠成一个矩阵
        self.longest = 0  # 记录最长的单词
 
    def get_words(self):
        for term in self.data:
            s = term[2]  # 取出句子
            s = s.upper()  # 将其转化为大写，避免识别i和I为不同的两个单词
            words = s.split()
            for word in words:  # 一个一个单词进行寻找
                if word not in self.dict_words:
                    self.dict_words[word] = len(self.dict_words) + 1  # padding是第0个，所以要+1
        self.len_words = len(self.dict_words)  # 单词数目，暂未包括padding的id0
 
    def get_id(self):
        for term in self.train:  # 训练集
            s = term[2]
            s = s.upper()
            words = s.split()
            item = [self.dict_words[word] for word in words] # 找到id列表（未进行padding）
            self.longest = max(self.longest,len(item))  # 记录最长的单词
            self.train_matrix.append(item)
        for term in self.test:  # 测试集
            s = term[2]
            s = s.upper()
            words = s.split()
            item = [self.dict_words[word] for word in words]  # 找到id列表（未进行padding）
            self.longest = max(self.longest,len(item))
            self.test_matrix.append(item)
        self.len_words += 1  # 单词数目，包含padding的id0
 
 
class Glove_embedding():
    def __init__(self,data,trained_dict,test_rate=0.3):
        self.dict_words = dict()  # 单词->ID的映射
        self.trained_dict = trained_dict  # 记录预训练词向量模型
        data.sort(key = lambda x:len(x[2].split()))  # 按照句子长度排序，短着在前，这样做可以避免后面一个batch内句子长短不一，导致padding过度
        self.data = data
        self.len_words = 0 # 单词数目（包含padding的id0）
        self.train,self.test = data_split(data,test_rate=test_rate)  # 测试集和训练集的划分
        self.train_y = [int(term[3]) for term in self.train]  # 训练集类别
        self.test_y = [int(term[3]) for term in self.test]  # 测试集类别
        self.train_matrix = list()
        self.test_matrix = list()
        self.longest = 0
        self.embedding = list()  # 抽取出用到的，即预训练模型的单词
 
    def get_words(self):
        self.embedding.append([0] * 50)  # 先加padding的词向量
        for term in self.data:
            s = term[2]  # 取出句子
            s = s.upper()
            words = s.split()
            for word in words:
                if word not in self.dict_words:
                    self.dict_words[word] = len(self.dict_words)+1  # padding是第0个所以要加一
                    if word in self.trained_dict:  # 如果预训练模型中有这个单词，直接记录词向量
                        self.embedding.append(self.trained_dict[word])
                    else:  # 如果预训练模型中没有这个单词，则初始化该词的对应词向量为0向量
                        self.embedding.append([0]*50)
        self.len_words = len(self.dict_words)  # 单词数目（暂未包括padding的id0）
 
    def get_id(self):
        for term in self.train:  # 训练集
            s = term[2]
            s = s.upper()
            words = s.split()
            item = [self.dict_words[word] for word in words]  # 找到id列表（未进行padding）
            self.longest = max(self.longest,len(item))  # 记录最长的单词
            self.train_matrix.append(item)
        for term in self.test:  # 测试集
            s = term[2]
            s = s.upper()
            words = s.split()
            item = [self.dict_words[word] for word in words]
            self.longest = max(self.longest,len(item))
            self.test_matrix.append(item)
        self.len_words += 1  # 单词数目（暂未包括padding的id0）
 
 
# 自定义数据集的结构
class ClsDataset(Dataset):
        def __init__(self,sentence,emotion):
            self.sentence = sentence
            self.emotion = emotion
 
        def __getitem__(self, item):
            return self.sentence[item],self.emotion[item]
 
        def __len__(self):
            return len(self.emotion)
 
 
# 自定义数据集的内数据返回类型，并进行padding
def collate_fn(batch_data):
    sentence,emotion = zip(*batch_data)
    sentences = [torch.LongTensor(sent) for sent in sentence]  # 把句子变成LongTensor类型
    padded_sents = pad_sequence(sentences,batch_first=True,padding_value=0)  # 自动padding操作
    return torch.LongTensor(padded_sents),torch.LongTensor(emotion)
 
 
# 利用dataloader划分batch
def get_batch(x,y,batch_size):
    dataset = ClsDataset(x,y)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False,drop_last=True,collate_fn=collate_fn)
    return dataloader
    # shuffle是指每个epoch都随机打乱数据再分batch，设置成False，否则之前的顺序会直接打乱
    # drop_last是指不利用最后一个不完整的batch（数据大小不能被batch_size整除）

