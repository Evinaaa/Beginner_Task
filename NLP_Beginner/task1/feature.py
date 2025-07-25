#特征提取
import numpy as np
import random
 
def data_split(data,test_rate=0.3,max_item=1000):
    """把数据按一定比例分成训练集和测试集"""
    train = list()
    test = list()
    i = 0
    for datum in data:
        i += 1
        if random.random() >test_rate:
            train.append(datum)
        else:
            test.append(datum)
        if i>max_item:
            break
    return train,test
 
class Bag:
    """Bag of words"""
    def __init__(self,my_data,max_item):
        self.data = my_data[:max_item]
        self.max_item = max_item
        self.dict_words = dict()    #单词到单词编号的映射
        self.len = 0                #单词数量
        self.train,self.test = data_split(my_data,test_rate=0.3,max_item=1000)  #划分训练集测试集
        self.train_y = [int(term[3]) for term in self.train]    #训练集标签
        self.test_y = [int(term[3]) for term in self.test]      #测试集标签
        self.train_matrix = None    #训练集的0-1矩阵（每行一个句子）
        self.test_matrix = None     #测试集的0-1矩阵（每行一个句子）
 
    def get_words(self):
        for term in self.data:
            s = term[2]
            s = s.upper()   #统一转化为大写（或者全部转化为小写，否则I和i会被识别为两个不同的单词）
            words = s.split()
            for word in words:
                if word not in self.dict_words: #判断此单词是否出现过，如果没有则加入字典
                    self.dict_words[word] = len(self.dict_words)    #单词加入字典后，字典长度增加，则下一个单词的编号也增加
        self.len = len(self.dict_words)
        self.test_matrix = np.zeros((len(self.test),self.len))   #初始化0-1矩阵
        self.train_matrix = np.zeros((len(self.train),self.len))
 
    def get_matrix(self):
        for i in range(len(self.train)):    #训练集矩阵
            s = self.train[i][2]
            words = s.split()
            for word in words:
                word = word.upper()
                self.train_matrix[i][self.dict_words[word]]=1
        for i in range(len(self.test)):     #测试集矩阵
            s = self.test[i][2]
            words = s.split()
            for word in words:
                word = word.upper()
                self.test_matrix[i][self.dict_words[word]]=1
 
class Gram:
    """N-gram"""
    def __init__(self,my_data,dimension=2,max_item=1000):
        self.data = my_data[:max_item]
        self.max_item = max_item
        self.dict_words = dict()    #特征到编号的映射
        self.len = 0                #特征数量
        self.dimension = dimension  #使用几元特征，1-gram,2-gram...
        self.train,self.test = data_split(my_data,test_rate=0.3,max_item=max_item)
        self.train_y = [int(term[3]) for term in self.train]  #训练集类别
        self.test_y =[int(term[3]) for term in self.test]
        self.train_matrix = None                    #训练集0-1矩阵（每行代表一句话）
        self.test_matrix = None
 
    def get_words(self):
        for d in range(1,self.dimension+1):          #提取1-gram,2-gram...N-gram特征
            for term in self.data:
                s = term[2]
                s = s.upper()
                words = s.split()
                for i in range(len(words)-d+1): #一个一个遍历d-gram下的每一个特征
                    temp = words[i:i+d]
                    temp = '_'.join(temp)       #形成i d-gram 特征
                    if temp not in self.dict_words:
                        self.dict_words[temp] = len(self.dict_words)
        self.len = len(self.dict_words)
        self.train_matrix = np.zeros((len(self.train),self.len))  #训练集矩阵初始化
        self.test_matrix = np.zeros((len(self.test),self.len))
 
    def get_matrix(self):
        for d in range(1,self.dimension+1):
            for i in range(len(self.train)):    #训练集矩阵
                s = self.train[i][2]
                s = s.upper()
                words = s.split()
 
                for j in range(len(words)-d+1):
                    temp = words[j:j+d]
                    temp = '_'.join(temp)
                    self.train_matrix[i][self.dict_words[temp]] = 1
 
            for i in range(len(self.test)):     #测试集矩阵
                s = self.test[i][2]
                s = s.upper()
                words = s.split()
 
                for j in range(len(words)-d+1):
                    temp = words[j:j+d]
                    temp = '_'.join(temp)
                    self.test_matrix[i][self.dict_words[temp]] = 1

