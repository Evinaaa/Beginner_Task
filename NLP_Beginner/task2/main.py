import csv
import random
import torch
from feature_extraction import Word_embedding, Random_embedding, Glove_embedding
from comparison_plot import comparison_plot
 
# 读取数据
with open('data/train.tsv') as f:
    tsvreader = csv.reader (f, delimiter ='\t')
    temp = list(tsvreader)
 
# 读取预训练词向量模型glove
with open('data/glove.6B.50d.txt', 'rb') as f:
    lines = f.readlines()
 
# 将GloVe模型训练得到的词向量存储到字典中
trained_dict = dict()
n=len(lines)
for i in range(n):
    line = lines[i].split()
    trained_dict[line[0].decode("utf-8").upper()]=[float(line[j]) for j in range(1,51)]
 
# 初始化参数设置
iter_times = 50
alpha = 0.001
data = temp[1:]
batch_size = 500
 
# word embedding的初始化方式
random.seed(2023)
word_embedding = Word_embedding(data=data, model_path="result/Word2Vec.model")
word_embedding.get_words()
word_embedding.get_id()
 
# 随机embedding的初始化方式
random.seed(2023)  #随机种子均设置为相同值
random_embedding = Random_embedding(data=data)
random_embedding.get_words()
random_embedding.get_id()
 
# 用glove预训练的embedding进行初始化
random.seed(2023)
glove_embedding = Glove_embedding(data=data, trained_dict=trained_dict)
glove_embedding.get_words()
glove_embedding.get_id()
 
#绘图比较结果
comparison_plot(word_embedding, random_embedding, glove_embedding, alpha, batch_size, iter_times)
