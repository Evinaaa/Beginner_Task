import random
from feature import Random_embedding, Glove_embedding, get_batch
from comparison_plot import NN_plot, NN_embdding
from Neural_Network import ESIM
 
# 读取数据
with open('data/snli_1.0/snli_1.0_train.txt', 'r') as f:
    temp = f.readlines()
 
# 读取预训练词向量模型glove
with open('data/glove.6B.50d.txt', 'rb') as f:
    lines = f.readlines()
 
# 将GloVe模型训练得到的词向量存储到字典中
trained_dict = dict()
n = len(lines)
for i in range(n):
    line = lines[i].split()
    trained_dict[line[0].decode("utf-8").upper()] = [float(line[j]) for j in range(1, 51)]
 
# 初始化参数设置
data = temp[1:]
learning_rate = 0.001
len_feature = 50
len_hidden = 50
iter_times = 50
batch_size = 1000
 
# random embedding
random.seed(2025)
random_embedding = Random_embedding(data=data)
random_embedding.get_words()
random_embedding.get_id()
 
# trained embedding : glove
random.seed(2025)
glove_embedding = Glove_embedding(data=data, trained_dict=trained_dict)
glove_embedding.get_words()
glove_embedding.get_id()
 
# 绘图比较结果
NN_plot(random_embedding, glove_embedding, len_feature, len_hidden, learning_rate, batch_size, iter_times)
