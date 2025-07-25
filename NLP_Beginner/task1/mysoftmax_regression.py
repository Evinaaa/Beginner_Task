import numpy as np
import random
 
class Softmax:
    """softmax regression"""
    def __init__(self,sample,typenum,feature):
        self.sample = sample    #训练集样本个数
        self.typenum = typenum  #情感种类
        self.feature = feature  #dict_words向量的长度
        self.W = np.random.randn(feature,typenum) #参数矩阵W初始化
 
    def softmax_calculation(self,x):
        """x是向量，计算softmax值"""
        exp = np.exp(x-np.max(x))   #先减去行最大值防止指数太大溢出
        return exp/exp.sum()
 
    def softmax_all(self,wtx):
        """wtx是矩阵，即许多向量叠在一起，按行计算softmax的值"""
        wtx -= np.max(wtx,axis=1,keepdims=True)
        wtx = np.exp(wtx)
        wtx = wtx/np.sum(wtx,axis=1,keepdims=True)
        return wtx
 
    def change_y(self,y):
        """把情感种类转化为一个one-hot向量"""
        ans = np.array([0]*self.typenum)
        # print("{}".format(y))
        ans[y] = 1
        return ans.reshape(-1,1)
 
    def prediction(self,x):
        """给定0-1矩阵X，计算每个句子的y_hat值（概率）"""
        prob = self.softmax_all(x.dot(self.W))
        return prob.argmax(axis=1)           #既然返回的是最大值，那为什么要需要经过softmax,直接返回 x.dot(self.W)的最大值也可以哇
 
    def correct_rate(self,train,train_y,test,test_y):
        """计算训练集和测试集的准确率"""
        #train set
        n_train = len(train)
        pred_train = self.prediction(train)
        train_correct = sum(train_y[i] == pred_train[i] for i in range(n_train)) / n_train
        #test set
        n_test = len(test)
        pred_test = self.prediction(test)
        test_correct = sum(test_y[i] == pred_test[i] for i in range(n_test)) / n_test
 
        print("train_correct:{}   test_correct:{}".format(train_correct,test_correct))
        return train_correct,test_correct
 
    def regression(self,x,y,alpha,times,strategy="mini",mini_size=100):
        """Softmax regression"""
        if self.sample != len(x) or self.sample != len(y):    #这里x 和 y不应该一样长吗
            raise Exception("Sample size does not match!")
        #mini-batch
        if strategy == "mini":
            for i in range(times):
                increment = np.zeros((self.feature,self.typenum))
                for i in range(mini_size):                  #随机抽mini-size次
                    k = random.randint(0,self.sample-1)
                    y_hat = self.softmax_calculation(self.W.T.dot(x[k].reshape(-1,1)))
                    increment += x[k].reshape(-1,1).dot((self.change_y(int(y[k]))-y_hat).T)     #梯度加和
                self.W += alpha/mini_size*increment         #参数更新
        #shuffle 随机梯度
        elif strategy == "shuffle":
            for i in range(times):
                k = random.randint(0,self.sample-1)     #每次抽一个
                y_hat = self.softmax_calculation(self.W.T.dot(x[k].reshape(-1,1)))
                # print("y[{}]:{}".format(k,y[k]))
                # print("y_hat:{}".format(y_hat))
                increment = x[k].reshape(-1,1).dot((self.change_y(int(y[k]))-y_hat).T)      #计算梯度
 
                self.W += alpha*increment   #参数更新
 
        #batch 整批量梯度
        elif strategy == "batch":
            for i in range(times):
                increment = np.zeros((self.feature,self.typenum))
                for i in range(self.sample):    #所有样本都要计算
                    k = random.randint(0,self.sample-1)
                    y_hat = self.softmax_calculation(self.W.T.dot(x[k].reshape(-1,1)))
                    increment += x[k].reshape(-1,1).dot((self.change_y(int(y[k]))-y_hat).T)     #梯度加和
                self.W += alpha*increment/self.sample   #参数更新
 
        else:
            raise Exception("Unkown strategy")

