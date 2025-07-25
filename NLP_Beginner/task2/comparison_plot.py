import matplotlib.pyplot
import torch
import torch.nn.functional as F
from torch import optim
from Neural_Network import RNN_model, CNN_model
from feature_extraction import get_batch
 
"""计算总损失和准确率"""
def accuracy(model, train, test, learning_rate, iter_times):
    #使用Adam优化器来优化模型的参数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 定义一个交叉熵损失函数
    loss_fun = F.cross_entropy
    # 分别用来记录训练损失、测试损失、长句子损失、训练准确率、测试准确率和长句子准确率
    train_loss_record=list()
    test_loss_record=list()
    long_loss_record=list()
    train_record=list()
    test_record=list()
    long_record=list()
    # torch.autograd.set_detect_anomaly(True)
 
    for iteration in range(iter_times):
        # 将模型设置为训练模式
        model.train()
        for i, batch in enumerate(train):
            x, y = batch
            y = y.cuda()
            pred = model(x).cuda()
            optimizer.zero_grad()
            loss = loss_fun(pred, y).cuda()
            # 使用训练数据对模型进行训练，对每个批次进行反向传播、梯度下降等操作
            loss.backward()
            optimizer.step()
 
        # 将模型设置为评估模式，即不进行梯度更新
        model.eval()
        train_acc = list()
        test_acc = list()
        long_acc = list()
        # 定义长句子的长度阈值为20
        length = 20
        train_loss = 0
        test_loss = 0
        long_loss = 0
        # 遍历训练集数据
        for i, batch in enumerate(train):
            x, y = batch
            y = y.cuda()
            # 使用模型对输入数据进行预测，并将结果移到GPU上进行加速计算
            pred = model(x).cuda()
            loss = loss_fun(pred, y).cuda()
            # 累加训练集损失值
            train_loss += loss.item()
            # 找到预测结果中概率最大的类别
            _, y_pre = torch.max(pred, -1)
            acc = torch.mean((torch.tensor(y_pre == y, dtype=torch.float)))
            train_acc.append(acc)
 
        # 遍历测试集数据
        for i, batch in enumerate(test):
            x, y = batch
            y = y.cuda()
            pred = model(x).cuda()
            loss = loss_fun(pred, y).cuda()
            test_loss += loss.item()
            _, y_pre = torch.max(pred, -1)
            acc = torch.mean((torch.tensor(y_pre == y, dtype=torch.float)))
            test_acc.append(acc)
 
            if(len(x[0]))>length:
              long_acc.append(acc)
              long_loss+=loss.item()
 
        # 计算平均准确率
        trains_acc = sum(train_acc) / len(train_acc)
        tests_acc = sum(test_acc) / len(test_acc)
        longs_acc = sum(long_acc) / len(long_acc)
 
        # 计算损失
        # 将所有训练数据的平均损失加入到train_loss_record中
        train_loss_record.append(train_loss / len(train_acc))  # 用于记录训练集上每个 epoch 的平均损失
        test_loss_record.append(test_loss / len(test_acc))
        long_loss_record.append(long_loss/len(long_acc))
 
        train_record.append(trains_acc.cpu())  # 用于记录训练集上每个 epoch 的准确率
        test_record.append(tests_acc.cpu())
        long_record.append(longs_acc.cpu())
 
        print("---------- 迭代轮数", iteration + 1, "----------")
        print("Train loss:", train_loss/ len(train_acc))  # 输出训练集上的平均损失
        print("Test loss:", test_loss/ len(test_acc))  # 输出测试集上的平均损失
        print("Train accuracy:", trains_acc)  # 输出训练集上的准确率
        print("Test accuracy:", tests_acc)  # 输出测试集上的准确率
        print("Long sentence accuracy:", longs_acc)  # 输出长句子测试集上的准确率
 
    return train_loss_record, test_loss_record, long_loss_record, train_record, test_record, long_record
 
 
"""绘制对比图"""
def comparison_plot(word_embedding, random_embedding, glove_embedding, learning_rate, batch_size, iter_times):
    # 将训练和测试数据分别按照批次大小分割成多个小批次，便于进行训练和测试
    train_word = get_batch(word_embedding.train_matrix, word_embedding.train_y, batch_size)
    test_word = get_batch(word_embedding.test_matrix, word_embedding.test_y, batch_size)
    train_random = get_batch(random_embedding.train_matrix,
                             random_embedding.train_y, batch_size)
    test_random = get_batch(random_embedding.test_matrix,
                            random_embedding.test_y, batch_size)
    train_glove = get_batch(glove_embedding.train_matrix,
                            glove_embedding.train_y, batch_size)
    test_glove = get_batch(glove_embedding.test_matrix,
                           glove_embedding.test_y, batch_size)
 
    # 使用PyTorch库中的"RNN_model"和"CNN_model"函数分别创建三种不同的神经网络模型
    # word_embedding
    torch.manual_seed(2023)
    torch.cuda.manual_seed(2023)
    word_rnn = RNN_model(50, 50, word_embedding.len_words, weight=torch.tensor(word_embedding.embedding, dtype=torch.float))
    torch.manual_seed(2023)
    torch.cuda.manual_seed(2023)
    word_cnn = CNN_model(50, word_embedding.len_words, word_embedding.longest, weight=torch.tensor(word_embedding.embedding, dtype=torch.float))
 
    # random_embedding
    torch.manual_seed(2023)
    torch.cuda.manual_seed(2023)
    random_rnn = RNN_model(50, 50, random_embedding.len_words)
    torch.manual_seed(2023)
    torch.cuda.manual_seed(2023)
    random_cnn = CNN_model(50, random_embedding.len_words, random_embedding.longest)
 
    # glove_embedding
    torch.manual_seed(2023)
    torch.cuda.manual_seed(2023)
    glove_rnn = RNN_model(50, 50, glove_embedding.len_words, weight=torch.tensor(glove_embedding.embedding, dtype=torch.float))
    torch.manual_seed(2023)
    torch.cuda.manual_seed(2023)
    glove_cnn = CNN_model(50, glove_embedding.len_words, glove_embedding.longest, weight=torch.tensor(glove_embedding.embedding, dtype=torch.float))
 
    # 调用函数"accuracy"，对上述六个神经网络模型进行训练和测试，并计算训练和测试数据的损失和准确率
    # word_embedding
    torch.manual_seed(2023)
    torch.cuda.manual_seed(2023)
    trl_wor_rnn, tel_wor_rnn, lol_wor_rnn, tra_wor_rnn, tes_wor_rnn, lon_wor_rnn = \
        accuracy(word_rnn, train_word, test_word, learning_rate, iter_times)
    torch.manual_seed(2023)
    torch.cuda.manual_seed(2023)
    trl_wor_cnn, tel_wor_cnn, lol_wor_cnn, tra_wor_cnn, tes_wor_cnn, lon_wor_cnn = \
        accuracy(word_cnn, train_word, test_word, learning_rate, iter_times)
 
    # random_embedding
    torch.manual_seed(2023)
    torch.cuda.manual_seed(2023)
    trl_ran_rnn, tel_ran_rnn, lol_ran_rnn, tra_ran_rnn, tes_ran_rnn, lon_ran_rnn=\
        accuracy(random_rnn, train_random, test_random, learning_rate, iter_times)
    torch.manual_seed(2023)
    torch.cuda.manual_seed(2023)
    trl_ran_cnn, tel_ran_cnn, lol_ran_cnn, tra_ran_cnn, tes_ran_cnn, lon_ran_cnn = \
        accuracy(random_cnn, train_random, test_random, learning_rate, iter_times)
 
    # glove_embedding
    torch.manual_seed(2023)
    torch.cuda.manual_seed(2023)
    trl_glo_rnn, tel_glo_rnn, lol_glo_rnn, tra_glo_rnn, tes_glo_rnn, lon_glo_rnn = \
        accuracy(glove_rnn, train_glove, test_glove, learning_rate, iter_times)
    torch.manual_seed(2023)
    torch.cuda.manual_seed(2023)
    trl_glo_cnn, tel_glo_cnn, lol_glo_cnn, tra_glo_cnn, tes_glo_cnn, lon_glo_cnn= \
        accuracy(glove_cnn, train_glove, test_glove, learning_rate, iter_times)
 
 
    x = list(range(1, iter_times+1))
    matplotlib.pyplot.subplot(2, 2, 1)
 
    matplotlib.pyplot.plot(x, trl_wor_rnn, 'k--', label='RNN+word')
    matplotlib.pyplot.plot(x, trl_wor_cnn, 'm--', label='CNN+word')
 
    matplotlib.pyplot.plot(x, trl_ran_rnn, 'r--', label='RNN+random')
    matplotlib.pyplot.plot(x, trl_ran_cnn, 'g--', label='CNN+random')
    matplotlib.pyplot.plot(x, trl_glo_rnn, 'b--', label='RNN+glove')
    matplotlib.pyplot.plot(x, trl_glo_cnn, 'y--', label='CNN+glove')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.legend(fontsize=10)
    matplotlib.pyplot.title("Train Loss")
    matplotlib.pyplot.xlabel("Iterations")
    matplotlib.pyplot.ylabel("Loss")
    matplotlib.pyplot.subplot(2, 2, 2)
 
    matplotlib.pyplot.plot(x, tel_wor_rnn, 'k--', label='RNN+word')
    matplotlib.pyplot.plot(x, tel_wor_cnn, 'm--', label='CNN+word')
 
    matplotlib.pyplot.plot(x, tel_ran_rnn, 'r--', label='RNN+random')
    matplotlib.pyplot.plot(x, tel_ran_cnn, 'g--', label='CNN+random')
    matplotlib.pyplot.plot(x, tel_glo_rnn, 'b--', label='RNN+glove')
    matplotlib.pyplot.plot(x, tel_glo_cnn, 'y--', label='CNN+glove')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.legend(fontsize=10)
    matplotlib.pyplot.title("Test Loss")
    matplotlib.pyplot.xlabel("Iterations")
    matplotlib.pyplot.ylabel("Loss")
    matplotlib.pyplot.subplot(2, 2, 3)
 
    matplotlib.pyplot.plot(x, tra_wor_rnn, 'k--', label='RNN+word')
    matplotlib.pyplot.plot(x, tra_wor_cnn, 'm--', label='CNN+word')
 
    matplotlib.pyplot.plot(x, tra_ran_rnn, 'r--', label='RNN+random')
    matplotlib.pyplot.plot(x, tra_ran_cnn, 'g--', label='CNN+random')
    matplotlib.pyplot.plot(x, tra_glo_rnn, 'b--', label='RNN+glove')
    matplotlib.pyplot.plot(x, tra_glo_cnn, 'y--', label='CNN+glove')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.legend(fontsize=10)
    matplotlib.pyplot.title("Train Accuracy")
    matplotlib.pyplot.xlabel("Iterations")
    matplotlib.pyplot.ylabel("Accuracy")
    matplotlib.pyplot.ylim(0, 1)
    matplotlib.pyplot.subplot(2, 2, 4)
 
    matplotlib.pyplot.plot(x, tes_wor_rnn, 'k--', label='RNN+word')
    matplotlib.pyplot.plot(x, tes_wor_cnn, 'm--', label='CNN+word')
 
    matplotlib.pyplot.plot(x, tes_ran_rnn, 'r--', label='RNN+random')
    matplotlib.pyplot.plot(x, tes_ran_cnn, 'g--', label='CNN+random')
    matplotlib.pyplot.plot(x, tes_glo_rnn, 'b--', label='RNN+glove')
    matplotlib.pyplot.plot(x, tes_glo_cnn, 'y--', label='CNN+glove')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.legend(fontsize=10)
    matplotlib.pyplot.title("Test Accuracy")
    matplotlib.pyplot.xlabel("Iterations")
    matplotlib.pyplot.ylabel("Accuracy")
    matplotlib.pyplot.ylim(0, 1)
    matplotlib.pyplot.tight_layout()
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(8, 8, forward=True)
    matplotlib.pyplot.savefig('result/main_plot1.jpg')
    matplotlib.pyplot.show()
    matplotlib.pyplot.subplot(2, 1, 1)
 
    matplotlib.pyplot.plot(x, lon_wor_rnn, 'k--', label='RNN+word')
    matplotlib.pyplot.plot(x, lon_wor_cnn, 'm--', label='CNN+word')
 
    matplotlib.pyplot.plot(x, lon_ran_rnn, 'r--', label='RNN+random')
    matplotlib.pyplot.plot(x, lon_ran_cnn, 'g--', label='CNN+random')
    matplotlib.pyplot.plot(x, lon_glo_rnn, 'b--', label='RNN+glove')
    matplotlib.pyplot.plot(x, lon_glo_cnn, 'y--', label='CNN+glove')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.legend(fontsize=10)
    matplotlib.pyplot.title("Long Sentence Accuracy")
    matplotlib.pyplot.xlabel("Iterations")
    matplotlib.pyplot.ylabel("Accuracy")
    matplotlib.pyplot.ylim(0, 1)
    matplotlib.pyplot.subplot(2, 1, 2)
 
    matplotlib.pyplot.plot(x, lol_wor_rnn, 'k--', label='RNN+word')
    matplotlib.pyplot.plot(x, lol_wor_cnn, 'm--', label='CNN+word')
 
    matplotlib.pyplot.plot(x, lol_ran_rnn, 'r--', label='RNN+random')
    matplotlib.pyplot.plot(x, lol_ran_cnn, 'g--', label='CNN+random')
    matplotlib.pyplot.plot(x, lol_glo_rnn, 'b--', label='RNN+glove')
    matplotlib.pyplot.plot(x, lol_glo_cnn, 'y--', label='CNN+glove')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.legend(fontsize=10)
    matplotlib.pyplot.title("Long Sentence Loss")
    matplotlib.pyplot.xlabel("Iterations")
    matplotlib.pyplot.ylabel("Loss")
    matplotlib.pyplot.tight_layout()
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(8, 8, forward=True)
    matplotlib.pyplot.savefig('result/sub_plot1.jpg')
    matplotlib.pyplot.show()
