import matplotlib.pyplot as plt
from mysoftmax_regression import Softmax
 
def alpha_gradient_plot(bag,gram,total_times,mini_size):
    """Plot categorization verses different parameters"""
    alphas = [0.001,0.01,0.1,1,10,100,1000,10000]
 
    #Bag of words
 
    #Shuffle
    shuffle_train = list()
    shuffle_test = list()
    for alpha in alphas:
        soft = Softmax(len(bag.train),5,bag.len)
        soft.regression(bag.train_matrix,bag.train_y,alpha,total_times,"shuffle")
        r_train,r_test = soft.correct_rate(bag.train_matrix,bag.train_y,bag.test_matrix,bag.test_y)
        shuffle_train.append(r_train)
        shuffle_test.append(r_test)
 
    #Batch
    batch_train = list()
    batch_test = list()
    for alpha in alphas:
        soft = Softmax(len(bag.train), 5, bag.len)
        soft.regression(bag.train_matrix, bag.train_y, alpha, int(total_times/bag.max_item), "batch")
        r_train, r_test = soft.correct_rate(bag.train_matrix, bag.train_y, bag.test_matrix, bag.test_y)
        batch_train.append(r_train)
        batch_test.append(r_test)
 
    #Mini-batch
    mini_train = list()
    mini_test = list()
    for alpha in alphas:
        soft = Softmax(len(bag.train), 5, bag.len)
        soft.regression(bag.train_matrix, bag.train_y, alpha, total_times, "mini")
        r_train, r_test = soft.correct_rate(bag.train_matrix, bag.train_y, bag.test_matrix, bag.test_y)
        mini_train.append(r_train)
        mini_test.append(r_test)
 
    plt.subplot(2, 2, 1)
    plt.semilogx(alphas, shuffle_train, 'r--', label='shuffle')
    plt.semilogx(alphas, batch_train, 'g--', label='batch')
    plt.semilogx(alphas, mini_train, 'b--', label='mini-batch')
    plt.semilogx(alphas, shuffle_train, 'ro-', alphas, batch_train, 'g+-', alphas, mini_train, 'b^-')
    plt.legend()
    plt.title("Bag of words -- Training Set")
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.subplot(2, 2, 2)
    plt.semilogx(alphas, shuffle_test, 'r--', label='shuffle')
    plt.semilogx(alphas, batch_test, 'g--', label='batch')
    plt.semilogx(alphas, mini_test, 'b--', label='mini-batch')
    plt.semilogx(alphas, shuffle_test, 'ro-', alphas, batch_test, 'g+-', alphas, mini_test, 'b^-')
    plt.legend()
    plt.title("Bag of words -- Test Set")
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
 
    # N-gram
    # Shuffle
    shuffle_train = list()
    shuffle_test = list()
    for alpha in alphas:
        soft = Softmax(len(gram.train), 5, gram.len)
        soft.regression(gram.train_matrix, gram.train_y, alpha, total_times, "shuffle")
        r_train, r_test = soft.correct_rate(gram.train_matrix, gram.train_y, gram.test_matrix, gram.test_y)
        shuffle_train.append(r_train)
        shuffle_test.append(r_test)
 
    # Batch
    batch_train = list()
    batch_test = list()
    for alpha in alphas:
        soft = Softmax(len(gram.train), 5, gram.len)
        soft.regression(gram.train_matrix, gram.train_y, alpha, int(total_times / gram.max_item), "batch")
        r_train, r_test = soft.correct_rate(gram.train_matrix, gram.train_y, gram.test_matrix, gram.test_y)
        batch_train.append(r_train)
        batch_test.append(r_test)
 
    # Mini-batch
    mini_train = list()
    mini_test = list()
    for alpha in alphas:
        soft = Softmax(len(gram.train), 5, gram.len)
        soft.regression(gram.train_matrix, gram.train_y, alpha, int(total_times / mini_size), "mini", mini_size)
        r_train, r_test = soft.correct_rate(gram.train_matrix, gram.train_y, gram.test_matrix, gram.test_y)
        mini_train.append(r_train)
        mini_test.append(r_test)
    plt.subplot(2, 2, 3)
    plt.semilogx(alphas, shuffle_train, 'r--', label='shuffle')
    plt.semilogx(alphas, batch_train, 'g--', label='batch')
    plt.semilogx(alphas, mini_train, 'b--', label='mini-batch')
    plt.semilogx(alphas, shuffle_train, 'ro-', alphas, batch_train, 'g+-', alphas, mini_train, 'b^-')
    plt.legend()
    plt.title("N-gram -- Training Set")
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.subplot(2, 2, 4)
    plt.semilogx(alphas, shuffle_test, 'r--', label='shuffle')
    plt.semilogx(alphas, batch_test, 'g--', label='batch')
    plt.semilogx(alphas, mini_test, 'b--', label='mini-batch')
    plt.semilogx(alphas, shuffle_test, 'ro-', alphas, batch_test, 'g+-', alphas, mini_test, 'b^-')
    plt.legend()
    plt.title("N-gram -- Test Set")
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()
 
 
 

