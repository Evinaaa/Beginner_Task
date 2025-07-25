import numpy
import csv
import random
from feature import Bag,Gram
from comparison_plot import alpha_gradient_plot
 
with open('train.tsv') as f:
    tsvreader = csv.reader(f,delimiter='\t')
    temp = list(tsvreader)
 
data = temp[1:]
max_item = int((len(data)-1)*0.2)
random.seed(1)
numpy.random.seed(1)
 
bag = Bag(data,max_item)
bag.get_words()
bag.get_matrix()
 
gram = Gram(data,dimension=2,max_item=max_item)
gram.get_words()
gram.get_matrix()
 
alpha_gradient_plot(bag,gram,10000,10)

