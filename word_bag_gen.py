import numpy as np
from pyts.bag_of_words import BagOfWords
import os

def read_data(dataset,delimiter=','):
    data = np.loadtxt(dataset, delimiter=delimiter)
    Y = data[:,0]
    X = data[:,1:]
    index = 0
    for i in Y:
        if i != 1.0:
            Y[index] = 0.0 
        index += 1
    # X = (X-X.min())/(X.max()-X.min())
    # target_shape = X.shape + (1,)
    # X = X.reshape(target_shape)
    return X, Y

def transfer(X, window_size = 4, word_size = 4):
    # X = np.array([[1,2,3,4,5,6,7,8,9,10,12]]).reshape(1, -1)
    # generate the bags in the same step 
    bow = BagOfWords(window_size=window_size, word_size=word_size,window_step = window_size, norm_mean=False,overlapping=False, norm_std=False,numerosity_reduction=False)
    
    bags = bow.transform(X)
    return bags

def make_symbolic_facts(task_name, facts_path, bags, Y):
    data_folder = os.path.join('deepDFOL',task_name, 'data')
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    with open(facts_path, 'w') as f:
        facts = []
        item_index = 0
        variable_time = len(bags[0].split(' '))
        for i in range(variable_time):
            for j in range(variable_time):
                if i < j:
                    facts.append('before(t'+str(i)+',t'+str(j)+').')
                if i == j - 1:
                    facts.append('neighbor(t'+str(i)+',t'+str(j)+').')
        # print the facts into a file
        for item in bags:
            item = item.split(' ')
            variable_number = 0
            for words in item:
                facts.append(words+'(item' + str(item_index)+',t' + str(variable_number) +').')
                variable_number += 1
            if Y[item_index] == 1:
                facts.append('target(item' + str(item_index)+',item'+str(item_index)+').')
            item_index += 1
        # print the facts into a file
        for  i in facts:
            print(i, file=f)
        f.close()
    return 1

task_name = 'GunPoint'
data_path = os.path.join('deepDFOL', task_name, 'train.csv')
facts_path = os.path.join('deepDFOL', task_name, 'data', task_name+'.nl')
windows_size = 20
X,Y = read_data(data_path, ',')
bags = transfer(X, window_size=windows_size, word_size = 10)
make_symbolic_facts(task_name, facts_path, bags, Y)

