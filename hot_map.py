from distutils.util import change_root
from time import time
import numpy as np 
import seaborn as sns
from pyts.bag_of_words import BagOfWords
import os 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
def make_features(logic_path):
    with open(logic_path, 'r') as f:
        line = f.readline()
        features = []
        while line:
            logic = line[: line.index('#')].replace(' ','')
            precision = line[line.index('#')+1:  ]
            pro = float(precision[1:precision.index(',')])
            body = logic[logic.index(':-')+2:]
            body_feature = body.split('&')
            clean_predicate = []
            for single_feature in body_feature:
                if single_feature == '':
                    continue
                clean_single = single_feature[:single_feature.index('(')]
                clean_predicate.append(clean_single)
            features.append([clean_predicate, pro])
            line = f.readline()
    return features
        
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

def save_multi_image(filename):
    pp = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
    
    
def transfer(X,Y,features ,window_size = 4, word_size = 4):
    # X = np.array([[1,2,3,4,5,6,7,8,9,10,12]]).reshape(1, -1)
    color_set = ['green', 'red', 'blue','orange']
    fig = plt.figure(figsize=(20,4)) 
    
    bow = BagOfWords(window_size=window_size, word_size=word_size, window_step= window_size,norm_mean=False,overlapping=False, norm_std=False,numerosity_reduction=False)
    bags = bow.transform(X)
    ini_index = 0
    time_intervals = window_size
    for item in X:
        ax1=plt.subplot(121) # positive 
        ax2 = plt.subplot(122) # negative
        class_label = Y[ini_index]
        if bool(class_label) == True:
            ax = ax1
        else:
            ax = ax2
        single_bag = bags[ini_index].split(' ')
        sns.lineplot(data=item, ax=ax)
        for rule_index in range(len(features)):
            # print(rule_index)
            color_area = []
            # According to each rule, we color an instance and plot them
            bag_index = 0 # store the bag information
            for i in single_bag:
                feature_list = features[rule_index][0]
                if i in feature_list:
                    color_area.append(bag_index)
                bag_index += 1
            for single_color in color_area:
                start_time=  single_color * time_intervals
                end_time = (single_color + 1) * time_intervals
                old_term = np.array(item)
                new_term = np.zeros_like(item)
                change_index = 0
                while change_index < len(new_term):
                    if change_index < end_time and change_index >= start_time:
                        new_term[change_index] = 1
                    else:
                        new_term[change_index] = np.NaN
                    change_index += 1
                new_term = old_term * new_term
                print(color_area)
                print(start_time, end_time)
                print(item[start_time : end_time])
                # sns.lineplot( data=item[start_time : end_time], color = color_set[rule_index], linewidth = 3 , ax=ax)
                sns.lineplot( data=new_term, color = color_set[rule_index], linewidth = 3 , ax=ax)
        ini_index += 1
    return bags

dataset_name = 'GunPoint'
target_relation = 'target'
logic_path = os.path.join('deepDFOL', dataset_name, 'result', target_relation, 'best.pl')
data_path = os.path.join('deepDFOL', dataset_name, 'train.csv')
pdf_path = os.path.join('deepDFOL', dataset_name, 'result','hot_map.pdf')

windows_size = 20
word_size = 10
features = make_features(logic_path)
X,Y = read_data(data_path, ',')
transfer(X[20:40],Y[20:40], features, windows_size, word_size)
save_multi_image(pdf_path)
