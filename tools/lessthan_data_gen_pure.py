import numpy as np
from numpy import random
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pand

def data_gen_lessthan(max_num):

    relation_list = {'succ','lessthan'}
    data = []
    positive_fuzzy_data = []
    negative_fuzzy_data = []
    target_data = []
    for i in range(max_num):
        if i+1 != max_num:
            data.append('succ('+ str(i) +',' + str(i+1)+ ').')        
            
    for i in range(max_num):
        for j in range(max_num):
            if i<j:
                data.append('lessthan('+ str(i) +',' + str(j)+ ').')
        

    file_path = 'NLP/lessthan/data/lessthan.nl'
    with open(file_path, 'w') as f:
        for i in data:
            print(i, file = f)
    
    
    return 0

if __name__ == '__main__':
    
    data_gen_lessthan(10)
