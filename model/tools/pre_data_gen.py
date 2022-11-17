import numpy as np
from numpy import random
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pand

def data_gen(stand_devriation = 0.1):
    all_label_value = []
    relation_list = {'zero','succ','pre'}
    data = []
    positive_fuzzy_data = []
    negative_fuzzy_data = []

    pro = random.normal(1,stand_devriation)
    latter_str = str(pro)[1:-1]
    if pro >= 1:
        latter_str = 'T'
    elif pro <= 0:
        latter_str = '.000'
    data.append('zero(0,0).'+ latter_str+'#')
    positive_fuzzy_data.append(pro)

    for i in range(20):
        pro = random.normal(1,stand_devriation)
        latter_str = str(pro)[1:-1]
        if pro >= 1:
            # pro = 1
            latter_str = 'T'
        elif pro <= 0:
            latter_str = '.000'
        data.append('succ('+ str(i) +',' + str(i+1)+ ').' + latter_str+'#')
        positive_fuzzy_data.append(pro)

        if i >= 1:
            pro = random.normal(0,stand_devriation)
            latter_str = str(pro)[1:-1]
            if pro <= 0:
                # pro = 0
                latter_str = '.000'
            elif pro >= 1:
                latter_str = 'T'
            data.append('succ('+ str(i) +',' + str(i-1)+ ').' + latter_str+'#-')
            negative_fuzzy_data.append(pro)

            pro = random.normal(1,stand_devriation)
            latter_str = str(pro)[1:-1]
            if pro >= 1:
                latter_str = 'T'
                # pro = 1
            elif pro <= 0:
                latter_str = '.000' 
            data.append('pre('+ str(i) +',' + str(i-1)+ '). '+latter_str+'#')
            positive_fuzzy_data.append(pro)
            
            pro = random.normal(0,stand_devriation)
            latter_str = str(pro)[1:-1]
            if pro <= 0:
                latter_str = '.000'
                # pro = 0
            elif pro >= 1:
                latter_str = 'T'
            data.append('pre('+ str(i) +',' + str(i+1)+ ').' + latter_str+'#-')
            negative_fuzzy_data.append(pro)

    pro = random.normal(1,stand_devriation)
    latter_str = str(pro)[1:-1]
    if pro >= 1:
        latter_str = 'T'
        # pro = 1
    elif pro <= 0:
        latter_str = '.000' 
    data.append('pre('+ str(20) +',' + str(19)+ '). '+latter_str+'#')
    positive_fuzzy_data.append(pro)
    file_path = 'DFOL/pre/data/'
    res_path = 'DFOL/pre/result/'
    with open(file_path + 'pre.nl', 'w') as f:
        for i in data:
            print(i, file=f)
        f.close()
    
    p_data = np.array(positive_fuzzy_data)
    n_data = np.array(negative_fuzzy_data)
    p_filtered = p_data[(p_data >= 0) & (p_data <= 1)]
    n_filtered = n_data[(n_data >= 0) & (n_data <= 1)]
    data = {'Positive labels':p_filtered, 'Negative labels':n_filtered}
    p_data = pand.Series(data, index = ["Positive examples labels",'Negative examples labels'])
    

    
    pd = sns.displot(data, kind="kde", fill=True)
    plt.xlabel("Probabilistic values of examples", fontsize=7)
    plt.savefig(res_path+'pd_'+str(stand_devriation)+'.pdf')
    
    return 0