import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import numpy as np



data_path = 'DFOL/gp/data/gp/'

with open(data_path+'para.dt', 'rb') as f:
    data = pickle.load(f)
    f.close()
data = np.array(data)
data = np.transpose(data)

# plt.imshow(data, cmap='hot', linewidth=0.5)

ax = sns.heatmap(data, linewidth=0.5)

t = {'mother': [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)], 'father': [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)], 'gp': [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]}
valid_index = {0: [0, 1, 3, 5], 1: [0, 1, 3, 4, 5], 2: [1, 3, 4, 5]}
var_table = {0:'x',1:'y', 2:'z'}
label_text = []

ini_index = 0
for relation_name in t:
    item = valid_index[ini_index]
    var_pair= []
    for i in item:
        var_pair.append(t[relation_name][i])
    for var in var_pair:
        first_variable = var[0]
        second_variable = var[1]
        target_pre = ''
        target_pre += relation_name + '('
        target_pre += var_table[first_variable] +','+var_table[second_variable]+')'
        label_text.append(target_pre)
    ini_index += 1

print(label_text)
ax.set_xticklabels(label_text, rotation=20, fontsize=5)
ax.set_xlabel("Valid first-order features", fontsize = 10)
ax.set_ylabel("Numbering of rules", fontsize = 10)
plt.savefig('DFOL/gp/data/head_gp.pdf')