import pickle 
import itertools as it
import os

data_name = 'locatedIn_S1'
predicate_name = 'locatedIn'

two_variable_data_path = os.path.join('NLP',data_name, 'data',predicate_name,'valid_index.dt')

dic = {'valid_index': [1, 3, 4, 5, 7, 10], 'template': {0: [1, 3, 4, 5], 1: [1, 4]}}
a = [0,1,2]
b = [0,1,2,3]
two_variable = list(it.permutations(a,2))
three_variable = list(it.permutations(b,2))
print(two_variable)
print(three_variable)

two_valid_predicate = {0:[],1:[]}
three_valid_predicate = {0:[],1:[]}
for i in dic['template']:
    for j in dic['template'][i]:
        two_valid_predicate[i].append(two_variable[j])

print(two_valid_predicate)
for i in two_valid_predicate:
    for j in two_valid_predicate[i]:
        three_valid_predicate[i].append(j)
        str_j = str(j)
        if '2' in str_j:
            new_j = str_j.replace('2','3')
            tu_j = (int(new_j[1]),int(new_j[-2]))
            three_valid_predicate[i].append(tu_j)
print(three_valid_predicate)
three_template  = {0:[],1:[]}

for i in three_valid_predicate:
    for j in three_valid_predicate[i]:
        index_j = three_variable.index(j)
        three_template[i].append(index_j)
print(three_template)
three_valid_index = []
for i in three_template:
    for j in three_template[i]:
        total = i * len(three_variable) + j 
        three_valid_index.append(total)
print(three_valid_index)

three_dic = {'valid_index': three_valid_index,'template':three_template}
print(three_dic)


three_variable_data_path = os.path.join('NLP','locatedIn_S3', 'data','locatedIn','valid_index.dt')
with open(three_variable_data_path, 'wb') as f:
    pickle.dump(three_dic, f)
    f.close()
three_variable_data_path_txt = os.path.join('NLP','locatedIn_S3', 'data','locatedIn','valid_index.txt')
with open(three_variable_data_path_txt, 'w') as f:
    print(three_dic, file=f)
    f.close()
