'''
@ Description: Add the test for a given head predicate string after the facts in the .nl file
@ Date: 2022.04.28
@ Author: Kun Gao
@ State: Activate
'''
import random
dataset_names = ['kinship']
targer_predicates = ['kinship']
for dataset_name in dataset_names:
    for target_pre in targer_predicates:
        data_path = 'NLP/'+dataset_name+'/data/'+target_pre +'_all.nl'
        test_data_path = 'NLP/'+dataset_name+'/data/'+target_pre +'.nl'
        rest_data = []
        head_data = []
        with open(data_path, 'r') as f:
            single = f.readline()
            while single:
                head = single[:single.index('(')]
                if head == target_pre:
                    head_data.append(single[:-1])
                else:
                    rest_data.append(single[:-1])
                single = f.readline()
            f.close()
        length_head = len(head_data)
        a=random.sample(range(0, length_head), int(0.1*length_head))
        for i in  a:
            head_data[i] += '#TEST'
        with open(test_data_path,'w') as f:
            for i in head_data:
                print(i, file=f)
            for i in rest_data:
                print(i, file=f)
            f.close()
        