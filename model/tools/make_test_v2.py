import random
dataset_names = ['nations']
targer_predicates = ['nations']
for dataset_name in dataset_names:
    for target_pre in targer_predicates:
        test_data_path = 'DFOL/'+dataset_name+'/data/'+target_pre +'_test.nl'
        data_path = 'DFOL/'+dataset_name+'/data/'+target_pre +'.nl'
        data = []
        with open(test_data_path, 'r') as f:
            single = f.readline()
            while single:
                data.append(single[:-1]+'#TEST')
                single = f.readline()
            f.close()
        with open(data_path,'a') as f:
            for i in data:
                print(i, file=f)
            f.close()
