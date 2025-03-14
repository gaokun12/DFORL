'''
@ Description: Transfer the data from NeuralLP to DFOL readable format. 
@ Comment time: 2022/05/3
@ Author: Kun Gao
'''

import os
sets = ['test','train','valid']

target_file_path = os.path.join('NLP','wn18rr','data','wn18rr.nl')
with open(target_file_path,'w') as writer:
    for subset in sets:
        file_path = os.path.join('Test/'+subset+'.txt')
        with open(file_path,'r') as reader:
            line = reader.readline()
            while line:
                new_data = line[:line.index('\n')].split('\t')
                if len(new_data) != 3:
                    continue
                if subset == 'test':
                    new = new_data[1]+'('+new_data[0]+','+new_data[2]+').#TEST'
                else:
                    new = new_data[1]+'('+new_data[0]+','+new_data[2]+').'
                print(new, file = writer)
                
                line = reader.readline()
            reader.close()
    writer.close()