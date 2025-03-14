import numpy as np
import os
def normalization(filename, new_file, total_num, show_time = 10000):

    with open(filename, 'r') as f:
        minnum = 500
        maxnum = -500
        line = f.readline()
        round_time =0
        while line:
            round_time += 1
            line_num = float(line)
            if minnum >= line_num:
                minnum = line_num
            if maxnum <= line_num:
                maxnum = line_num
            line = f.readline()
            if round_time % show_time == 0:
                print(str(round_time*100/total_num)[0:6]+'%', end='\r')  
        f.close()
    with open(new_file, 'w') as nor_file:
        with open(filename, 'r') as old_file:
            line = old_file.readline()
            round_time = 0
            while line:
                round_time += 1
                line_num = float(line)
                nor_num = (line_num - minnum) / (maxnum - minnum)
                print(str(nor_num), file=nor_file)                
                line = old_file.readline()
                if round_time % show_time == 0:
                    print(str(round_time*100/total_num)[0:6]+'%', end='\r')  
            old_file.close()
        nor_file.close()
    return 0

def show_data(file_name, data_num = 1000):
    data = []
    with open(file_name, 'r') as f:
        line = f.readline()
        round_time = 0
        while line:
            round_time += 1
            line_num = float(line)
            data.append(line_num)
            line = f.readline()
            if round_time >= data_num:
                break
        f.close()
    int_data = []
    for i in data:
        if i >= 0.5:
            int_data.append(1)
        else:
            int_data.append(0)
    print(data)
    print(int_data)
    return 0 
if __name__ == '__main__':
    file_name = os.path.join('KnowledgeGraphEmbedding-master','predicted.csv')
    new_file_name = os.path.join('NLP','nor_pre.csv')
    # normalization(file_name, new_file_name,24220896)
    show_data(new_file_name)