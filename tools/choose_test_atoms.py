import sys
import os 
from pathlib import Path
path = Path(os.getcwd())
parent_path = path.parent.absolute()
sys.path.append(os.getcwd())
sys.path.append(str(parent_path))
sys.path.append(os.path.join(os.getcwd(),'KnowledgeGraphEmbedding','codes'))
from NLP.model.predict_extract import return_all_predicate
import shutil

dataset_name = 'wn18rr_sub'
test_file_name =  os.path.join('NLP', dataset_name, 'data', 'test_old.nl')
new_test_file_name = os.path.join('NLP', dataset_name, 'data', 'test.nl')
all_predicate = return_all_predicate(dataset_name, None, False)
consider_predicate = {}
for i in all_predicate:
    consider_predicate[i] = []
each_number = 20
with open(test_file_name, 'r') as f:
    line = f.readline()
    while line:
        predicate = line[:line.index('(')]
        if len(consider_predicate[predicate]) < each_number:
            consider_predicate[predicate].append(line[:line.index('\n')])
        line= f.readline()
    f.close()
with open(new_test_file_name, 'w') as f:
    for i in consider_predicate:
        for j in consider_predicate[i]:
            print(j, file=f)
    f.close()