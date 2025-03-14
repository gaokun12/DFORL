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
task_name = 'nations'
test_file = 'best'

def copy(task_name, test_file, predicate):
    logic_path = os.path.join('NLP',task_name,'result', predicate, test_file+'.pl')
    logic_path_tem = os.path.join('NLP',task_name,'result', predicate, test_file+'.the_best.pl')
    shutil.copy(logic_path, logic_path_tem)
    return 1
all_predicate = return_all_predicate(task_name, None, True)
for current_predicate in all_predicate:
    copy(task_name, test_file, current_predicate)

