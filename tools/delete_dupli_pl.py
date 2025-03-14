import os 
import sys
from pathlib import Path
path = Path(os.getcwd())
parent_path = path.parent.absolute()
sys.path.append(os.getcwd())
sys.path.append(str(parent_path))
from NLP.model.predict_extract import return_all_predicate
import shutil
task_name = 'nations'
test_file = 'best'

def delete_duplicated(task_name, test_file, predicate):
    logic_path = os.path.join('NLP',task_name,'result', predicate, test_file+'.pl')
    logic_path_tem = os.path.join('NLP',task_name,'result', predicate, test_file+'.tem.pl')
    with open(logic_path_tem, 'w') as w:
        with open(logic_path, 'r') as r:
            line = r.readline()
            logic_program = []
            sound = []
            while line:
                head = line[:line.index(' :- ')]
                body = line[line.index(' :- ')+4:line.index('#')]
                soundness = line[line.index('#')+1:]
                body_list = body.split('& ')
                if body_list not in logic_program:
                    logic_program.append(body_list)
                    sound.append(soundness)
                line = r.readline()
            r.close()
        ini_index= 0
        for i in logic_program:
            rule = head + " :- "
            for j in i[:-1]:
                rule = rule + j + '& '
            rule += '#'+sound[ini_index]
            print(rule, file=w, end= '')
            ini_index += 1
        w.close()
    shutil.copy(logic_path_tem, logic_path)
    return 1
all_predicate = return_all_predicate(task_name, None, True)
for current_predicate in all_predicate:
    delete_duplicated(task_name, test_file,current_predicate)