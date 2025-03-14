'''
@ Description: Transfer the logic programs generated from NTPs to the formats generated from DFOL. 
@ Author: Kun 
@ Comment time: 2022.04.30
'''

import os
import sys
from pathlib import Path
from time import time
from typing_extensions import final
path = Path(os.getcwd())
parent_path = path.parent.absolute()
sys.path.append(os.getcwd())
sys.path.append(str(parent_path))
from NLP.model.predict_extract import return_all_predicate

def replace_function_head(predicate):
    relation = predicate[:predicate.index('(')]
    relation = relation.lower()
    return relation


def clean(newpl_path, all_predicate):
    for relation in all_predicate:
        if os.path.exists(newpl_path+relation+'/ntps.pl'):
            os.remove(newpl_path+relation+'/ntps.pl')
    return 1

def main():
    task_name = 'kinship'
    logic_program_path= 'NLP/'+task_name+'/result/ntps.pl'
    newpl_path =  'NLP/'+task_name+'/result/'
    # Get all predicate name
    all_predicate = return_all_predicate(task_name, task_name, sample_walk = 0)
    clean(newpl_path, all_predicate)
    
    all_pre_flag = {}
    for i in all_predicate:
        all_pre_flag[i] = 0
    
    with open(logic_program_path,'r') as f:
        single_line = f.readline()
        while single_line:
            two_part = single_line.split('\t')
            possible = two_part[0]
            rule = two_part[1][:-1]
            head = rule[:rule.index(')')+1]
            rule = rule.replace('),',')&')
            relation = replace_function_head(head)
            new_rule = rule +  '#(' + possible + ', 0, 0)'
            with open(newpl_path+relation+'/ntps.pl', 'a') as f2:
                print(new_rule, file=f2)
                f2.close()
            all_pre_flag[relation] = 1
            single_line=f.readline()
    for i in all_pre_flag:
        if all_pre_flag[i] == 0:
            with open(newpl_path+i+'/ntps.pl', 'a') as f2:
                tem_rule = i+'(X,Y) :- '
                print(tem_rule, file=f2)
                f2.close()
    return 0
main()