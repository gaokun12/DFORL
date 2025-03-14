'''
@ Description: Transfer the logic programs generated from NeuralILP to the formats generated from DFOL. 
@ Author: Kun 
@ Comment time: 2022.04.30
'''

import os
def replace_function_head(predicate):
    relation = predicate[:predicate.index('(')]
    variable = predicate[predicate.index('('):]
    if 'inv_' in relation:
        return 'ERROE', -1,'ERROR'
    relation = relation.lower()
    variable_dic = {'A':'', 'B':'', 'C':''}
    variable_inedx = [variable.index('(')+1,variable.index(')')-1]
    variable_pool = ['X','Y']
    tem_index = 0
    for i in variable_inedx:
        if variable[i] == 'B':
            variable_dic['B'] = variable_pool[tem_index]
        elif variable[i] == 'C':
            variable_dic['C'] =  variable_pool[tem_index]
        elif variable[i] == 'A':
            variable_dic['A'] =  variable_pool[tem_index]
        tem_index += 1
    new_pre = relation+'(X,Y)'
    for i in variable_dic:
        if variable_dic[i] == '':
            variable_dic[i] = 'Z'
    return new_pre, variable_dic, relation

def replace_for_body(body, variable_dic):
    body_set = body.split('&')
    new_bosy = ''
    for i in body_set:
        relation = i[:i.index('(')].lower()
        if 'inv_' in relation:
            relation = relation.replace('inv_', '~')
        variable = i[i.index('('):]
        variable_index = [variable.index('(')+1, variable.index(')')-1]
        new_var = []
        for j in variable_index:
            new_var.append(variable_dic[variable[j]])
        new_pair_variable = '('+new_var[0]+','+new_var[1]+')'
        new_pre = relation + new_pair_variable
        new_pre += '& '
        new_bosy += new_pre 
    return new_bosy

def clean(logic_program_path, newpl_path):
    
    with open(logic_program_path,'r') as f:
        single_line = f.readline()
        rule = single_line.split('\t')[1][:-1]
        head = rule[:rule.index(')')+1]
        head_final, variable_dic, relation = replace_function_head(head)
        if os.path.exists(newpl_path+relation+'/neurlp.pl'):
            os.remove(newpl_path+relation+'/neurlp.pl')
        while single_line:
            rule = single_line.split('\t')[1][:-1]
            head = rule[:rule.index(')')+1]
            head_final, variable_dic, relation = replace_function_head(head)
            if os.path.exists(newpl_path+relation+'/neurlp.pl'):
                os.remove(newpl_path+relation+'/neurlp.pl')
            single_line = f.readline()
    return 1

def main(task_name):
    logic_program_path= 'NLP/'+task_name+'/result/neurallp.pl'
    newpl_path =  'NLP/'+task_name+'/result/'
    clean(logic_program_path, newpl_path)
    with open(logic_program_path,'r') as f:
        single_line = f.readline()
        while single_line:
            two_part = single_line.split('\t')
            possible = two_part[0]
            possible = possible[possible.index('(')+1: possible.index(')')]
            rule = two_part[1][:-1]
            head = rule[:rule.index(')')+1]
            body = rule[rule.index('<--')+1:]
            body = body[body.index(' ')+1:]
            body = body.replace('),',')&')
            head_final, variable_dic, relation = replace_function_head(head)
            if head_final == 'ERROE':
                single_line =f.readline()
                continue
            body_final = replace_for_body(body, variable_dic)
            if head_final == body_final[:-2]:
                single_line =f.readline()
                continue
            new_rule = head_final + ' :- ' + body_final + '#(' + possible + ', 0, 0)'
            with open(newpl_path+relation+'/neurlp.pl', 'a') as f2:
                print(new_rule, file=f2)
                f2.close()
            single_line=f.readline()
    return 0
task_name = 'wn18'
main(task_name)