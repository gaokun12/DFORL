# -*- coding: utf-8 -*-
import itertools
from json import dump
import os
import re
import sys
import numpy as np
from pathlib import Path
from numpy.core.fromnumeric import sort

from numpy.core.records import array
from tensorflow.python.keras.backend import sin, variable
from tensorflow.python.ops.gen_array_ops import concat
#path = os.getcwczd()
path = Path(os.getcwd())
parent_path = path.parent.absolute()
sys.path.append(os.getcwd())
sys.path.append(str(parent_path))
#sys.path.append(os.getcwd())
import logging
logger = logging.getLogger()
logger.basicConfig = logging.basicConfig(level=logging.DEBUG)
import DFOL.model.nID as nID
import pickle
from tensorflow import keras
from pyDatalog import pyDatalog 
import shutil
import datetime



def prediction(task_name,t_relation): 
    '''
    Make prediction 
    '''
    data_path = 'DFOL/'+task_name+'/data/'
    result_path = 'DFOL/'+task_name+'/result/'
    model = '/model'
    model_path = result_path + t_relation + model  
    reconstructed_model = keras.models.load_model(model_path)
    input = []
    truth_predicate = [[0,1,0,0,0,0,
                        0,0,0,0,0,1,
                        0,0,0,0,0,0,0],[1]]
    for i in truth_predicate:
        input.append(np.reshape((np.array(i)),[1,-1]))
    output = nID.prediction(reconstructed_model,true_value_dictionary=input)


def test_model(data_path, t_relation, result_path, start_index = 813, end_index = 816):
        
    # # extract the datasaets
    # x_file = data_path+'/data/'+data_index+'/x_test.dt' 
    # y_file = data_path+'/data/'+data_index+'/y_test.dt'
    # with open(x_file, 'rb') as xf:
    #     x = pickle.load(xf)
    #     xf.close()
    # with open(y_file, 'rb') as yf:
    #     label = pickle.load(yf)
    #     yf.close()

    
    # # print(x[0:5],len(x))
    # # time.sleep(1)
    # # print(label[2][0:5])
    # # print(len(label[2]))
    # # time.sleep(1000)
    # print(len(x))
    # print(np.argwhere( np.array(label) == 1))
    # print(x[0][813:816],x[1][813:816])
    # print(label[813:816])
    
    # model_path = data_path+'/result/'+data_index+'/model'
    # new_x = []
    # test_x = np.array(x[0:5])
    # new_x.append(test_x)
    # shape = np.shape(x[0:5])
    # new_x.append(np.ones((shape[0], 1 ),dtype= float))
    
    # extract the datasaets
    x_file = data_path+t_relation+'/x_test.dt' 
    y_file = data_path+t_relation+'/y_test.dt'
    with open(x_file, 'rb') as xf:
        x = pickle.load(xf)
        xf.close()
    with open(y_file, 'rb') as yf:
        label = pickle.load(yf)
        yf.close()
    model_path = result_path+t_relation+'/model'
    x = np.concatenate(x,axis=1)

    tem_x = []
    tem_x.append(x)

    
    #Make more data for sptial constrains and one_sum constrains
    shape_data = len(label) # Get the first shape of the training data

    # Make label for the final label data
    tem_x.append(np.ones((int(shape_data), 1 ),dtype= float)) 
    
    new_x = [] 
    new_x.append(tem_x[0][start_index:end_index,:])
    new_x.append(tem_x[1][start_index:end_index])

    nID.test_model(new_x,model_path)
    print("::> actual label is:")
    print(label[start_index:end_index])
    

# Extract the logic program from trained matrices in the neural logic program layer
def extract(task_name, data_path, threshold, t_relation, result_path, head_pre , model, variable_number): 
    with open(data_path + t_relation +'/all_relation_dic.dt','rb') as f:
        relation = list(pickle.load(f).keys())
    # The order of elements in the realtion list should follow the oreder defined in the train.py file
    rules, layer_weight= nID.extract_symbolic_logic_programs(relation=relation, data_path = data_path + t_relation,
                                               result_path = result_path + t_relation, model = model,
                                               number_of_variable=variable_number, threshold = threshold, head_pre=head_pre)
    if rules == -2:
        return -2, -2
    if rules == -1:
        return -1, layer_weight 
    print("::> All rules extracted from neural logic prgoram are:")
    with open(result_path + t_relation + '/logic_program.pl', 'w') as f:
        for i in rules:    
            print(i)
            print(i, file=f)
        f.close()
    return 0, layer_weight


def calculate_Hits(target_dic, valid_expressions, probabiliaties_rules, valid_index):
    '''
    - target_tuple = [(a,b),...,()] each element in the list are the test pair of variables under the target relation
    - valid_expressiob are the list of list, the number of this super list is equal with the number of rules. Each element in the single list 
    are the ground substitutions meeting the body of the rule 
    - probabilisties_rules: the number of element in this list is equal with the number of rules. The first elements in all list is the 
    probabilistic of the corresponding rule 
    '''
    expression_dic = {}
    probabiliaty_list  = []
    
    for i in  range(len(valid_expressions)):
        expression_dic[i] = valid_expressions[i]
    for i in range(len(probabiliaties_rules)):
        single_tuple = (probabiliaties_rules[i][0], i)
        probabiliaty_list.append(single_tuple)
    probabiliaty_list.sort(key= lambda x:x[0], reverse=True)
    


    hit_number = [1,3,10]
    hit_res = {}
    for i in hit_number:
        hit_res[i] = 0
    
    length_target_pairs = len(target_dic)
    
    for target_pairs in target_dic:
        
        result=[]
        target_tuple = target_dic[target_pairs]
        
        for h_tuple in target_tuple:
            x = h_tuple[0]
            y = h_tuple[1]
            right_label = 0
            if h_tuple == target_pairs:
                right_label = 1
            
            confirm_flag = 0
            for rule_num in range(len(probabiliaty_list)):
                x_index = valid_index[rule_num]['X']
                y_index = valid_index[rule_num]['Y']
                if probabiliaty_list[rule_num][0] == -0.0:
                    break            
                if confirm_flag == 1:
                    break
                rule_index = probabiliaty_list[rule_num][1] # Find the substitutation from the correspinding rule
                current_sub = expression_dic[rule_index]
                for xyz in current_sub:
                    if x == xyz[x_index] and y == xyz[y_index]:
                        p_tuple = probabiliaty_list[rule_num][0]
                        result.append((h_tuple, p_tuple, right_label))
                        confirm_flag = 1
                        break
            if confirm_flag == 0:
                result.append((h_tuple, -0.0, right_label))

        # sort the result and analyze whethere the target_pair in the top m result.
        result.sort(key=lambda x:x[1], reverse=True)
        
        set_hit = {}
        for i in result:
            if i[1] != -0.0:
                set_hit[i[1]] = set()
        for i in result:
            if i[1] != -0.0:
                set_hit[i[1]].add(i[0])
                if i[2] == 1:
                    set_hit[i[1]].add("Target")
        set_hit_list = list(set_hit)
        for i in hit_number:
            for item in range(i):
                if item >= len(set_hit):
                    break 
                if "Target" in set_hit[set_hit_list[item]]:
                    hit_res[i] += 1
                    break
        
        
        # calculate hit5,10,3,1
    
    for i in hit_res:
        num = hit_res[i]
        ave = num / length_target_pairs
        hit_res[i] = (num,len(target_dic) ,ave)
    print("valid/all/ave")
    print(hit_res)
    return hit_res
    


def check_accuracy_of_logic_program(task_name,data_path, t_relation, 
                                    result_path, logic_program_name, 
                                    hit_flag = False,t_arity=1, test_mode = False, hit_test_predicate = [] ,
                                    cap_flag = False):
    '''
    Input the symbolic logic program, return the correctness of each rule in the logic prgoram.
    '''
    # expresison_list, variable_index, head_relation = build_datalog_base_and_expression(data_path, t_relation, logic_program_name, task_name, result_path)
    
    entity_path = data_path + t_relation + '/all_ent.dt' 
    with open(entity_path,'rb') as f:
        all_ent = pickle.load(f)
        f.close()
    
    relation_path = data_path +  t_relation + '/all_relation_dic.dt'
    with open(relation_path,'rb') as f:
        all_relation = pickle.load(f)
        f.close()
    
    logic_path = result_path + t_relation + '/' + logic_program_name        
    
    precision_rule = [] 
    with open(logic_path, 'r') as f:
        rule = f.readline()[:-2]    
        body = []
        while rule:
            if '#' in rule:
                latter_part = rule[rule.index('#'):]
                rule = rule[:rule.index('#')]
                probability = float(latter_part[latter_part.index('(')+1:latter_part.index(',')])
                precision_rule.append(probability)
            one_body = []
            rule = rule.replace(' ','')
            rule = rule.replace('-','')
            rule = rule.split(':')
            
            head_relation = rule[0]
            head_relation = head_relation[:head_relation.index('(')]
            body_relation = rule[1]
            
            single_body = body_relation.split('&')
            for item in single_body:
                one_body.append(item)

            
            body.append(one_body)
            rule = f.readline()[:-2]
        f.close()

    # ! build the variable into Datalog
    term_log = ''
    term_log += 'X,Y,Z,W,M,N,T,'
    for i in all_relation:
        term_log += i + ','
        
    term_log = term_log[0:-1]
    pyDatalog.create_terms(term_log)
    
    # ! Build the databse 
    if cap_flag == True:
        all_trituple_path = data_path +task_name +'.nl'
    else:
        all_trituple_path = data_path +t_relation+'.nl'
    with open(all_trituple_path, 'r') as f:
        single_tri = f.readline()
        while single_tri:
            # skip the negative examples in the datasets
            # if  '@' in single_tri and '-' not in single_tri:
            if '-' in single_tri: # Do not build the negative examples
                single_tri = f.readline()
                continue
            # Do not add any test data in the file
            # !During the training process, do not inclue the testing facts. 
            # !But during the testing process, containing the testing facts.
            if test_mode == False and 'TEST' in single_tri:
                single_tri = f.readline()
                continue
            #prepare objects
            single_tri = single_tri[:single_tri.index('.')]
            relation_name = single_tri[:single_tri.index('(')]
            first_entity = single_tri[single_tri.index('(') + 1 : single_tri.index(',')]
            second_entity = single_tri[single_tri.index(',')+1 : single_tri.index(')')]
            #add to database
            + locals()[relation_name](first_entity,second_entity)
            
            single_tri = f.readline()
        f.close()
    
    # Check each generated rules 
    expresison_list = []
    variable_index = []
    for rules in body:
        # Find the order of variables x and y and z
        str = ''
        va_f = 0
        for i in rules:
            str += i
        o_variable_index = []
        for var in ['X','Y','Z','W','M','N','T']:
            if var in str:
                index = str.index(var)
                va_f = 1
            else:
                index = 1e6
            o_variable_index.append((var, index))
        
        o_variable_index.sort(key= lambda y:y[1])
        
        var_dic = {}
        for i in range(len(o_variable_index)):
            var_dic[o_variable_index[i][0]] = i
        if va_f == 1:
            variable_index.append(var_dic)
        flag = 0
        for item in rules:
            negation_flag = False
            if item == '':
                continue
            else:
                name = item[:item.index('(')]
                # if the neagation operator in the rule 
                if '~' in name:
                    negation_flag = True
                    name = name[1:]
                first_variable = item[item.index('(')+1: item.index(',')].upper()
                second_variable = item[item.index(',')+1: item.index(')')].upper()
                if flag == 0:
                    if negation_flag == True:
                        expression =  ~(locals()[name](locals()[first_variable],locals()[second_variable]))
                    else:
                        expression =  locals()[name](locals()[first_variable],locals()[second_variable])
                else:
                    if negation_flag == True:
                        expression &=  ~(locals()[name](locals()[first_variable],locals()[second_variable]))
                    else:
                        expression &=  locals()[name](locals()[first_variable],locals()[second_variable])
                flag += 1
        if flag != 0:
            expresison_list.append(expression)
    # ! each expression corresponds a rule
    # read the target predicate file 
    if hit_flag == True:
        target_dic = hit_test_predicate
    else:
        target_dic = read_target_predicate(task_name, t_relation)
    
    # The following check process follows the T_P operator of the logic program 
    correct_f = []
    search_index = 0 
    for res in expresison_list: # expression_list: [[[g(x),g(y),g(z)],[g(x),g(y),g(z)],[g(x),g(y),g(z)]]...]
        num_validate = 0
        correct = 0
        for re in res:
            x_index = variable_index[search_index]['X']
            y_index = variable_index[search_index]['Y']
            if x_index >= len(re) or y_index >= len(re):
                break
            num_validate += 1
            if t_arity == 2:
                first_res = re[x_index]
                # if first_res == 'iraq':
                #     b = 90
                second_res = re[y_index]
            elif t_arity == 1:
                first_res = re[x_index]
                second_res = re[x_index]
            final = len(locals()[head_relation](first_res,second_res)) # The ground predicate 
            # ! test mode open iff when check accuracy basedon the .nl file
            if test_mode == True:
                predicate = head_relation + '(' + first_res+',' +second_res+ ')'
                if predicate in target_dic:
                    if hit_flag == True:
                        current_precision_value = target_dic[predicate]
                        # iff the new precision are larger than the current one, updata the value 
                        if precision_rule[search_index] >= current_precision_value: 
                            target_dic[predicate] = precision_rule[search_index]
                    else:
                        target_dic[predicate] = 1
                    correct += 1    
            # ! when the test mode is false and the target ground predicate in the database
            elif final == 1:  
                predicate = head_relation + '(' + first_res+',' +second_res+ ')'
                if predicate in target_dic:
                    target_dic[predicate] = 1
                correct += 1
        if num_validate == 0:
            num_validate = -1
        correct_f.append((correct/num_validate ,correct, num_validate))
        search_index += 1
    print(correct_f)
    # write the state of target predicate into the disk 
    # When executing single task, writing each test predicate  logic in the disk 
    if hit_flag == False:
        write_target_predicate(task_name, target_dic, t_relation)
    if logic_program_name == 'logic_program.pl': 
        all_trituple_path_acc = 'DFOL/'+ task_name +'/result/'+t_relation+'/logic_program.pl'
        with open(all_trituple_path_acc, 'a') as f:
            print(correct_f, file= f)
            f.close()
        
    # if hit == True:
    #     with open(data_path +  t_relation + '/relation_entities.dt', 'rb') as f:
    #         relation_entity = pickle.load(f)
    #         f.close()
    #     res = calculate_Hits(target_tuple,expresison_list, correct_f, variable_index)
    #     print("Hits result:", res)
    
    
    if hit_flag == True:
        correct_f = target_dic
        
    return correct_f


    
def build_best_logic_program(t_relation, task_name, index_rule,correct_threshold, best_rule_name = 'best.pl'):
    result_path = 'DFOL/' + task_name + '/result/' + t_relation+'/'
    all_rule = []
    with open(result_path+'logic_program.pl','r') as f:
        single_line = f.readline()
        while single_line:
            if ':-' not in single_line:
                single_line = f.readline()
                continue
            latter = single_line[single_line.index('-')+1:]
            if '(' not in latter: # the body is empty 
                single_line = f.readline()
                continue
            all_rule.append(single_line)
            single_line = f.readline()
        f.close()
    best_rule = []
    ini_index = 0
    for i in index_rule:
        if i[0] >= correct_threshold:
            best_rule.append(all_rule[ini_index][:-1] + '#' + str(i)+'\n')
        ini_index += 1           
    with open(result_path + best_rule_name, 'a') as f:
        for i in best_rule:
            i = i[:i.index('\n')]
            print(i, file=f)
        f.close()
    return 0 


def  build_target_predicate(t_relation, task_name, test_mode=False, cap_flag = False):
    '''
    build target predicate and return a dictionary consisting the predicate 
    '''
    data_path = 'DFOL/'+task_name+'/data/'
    result_path = 'DFOL/' + task_name + '/data/' + t_relation +'/'
    all_target_predicate_dic = {}
    if cap_flag == True:
        facts_path = data_path+task_name+'.nl'
    else:
        facts_path = data_path+t_relation+'.nl'
    with open(facts_path, 'r') as f:
        single_line = f.readline()
        while single_line:
            # skip the negative examples in the datasets, we won't add the negative examples in test datasets
            if '-' in single_line:
                single_line = f.readline()
                continue
            if test_mode == True:
                # if the test mode is True, then check the accuracy based on the test datasets 
                if 'TEST' in single_line:
                    head_relation = single_line[:single_line.index('(')]
                    if head_relation == t_relation:
                        all_target_predicate_dic[single_line[:single_line.index(')')+1]] = 0
            else:
                # if the test mode is Off, then check auucracy expecting the test datasets
                if 'TEST' in single_line:
                    single_line = f.readline()
                    continue
                head_relation = single_line[:single_line.index('(')]
                if head_relation == t_relation:
                    all_target_predicate_dic[single_line[:single_line.index(')')+1]] = 0
            single_line = f.readline()
        f.close()
    with open(result_path+'target_pred.txt','w') as f:
        print(all_target_predicate_dic, file=f)
        f.close()
    with open(result_path+'target_pred.dt','wb') as f:
        pickle.dump(all_target_predicate_dic, file=f)
        f.close()
    return all_target_predicate_dic

def  read_target_predicate(task_name, t_relation):
    '''
    Read target predicate and return a dictionary consisting the predicate 
    '''
    result_path = 'DFOL/' + task_name + '/data/' + t_relation+'/'
    with open(result_path+'target_pred.dt','rb') as f:
        dic = pickle.load(f)
        f.close()
    return dic

def  write_target_predicate(task_name, dic, t_relation):
    '''
    write target predicate and return a dictionary consisting the predicate 
    '''
    result_path = 'DFOL/' + task_name + '/data/' + t_relation+'/'
    with open(result_path+'target_pred.dt','wb') as f:
        pickle.dump(dic, f)
        f.close()
    with open(result_path+'target_pred.txt','w') as f:
        print(dic, file = f)
        f.close()
    return 0
    
def calculate_accuracy_from_target(task_name, t_relation, all_test_mode = False):
    data_path = 'DFOL/' + task_name + '/data/' + t_relation+'/'
    with open(data_path+'target_pred.dt','rb') as f:
        tar_dic = pickle.load(f)    
        f.close()
    all_value = tar_dic.values()
    print("The sate of the target atoms:")
    print(tar_dic)
    all_number = len(tar_dic)
    acc = 0.0
    correct_number = 0
    for i in all_value:
        if i == 1:
            acc += 1/all_number    
            correct_number += 1
    if all_test_mode == True:
        return correct_number, all_number
    return acc



def get_best_logic_programs(task_name, t_relation,head_pre, t_arity, variable_depth = 1, final_threshold = 0.3):
    '''
    Iterately compute the best logic program and return the best accuracy 
    '''
    data_path = 'DFOL/'+task_name+'/data/'
    result_path = 'DFOL/'+task_name+'/result/'
    model_path = result_path+ t_relation + '/model'  
    reconstructed_model = keras.models.load_model(model_path)
    # set the basic parameters 
    threshold = 1e-8
    step = 0.05
    all_res = []
    variable_number = variable_depth + 2
    
    # make the target dic in the training datasets
    build_target_predicate(t_relation=t_relation,task_name=task_name)
    
    while threshold <= 1:
        print("💥 Current Threshold:",threshold)
        state, weights = extract(task_name,data_path,threshold, t_relation, result_path,head_pre, reconstructed_model, variable_number)
        if state == -2:
            return -2
        if state == -1:
            break
        tem_res = check_accuracy_of_logic_program(task_name,data_path,t_relation,result_path, 'logic_program.pl',t_arity=t_arity)
        number_of_one = 0
        average_score = 0
        length = len(tem_res)
        for i in tem_res:
            if i[0] == 1:
                number_of_one += 1
            average_score = average_score + (i[0]/length)
        all_res.append((threshold, number_of_one, average_score))
        threshold += step
    all_res.sort(key= lambda x:x[1],reverse=True)
    all_res.sort(key= lambda y:y[2],reverse=True)
    best_threshold = all_res[0][0]
    print("All thrshold and number_one and average number")
    print(all_res)
    # extract template rule from the best threshold value
    extract(task_name,data_path,best_threshold,t_relation,result_path,head_pre,reconstructed_model, variable_number )
    # compute the number rule from in the template rule
    correct_list = check_accuracy_of_logic_program(task_name,data_path,t_relation,result_path, 'logic_program.pl',t_arity=t_arity)
    # save the best template rule's parameter 
    correct_threshold = final_threshold #! change to 1
    save_best_weights(task_name, correct_list ,weights, t_relation,correct_threshold) #! change threshold 
    
    # add the correct rule from template file into the best file 
    build_best_logic_program(t_relation, task_name, correct_list, correct_threshold) #! change threshold 
    # reini the target predicate 
    build_target_predicate(t_relation=t_relation,task_name=task_name)
    # compute and update the target predicate state through the check function 
    correct_list = check_accuracy_of_logic_program(task_name,data_path,t_relation,result_path, 'best.pl',t_arity=t_arity)
    # compute the accuracy based on the target predicates 
    target_pre_acc = calculate_accuracy_from_target(task_name, t_relation)
    print("The accuracy from training target predicate")
    print(target_pre_acc)
    with open(result_path+t_relation+'/acc_pred.txt', 'w') as f:
        print(target_pre_acc, file=f)
        f.close()
    return target_pre_acc

def delete_noisy_data( x, y, logic_program= 'isa(X,Y) :- isa(Z,Y)'):
    # We do not need to implement now
    logic_program = logic_program.replace(" ","")
    head = logic_program[:logic_program.index(':')]
    body = logic_program[logic_program.index(":+1"): logic_program.index("(")]
    

def make_relation_entities(file_path):
    '''
    Usage: Reading .nl file and generate all symbolic predicate in a list or a dictionary. 
    The format of the generated dictionary is: {[relation] : [(pair_of_entities),...,(pair_of_entities)]}
    '''
    # Make relational_entitile 
    all_symbolic_predicate = []
    all_predicate = []
    relation = {}
    with open(file_path,'r') as f:
        single_line  = f.readline()        
        while single_line:
            one_perd = []
            # retrieval the relation name and two objects
            single_line = single_line[0:single_line.index(')')]
            single_line = single_line.split('(')
            relation_name = single_line[0]
            the_rest = single_line[1].split(",")
            first_obj = the_rest[0]
            second_obj = the_rest[1]
            one_perd.append(relation_name)
            one_perd.append(first_obj)
            one_perd.append(second_obj)
            all_predicate.append(one_perd)
            relation[relation_name] = []
            all_symbolic_predicate.append(relation_name+'('+first_obj+','+second_obj+')')
            single_line = f.readline()
        f.close()
    
    for pred in all_predicate:
        one_tuple = []
        one_tuple.append(pred[1])
        one_tuple.append(pred[2])
        one_tuple = tuple(one_tuple) 
        relation[pred[0]].append(one_tuple) # {'relation_name': (),(),...,()}
    return relation, all_symbolic_predicate
    
def indicator_build(relation, corrupted_pairs, all_symbolic_predicate):
    '''
    Return a corrupted_pairs in the dictionary. 
    [[symbolic_corupted_predicates]:[probability (initial with 0)]]
    The corrupted_pairs has no common elements with the facts in the both trainig and testing datasets
    '''
    symbolic_p = {}
    for i in corrupted_pairs:
        single = relation + '('+i[0]+','+i[1] + ')'
        if single in all_symbolic_predicate:
            continue
        symbolic_p[single] = 0 
    return symbolic_p


def check_Hits(task_name,t_relation ,test_file):
    MRR_mean = []
    hits_number = [1,3,10]
    hits_info = [[] for i in range(len(hits_number))]
    
    total_test = 0
    data_path = 'DFOL/'+task_name+'/data/'
    result_path = 'DFOL/'+task_name+'/result/'
    
    test_file_path = data_path + '/test.nl'
    

    test_r2e,_ = make_relation_entities(test_file_path)
    _,all_symbolic_predicate = make_relation_entities(data_path + '/'+t_relation+'.nl')
    
    
    
    all_entity_path = data_path+'all_ent.dt'
    temp_relation_name = list(test_r2e.keys())[0]
    if not os.path.exists(all_entity_path):
        shutil.copy2(data_path+temp_relation_name+'/all_ent.dt', all_entity_path) 
    with open(all_entity_path,'rb') as f:
        all_ent = pickle.load(f)
        f.close()
    for relation in test_r2e:
        for pairs in test_r2e[relation]:
            for kept_entity_index in range(2):
                change_entity_index = (kept_entity_index+1) % 2
                kept_entity = pairs[kept_entity_index]
                if change_entity_index == 1:
                    corrupted_pairs = list(itertools.product([kept_entity],all_ent))
                else:
                    corrupted_pairs = list(itertools.product(all_ent,[kept_entity]))
                symbolic_predicate_value = indicator_build(relation, corrupted_pairs, all_symbolic_predicate)
                correct_predicate = relation+'('+pairs[0]+','+pairs[1]+')'
                symbolic_predicate_value[correct_predicate] = 0
                # the target arity is 2 in both UMLS and Nations datasets 
                symbolic_predicate_prob = check_accuracy_of_logic_program(task_name, data_path, relation, result_path, test_file+'.pl',hit_flag=True, t_arity=2, test_mode=True, hit_test_predicate = symbolic_predicate_value)      
                print(symbolic_predicate_prob)
                # begin to sort the result 
                list_symbolic = []
                for i in symbolic_predicate_prob:
                    list_symbolic.append((i,symbolic_predicate_prob[i]))
                list_symbolic.sort(key=lambda tup: tup[1],reverse=True)
                if list_symbolic[0][1] == 0:
                    ini_rank = 1e8
                else:
                    ini_rank = 1
                symbolic_rank ={}
                for i in list_symbolic:
                    symbolic_rank[i[0]] = [i[1]]
                index_sym = list_symbolic[0][0]
                symbolic_rank[index_sym].append(ini_rank)
                index = 0
                while index < len(list_symbolic)-1:
                    index_sym = list_symbolic[index+1][0]
                    if list_symbolic[index+1][1] == 0:
                        symbolic_rank[index_sym].append(1e8)
                    elif list_symbolic[index][1] <= list_symbolic[index+1][1]:
                        symbolic_rank[index_sym].append(ini_rank)
                    else:
                        ini_rank += 1
                        symbolic_rank[index_sym].append(ini_rank)
                    index +=1
                corrext_rank = symbolic_rank[correct_predicate][1]
                MRR_mean.append(1/corrext_rank)
                total_test += 1
                hit_index = 0
                for i in hits_number:
                    if corrext_rank <= i:
                        hits_info[hit_index].append(correct_predicate)
                    hit_index += 1
        with open(result_path+'MRR'+test_file+'.dt','wb') as f:
            pickle.dump(MRR_mean,f)
            f.close()
        with open(result_path+'HINT'+test_file+'.dt','wb') as f:
            pickle.dump(hits_info,f)
            f.close()
        MRR_mean_value = sum(MRR_mean)/ len(MRR_mean)
        hit_value = []
        for i in hits_info:
            a = len(i)
            hit_value.append(a/total_test)
        with open(result_path+'MRR'+test_file+'.txt','a') as f:
            print(datetime.datetime.now(), file = f)
            print(MRR_mean, file=f)
            print(MRR_mean_value, file=f)
            f.close()
        with open(result_path+'HINT'+test_file+'.txt','a') as f:
            print(datetime.datetime.now(),file = f)
            print(hits_info, file=f)
            print(hit_value, file=f)
            f.close()
    

    # entity_path = data_path + current_predicate + '/all_ent.dt' 
    # with open(entity_path,'rb') as f:
    #     all_ent = pickle.load(f)
    #     f.close()
    # # All combinaiton of target logic program 
    # all_set = list(itertools.product(all_ent, all_ent))
    
    
    
    # target_relation = test_relaltion[t_relation]
    # tar_dic = {}
    # for item in target_relation:
    #     first_var =[item[0]]
    #     all_prod = list(itertools.product(first_var, all_ent))
    #     tar_dic[item] = all_prod
    
    # check_accuracy_of_logic_program(task_name, data_path, t_relation, result_path, 'logic_program.pl', True, tar_dic, t_arity=t_arity)
    print('Check success')
    print(MRR_mean_value)
    print(hit_value)
    return MRR_mean_value, hit_value

    
def save_best_weights(task_name, correct_index, weights, t_relation, correct_threshold):
    '''
    Save the parameters if we already find the best weights. 
    '''
    data_path = 'DFOL/' + task_name + '/data/' + t_relation+'/'
    correct = []
    index = 0
    for i in correct_index:
        if i[0] >= correct_threshold:  #! can change back to 1 
            correct.append(index)
        index += 1
    correct_weight  = np.array(weights)[:,correct]
    print(correct_weight)
    try:
        with open(data_path+'prior_knowledge.dt','rb') as f:
            exist_weight = pickle.load(f)
            f.close()
        new_weight = np.concatenate([exist_weight, correct_weight], axis=1)
    except FileNotFoundError:
        new_weight = correct_weight
    

    with open(data_path+'prior_knowledge.dt','wb') as f:
        pickle.dump(new_weight, f)
        f.close()
    print('SAVED WEIGHTS:')
    print(new_weight)
    return 0 



# Test mode
def check_pl(task_name, t_relation, t_arity, file_name, cap_flag = False):
    '''
    - In this function, the test_mode is on.
    '''
    test_model_flag = True
    data_path = 'DFOL/'+task_name+'/data/'
    result_path = 'DFOL/'+task_name+'/result/'
    build_target_predicate(t_relation=t_relation,task_name=task_name, test_mode=test_model_flag, cap_flag=cap_flag)
    # compute and update the target predicate state through the check function 
    correct_list = check_accuracy_of_logic_program(task_name,data_path,t_relation,result_path, file_name,t_arity=t_arity, test_mode=test_model_flag, cap_flag=cap_flag)
    # compute the accuracy based on the target predicates 
    target_pre_acc = calculate_accuracy_from_target(task_name, t_relation, cap_flag)
    print('Acc is', target_pre_acc)
    return target_pre_acc

def return_all_predicate(dataset):
    '''
    Return all relations in the task 
    '''
    original_data_path = 'DFOL/'+dataset+'/data/'+dataset+'.nl'    
    all_predicate = set([])
    with open(original_data_path,'r') as f:
            new_single_line  = f.readline()        
            while new_single_line:
                # retrieval the relation name and two objects
                predicate = new_single_line[0:new_single_line.index(')')]
                single_line = predicate.split('(')
                relation_name = single_line[0]
                all_predicate.add(relation_name)
                new_single_line = f.readline()
            f.close()
    return all_predicate

def accuracy_all_relation(dataset, test_file):
    all_relation = return_all_predicate(dataset)
    all_acc = []
    all_number = []
    failed_relation = []
    for i in all_relation:
        acc_single_relation,number = check_pl(dataset, i, 2, test_file+'.pl', cap_flag = True)
        if acc_single_relation == 0:
            failed_relation.append(i)
        all_acc.append(acc_single_relation)
        all_number.append(number)
    mean_acc = sum(all_acc)/sum(all_number)
    with open('DFOL/'+dataset+'/result/all_ac'+test_file+'.txt', 'w') as f:
        print(all_acc, all_number,mean_acc, file=f)
        print("all test instance", sum(all_number), file=f)
        print(failed_relation, len(failed_relation), file= f)
        f.close()
    return all_acc


if __name__ == "__main__":
    task_name = sys.argv[2] # unml
    t_relation = sys.argv[3]
    
        
    data_path = 'DFOL/'+task_name+'/data/'
    result_path = 'DFOL/'+task_name+'/result/'

    function = sys.argv
    print(function)
    head_pre = ''
    t_arity = 1
    if sys.argv[1] == 'extr':
        extract(task_name,data_path,None,t_relation,result_path,head_pre,None)
    elif sys.argv[1] == 'pred':
        prediction(data_path,t_relation,result_path)
    elif sys.argv[1] == 'test':
        test_model(data_path,t_relation,result_path)    
    elif sys.argv[1] == 'che':
        check_accuracy_of_logic_program(task_name,data_path,t_relation,result_path, 'logic_program.pl',t_arity=t_arity)
    elif sys.argv[1] == 'all':
        get_best_logic_programs(task_name,t_relation,head_pre)
    else:
        print("🔥 Please input extr/pred/test/che/all for different function")

# extract()
# check_accuracy_of_logic_program()

# test_model()  