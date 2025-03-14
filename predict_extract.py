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
import nID as nID
import pickle
from tensorflow import keras
from pyDatalog import pyDatalog 
import shutil
import datetime



def prediction(task_name,t_relation): 
    '''
    Make prediction 
    '''
    data_path = 'deepDFOL/'+task_name+'/data/'
    result_path = 'deepDFOL/'+task_name+'/result/'
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

def create_database(task_name, test_mode, cap_flag, sample_walk, data_path, t_relation, all_relation):
    pyDatalog.clear()
    # ! build the variable into Datalog
    term_log = ''
    term_log += 'X,Y,Z,W,M,N,T,'
    for i in all_relation:
        term_log += i + ','
        
    term_log = term_log[0:-1]
    pyDatalog.create_terms(term_log)
    
    
    # ! Build the databse 
    if cap_flag == True:
        if sample_walk == True:
            all_trituple_path = data_path +task_name +'.onl'
        else:
            all_trituple_path = data_path +task_name +'.nl'
    else:
        if sample_walk == True:
            all_trituple_path = data_path +t_relation+'.onl'
        else:
            all_trituple_path = data_path +t_relation+'.nl'
    
    logging.info("Check the file with databae:")
    logging.info(all_trituple_path)
    
    
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
                print('Skip Test Data')
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
    return 0

def assemble_rule(body,all_relation,logic_path):
    logging.info('Read logic program to Datalog from:')
    logging.info(logic_path)
    # ! build the variable into Datalog
    term_log = ''
    term_log += 'X,Y,Z,W,M,N,T,'
    for i in all_relation:
        term_log += i + ','
        
    term_log = term_log[0:-1]
    pyDatalog.create_terms(term_log)
    
    # Check each generated rules 
    expresison_list = []
    variable_index = []
    for rules in body:
        rule_skip = False
        
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
                        try:
                            expression =  locals()[name](locals()[first_variable],locals()[second_variable])
                        except:
                            rule_skip = True
                            break
                else:
                    if negation_flag == True:
                        expression &=  ~(locals()[name](locals()[first_variable],locals()[second_variable]))
                    else:
                        try:
                            expression &=  locals()[name](locals()[first_variable],locals()[second_variable])
                        except:
                            rule_skip = True
                            break
                flag += 1
        
        if rule_skip == True:
            logging.warning("Pass current rule")
            logging.warning(rules)
            continue
        
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
        if flag != 0:
            expresison_list.append(expression)
    
    
    logging.info('Create new logic program to Datalog success.')
    return expresison_list, variable_index
    
    
First_call = True
last_relation = ''


def make_deduction_in_mrr_hit_mode(task_name,data_path, t_relation, 
result_path, logic_program_name, all_relation, hit_test_predicate = {}):
    '''
    - Check the MRR and HITS indicator with corrupted_pairs
    - No_training_mode: When open, only create database one time, else, create database every time when checking accuracy of logic programs. 
    - hit_test_predicate: A dictionary with 
    '''

    global last_relation
    if last_relation != t_relation:
        generate_search = True
        last_relation = t_relation
    else:
        generate_search = False

    logic_path = result_path + t_relation + '/' + logic_program_name        
    if not os.path.exists(logic_path):
        return -1
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
    
    # Build database
    global First_call 
    if First_call == True:
        create_database(task_name, True, True, False, data_path, t_relation, all_relation)
        First_call = False

    global expression_list, variable_index
    if generate_search == True:
        expression_list_all,variable_index = assemble_rule(body, all_relation, logic_path)
        search_index = 0 
        expression_list = []
        # for res in expression_list_all:
        #     x_index = variable_index[search_index]['X']
        #     y_index = variable_index[search_index]['Y']
        #     all_sub = np.array(res)
        #     try:
        #         two_entity = all_sub[:,[x_index, y_index]]
        #     except:
        #         two_entity = []
        #     cet = set(map(tuple, two_entity))
        #     expression_list.append(cet)
        #     search_index += 1 
        for res in expression_list_all:
            x_index = variable_index[search_index]['X']
            y_index = variable_index[search_index]['Y']
            try:
                cet = set(map(lambda obj:(obj[x_index],obj[y_index]) , res))
            except:
                cet = set([])
            expression_list.append(cet)
            search_index += 1 

    # ! each expression corresponds a rule
    # read the target predicate file 


    target_dic_set = set(hit_test_predicate)
    # The following check process follows the T_P operator of the logic program 

    research_index = 0 
    all_over_lap = set([])
    for res in expression_list: 
        # expression_list: [[[g(x),g(y),g(z)],[g(x),g(y),g(z)],[g(x),g(y),g(z)]]...]
        # num_validate = 0
        # correct = 0
        # Get the minus set and return the value to the input dic 
        if len(res) == 0:
            research_index += 1
            continue
        overlap = res.intersection(target_dic_set)
        for ele in overlap:
            if isinstance(hit_test_predicate[ele][-1], (int, float)) and hit_test_predicate[ele][-1] <= precision_rule[research_index]:
                hit_test_predicate[ele].append(precision_rule[research_index])
            else:
                hit_test_predicate[ele].append(precision_rule[research_index])
                
        all_over_lap = all_over_lap | overlap
        research_index += 1 
        
        # for re in res:
        #     # x_index = variable_index[search_index]['X']
        #     # y_index = variable_index[search_index]['Y']
        #     if x_index >= len(re) or y_index >= len(re):
        #         break
        #     num_validate += 1

    return hit_test_predicate, all_over_lap

def check_acc_sound(task_name,data_path, t_relation, 
result_path, logic_program_name, all_relation , test_mode = False, cap_flag = False, sample_walk = False):
    '''
    check acc and sound of a rule. 
    - no_training_mode: When open, only create database one time, else, create database everytime when checking accuracy of logic programs. 
    '''
    global last_relation
    if last_relation != t_relation:
        generate_search = True
        last_relation = t_relation
    else:
        generate_search = False
        
    logic_path = result_path + t_relation + '/' + logic_program_name        
    if not os.path.exists(logic_path):
        return -1
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
    
    # Build database

    global First_call 
    if First_call == True:
        create_database(task_name, test_mode, cap_flag, sample_walk, data_path, t_relation, all_relation)
        First_call = False


    global expression_list, variable_index
    if generate_search == True:
        expression_list_all,variable_index = assemble_rule(body, all_relation, logic_path)
        search_index = 0 
        expression_list = []
        for res in expression_list_all:
            x_index = variable_index[search_index]['X']
            y_index = variable_index[search_index]['Y']
            try:
                cet = set(map(lambda obj:(obj[x_index],obj[y_index]) , res))
            except:
                cet = set([])
            # all_sub = np.array(res)
            # try:
                # two_entity = all_sub[:,[x_index, y_index]]
            # except:
                # two_entity = []
            # cet = set(map(tuple, two_entity))
            expression_list.append(cet)
            search_index += 1 
    
    # ! each expression corresponds a rule
    # read the target predicate file 
    target_dic = read_target_predicate(task_name, t_relation)
    target_dic_set = set([])
    for i in target_dic:
        first_obj = i[i.index('(')+1:i.index(',')]
        second_obj = i[i.index(',')+1:i.index(')')]
        target_dic_set.add((first_obj, second_obj))
    # target_dic_set = set(target_dic)
    
    # The following check process follows the T_P operator of the logic program 
    correct_f = []

    total_overlap = set([])
    for res in expression_list: 
        num_validate = len(res)
        correct = 0
        overlap = res.intersection(target_dic_set)
        correct = len(overlap)
        
        if num_validate == 0:
            num_validate = -1
        correct_f.append((correct/num_validate ,correct, num_validate))
        total_overlap = total_overlap | overlap

    logging.info("The soundness in best according to the current check file")
    logging.info(correct_f)
    
    for i in total_overlap:
        sym = t_relation+'('+i[0]+','+i[1]+')'
        target_dic[sym] = 1
    write_target_predicate(task_name, target_dic, t_relation)
    
    if logic_program_name == 'logic_program.pl': 
        all_trituple_path_acc = 'deepDFOL/'+ task_name +'/result/'+t_relation+'/logic_program.pl'
        with open(all_trituple_path_acc, 'a') as f:
            print(correct_f, file= f)
            f.close()

    return correct_f
    
    
    
def check_accuracy_of_logic_program(task_name,data_path, t_relation, 
result_path, logic_program_name, all_relation , hit_flag = False,t_arity=1, test_mode = False, hit_test_predicate = [], cap_flag = False, sample_walk = False, no_training_mode = False):
    '''
    Input the symbolic logic program, return the correctness of each rule in the logic prgoram.
    - no_training_mode: When open, only create database one time, else, create database everytime when checking accuracy of logic programs. 
    '''
    # expresison_list, variable_index, head_relation = build_datalog_base_and_expression(data_path, t_relation, logic_program_name, task_name, result_path)
    
    # entity_path = data_path + t_relation + '/all_ent.dt' 
    # with open(entity_path,'rb') as f:
    #     all_ent = pickle.load(f)
    #     f.close()
    
    # relation_path = data_path +  t_relation + '/all_relation_dic.dt'
    # with open(relation_path,'rb') as f:
    #     all_relation = pickle.load(f)
    #     f.close()
    if no_training_mode == True:
        global last_relation
        if last_relation != t_relation:
            generate_search = True
            last_relation = t_relation
        else:
            generate_search = False
        
    logic_path = result_path + t_relation + '/' + logic_program_name        
    if not os.path.exists(logic_path):
        return -1
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
    
    # Build database
    if no_training_mode == True:
        global First_call 
        if First_call == True:
            create_database(task_name, test_mode, cap_flag, sample_walk, data_path, t_relation, all_relation)
            First_call = False
    else:
        create_database(task_name, test_mode, cap_flag, sample_walk, data_path,t_relation, all_relation)
        
    
    
    if no_training_mode == True:
        global expresison_list, variable_index
        if generate_search == True:
            expresison_list,variable_index = assemble_rule(body, all_relation, logic_path)
    else:
        expresison_list,variable_index = assemble_rule(body, all_relation,logic_path)
    
    # ! each expression corresponds a rule
    # read the target predicate file 
    if hit_flag == True:
        target_dic = hit_test_predicate
    else:
        target_dic = read_target_predicate(task_name, t_relation)
    target_dic_set = set(target_dic)
    # The following check process follows the T_P operator of the logic program 
    correct_f = []
    search_index = 0 
    for res in expresison_list: # expression_list: [[[g(x),g(y),g(z)],[g(x),g(y),g(z)],[g(x),g(y),g(z)]]...]
        num_validate = 0
        correct = 0
        x_index = variable_index[search_index]['X']
        y_index = variable_index[search_index]['Y']
        for re in res:
            # x_index = variable_index[search_index]['X']
            # y_index = variable_index[search_index]['Y']
            if x_index >= len(re) or y_index >= len(re):
                break
            num_validate += 1
            if t_arity == 2:
                first_res = re[x_index]
                second_res = re[y_index]
            elif t_arity == 1:
                first_res = re[x_index]
                second_res = re[x_index]
            final = len(locals()[head_relation](first_res,second_res)) # The ground predicate 
            # ! test mode open iff when check accuracy basedon the .nl file
            if test_mode == True:
                predicate = head_relation + '(' + first_res+',' +second_res+ ')'
                if predicate in target_dic_set:
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
                if predicate in target_dic_set:
                    target_dic[predicate] = 1
                correct += 1
        if num_validate == 0:
            num_validate = -1
        correct_f.append((correct/num_validate ,correct, num_validate))
        search_index += 1
    logging.info("The soundness in best according to the current check file")
    logging.info(correct_f)
    # write the state of target predicate into the disk 
    # When executing single task, writing each test predicate  logic in the disk 
    if hit_flag == False:
        write_target_predicate(task_name, target_dic, t_relation)
    if logic_program_name == 'logic_program.pl': 
        all_trituple_path_acc = 'deepDFOL/'+ task_name +'/result/'+t_relation+'/logic_program.pl'
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
    # if hit_flag == False:
    # clear the database 
        # pyDatalog.clear() 
    return correct_f


    
def build_best_logic_program(t_relation, task_name, index_rule,correct_threshold, best_rule_name = 'best.pl'):
    result_path = 'deepDFOL/' + task_name + '/result/' + t_relation+'/'
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


def  build_target_predicate(t_relation, task_name, test_mode=False, cap_flag = False, sample_walk = False):
    '''
    build target predicate and return a dictionary consisting the predicate 
    '''
    data_path = 'deepDFOL/'+task_name+'/data/'
    result_path = 'deepDFOL/' + task_name + '/data/' + t_relation +'/'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    all_target_predicate_dic = {}
    if cap_flag == True:
        if sample_walk == True:
            facts_path = data_path+task_name+'.onl'
        else:
            facts_path = data_path+task_name+'.nl'
    else:
        if sample_walk == True:
            facts_path = data_path+t_relation+'.onl'
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
    result_path = 'deepDFOL/' + task_name + '/data/' + t_relation+'/'
    with open(result_path+'target_pred.dt','rb') as f:
        dic = pickle.load(f)
        f.close()
    return dic

def  write_target_predicate(task_name, dic, t_relation):
    '''
    write target predicate and return a dictionary consisting the predicate 
    '''
    result_path = 'deepDFOL/' + task_name + '/data/' + t_relation+'/'
    with open(result_path+'target_pred.dt','wb') as f:
        pickle.dump(dic, f)
        f.close()
    with open(result_path+'target_pred.txt','w') as f:
        print(dic, file = f)
        f.close()
    return 0
    
def calculate_accuracy_from_target(task_name, t_relation, all_test_mode = False):
    data_path = 'deepDFOL/' + task_name + '/data/' + t_relation+'/'
    with open(data_path+'target_pred.dt','rb') as f:
        tar_dic = pickle.load(f)    
        f.close()
    all_value = tar_dic.values()
    logging.info("The sate of the target atoms:")
    logging.info(tar_dic)
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



def get_best_logic_programs(task_name, t_relation,head_pre, t_arity, variable_depth = 1, final_threshold = 0.3, sample_walk = False):
    '''
    Iterately compute the best logic program and return the best accuracy 
    '''
    data_path = 'deepDFOL/'+task_name+'/data/'
    result_path = 'deepDFOL/'+task_name+'/result/'
    model_path = result_path+ t_relation + '/model'  
    reconstructed_model = keras.models.load_model(model_path)
    # set the basic parameters 
    threshold = 1e-8
    step = 0.05
    all_res = []
    variable_number = variable_depth + 2
    
    # make the target dic in the training datasets
    build_target_predicate(t_relation=t_relation,task_name=task_name, sample_walk = sample_walk)
    
    all_relation = return_all_predicate(task_name, t_relation, sample_walk=sample_walk)
    
    while threshold <= 1:
        print("💥 Current Threshold:",threshold)
        state, weights = extract(task_name,data_path,threshold, t_relation, result_path,head_pre, reconstructed_model, variable_number)
        if state == -2:
            return -2
        if state == -1:
            break
        tem_res = check_accuracy_of_logic_program(task_name,data_path,t_relation,result_path, 'logic_program.pl',all_relation=all_relation,t_arity=t_arity, sample_walk= sample_walk)
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
    correct_list = check_accuracy_of_logic_program(task_name,data_path,t_relation,result_path, 'logic_program.pl',all_relation=all_relation, t_arity=t_arity, sample_walk = sample_walk)
    # save the best template rule's parameter 
    correct_threshold = final_threshold #! change to 1
    save_best_weights(task_name, correct_list ,weights, t_relation,correct_threshold) #! change threshold 
    
    # add the correct rule from template file into the best file 
    build_best_logic_program(t_relation, task_name, correct_list, correct_threshold) #! change threshold 
    # reini the target predicate 
    build_target_predicate(t_relation=t_relation,task_name=task_name, sample_walk = sample_walk) 
    # compute and update the target predicate state through the check function 
    correct_list = check_accuracy_of_logic_program(task_name,data_path,t_relation,result_path, 'best.pl',all_relation = all_relation,t_arity=t_arity,sample_walk = sample_walk)
    # compute the accuracy based on the target predicates 
    target_pre_acc = calculate_accuracy_from_target(task_name, t_relation)
    logging.info("The accuracy from training target predicate")
    logging.info(target_pre_acc)
    with open(result_path+t_relation+'/acc_pred.txt', 'w') as f:
        print(target_pre_acc, file=f)
        f.close()
    return target_pre_acc

    

def make_relation_entities(file_path):
    '''
    Usage: Reading .nl file and generate all symbolic predicate in a list or a dictionary. 
    The format of the generated dictionary is: {[relation] : [(pair_of_entities),...,(pair_of_entities)]}
    '''
    # Make relational_entitile 
    all_symbolic_predicate = []
    all_predicate = []
    relation = {}
    all_entities = set()
    logging.info('[Read Data] from:')
    logging.info(file_path)
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
            if first_obj not in all_entities:
                all_entities.add(first_obj)
            if second_obj not in all_entities:
                all_entities.add(second_obj)
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
    return relation, all_symbolic_predicate, list(all_entities)
    
# def indicator_build(relation, corrupted_pairs, all_symbolic_predicate, source_pairs):
#     '''
#     Return a corrupted_pairs in the dictionary. 
#     [[symbolic_corupted_predicates]:[probability (initial with 0)]]
#     The corrupted_pairs has no common elements with the facts in the both trainig and testing datasets
#     '''
#     symbolic_p = {}
#     for i in corrupted_pairs:
#         single = relation + '('+i[0]+','+i[1] + ')'
#         if single in all_symbolic_predicate:
#             continue
#         symbolic_p[i] = source_pairs
#     return symbolic_p


def check_MRR_Hits(task_name ,test_file, sample_walk):
    MRR_mean = []
    hits_number = [1,3,10]
    hits_info = [[] for i in range(len(hits_number))]
    
    data_path = 'deepDFOL/'+task_name+'/data/'
    result_path = 'deepDFOL/'+task_name+'/result/'
    
    if not os.path.exists(data_path+task_name+'.onl'):
        shutil.copy(data_path+task_name+'.nl', data_path+task_name+'.onl')
    
    test_file_path = data_path + 'test.nl'
    
    all_relation = return_all_predicate(task_name, task_name, sample_walk)


    test_r2e,_,_ = make_relation_entities(test_file_path)
    _,all_symbolic_predicate,all_ent = make_relation_entities(data_path +task_name+'.nl')
    
    all_symbolic_predicate_set = set(all_symbolic_predicate)
    def make_all_test_atoms(relation, start_flag, max_single_scanned):
        '''
        Make all corrupt atoms according each test atoms with a same relation.
        '''
        logging.info('Make all corrupt atoms for the relation:')
        logging.info(relation)
        all_test_atom = {}
        pairs_rank = {}

        # this_trun_start_flag = start_flag
        if start_flag >= len(test_r2e[relation]):
            return -1,-1,-1
        if start_flag + max_single_scanned >= len(test_r2e[relation]):
            end_flag = len(test_r2e[relation])
        else:
            end_flag  = start_flag + max_single_scanned 
        for pairs in test_r2e[relation][start_flag:end_flag]:
            # for each test facts
            all_test_atom[pairs] = []
            for kept_entity_index in range(2):
                new_pair = (pairs[0], pairs[1],kept_entity_index )
                pairs_rank [new_pair]  = []
                change_entity_index = (kept_entity_index+1) % 2
                kept_entity = pairs[kept_entity_index]
                if change_entity_index == 1:
                    corrupted_pairs = list(itertools.product([kept_entity],all_ent))
                else:
                    corrupted_pairs = list(itertools.product(all_ent,[kept_entity]))
                for i in corrupted_pairs:
                    single = relation + '('+i[0]+','+i[1] + ')'
                    if single in all_symbolic_predicate_set:
                        continue
                    if i not in all_test_atom:
                        all_test_atom[i] = []
                        all_test_atom[i].append(new_pair)
                    else:
                        all_test_atom[i].append(new_pair)
                all_test_atom[pairs].append(new_pair)
            
            
        logging.info('Make corupted atoms success!')
        return all_test_atom, pairs_rank, start_flag+max_single_scanned

    total_test = 0
    for relation in test_r2e:
        logging.info("[MRR HITS]Check relation on:")
        logging.info(relation)
        
        start_flag = 0
        maximum_fact_single = 200
        while True:
            all_test_atom, pairs_rank, start_flag  = make_all_test_atoms(relation, start_flag, maximum_fact_single)
            if all_test_atom == -1:
                break
            
            all_test_atom, overlap = make_deduction_in_mrr_hit_mode(task_name, data_path, relation, result_path, test_file+'.pl', all_relation, hit_test_predicate = all_test_atom)
            
            for i in overlap:
                for p in all_test_atom[i]:
                    if not isinstance(p, (int, float)):
                        pairs_rank[p].append((i, all_test_atom[i][-1]))
            
            logging.info("Begin computing for the test instance")
            for target_fact in pairs_rank:
                # logging.info(target_fact)
                test_pro = pairs_rank[target_fact]
                test_pro.sort(key = lambda x: x[1], reverse = True)

                the_last_pro = 1e8
                correct_rank = 1e8
                tem_correct_rank = 0
                for all_corupt_pro in test_pro:
                    if all_corupt_pro[1] < the_last_pro:
                        tem_correct_rank += 1
                        the_last_pro = all_corupt_pro[1]
                    if all_corupt_pro[0] == (target_fact[0], target_fact[1]):
                        correct_rank = tem_correct_rank
                        break
                
                
                MRR_mean.append(1/correct_rank)
                
                total_test += 1
                
                hit_index = 0
                for i in hits_number:
                    if correct_rank <= i:
                        hits_info[hit_index].append(relation+'('+target_fact[0]+','+target_fact[1]+')')
                    hit_index += 1
                    
                with open(result_path+'MRR'+test_file+'.new.dt','wb') as f:
                    pickle.dump(MRR_mean,f)
                    f.close()
                with open(result_path+'HINT'+test_file+'.new.dt','wb') as f:
                    pickle.dump(hits_info,f)
                    f.close()
                    
                MRR_mean_value = sum(MRR_mean)/ len(MRR_mean)
                
                hit_value = []
                for i in hits_info:
                    a = len(i)
                    hit_value.append(a/total_test)
                    
            with open(result_path+'MRR'+test_file+'.new.txt','a') as f:
                print(datetime.datetime.now(), file = f)
                print(MRR_mean, file=f)
                print(MRR_mean_value, file=f)
                print('instance have checked:',total_test/2, file=f)
                f.close()
                
            with open(result_path+'HINT'+test_file+'.new.txt','a') as f:
                print(datetime.datetime.now(),file = f)
                print(hits_info, file=f)
                print(hit_value, file=f)
                print('instance have checked:',total_test/2, file=f)
                f.close()
                
    
    logging.info('Check [MRR and HINTS] for [%s] Success!'%task_name)
    logging.info('MRR')
    logging.info(MRR_mean_value)
    logging.info('HITNS@[1,3,10]')
    logging.info(hit_value)
    return MRR_mean_value, hit_value

    
def save_best_weights(task_name, correct_index, weights, t_relation, correct_threshold):
    '''
    Save the parameters if we already find the best weights. 
    '''
    data_path = 'deepDFOL/' + task_name + '/data/' + t_relation+'/'
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
def check_pl(task_name, t_relation, t_arity, file_name, cap_flag = False, sample_walk = False, check_test = True, soundness_check = False):
    '''
    - In this function, the test_mode is on. (test_model_flag = True)
    - It means we check the accuracy only on test target positive examples.
    '''
    print("Checking on ", end='')
    test_model_flag = check_test 
    string_print = ''
    if test_model_flag == True:
        string_print = 'TEST'
    else:
        string_print = 'TRAIN'
    print(string_print+' data.')
    data_path = 'deepDFOL/'+task_name+'/data/'
    result_path = 'deepDFOL/'+task_name+'/result/'
    build_target_predicate(t_relation=t_relation,task_name=task_name, test_mode=test_model_flag, cap_flag=cap_flag, sample_walk = sample_walk)
    # compute and update the target predicate state through the check function 
    all_relation = return_all_predicate(task_name, t_relation, sample_walk)
    correct_list = check_acc_sound(task_name, data_path, t_relation, result_path, file_name, all_relation, test_model_flag, cap_flag, sample_walk)
    if correct_list == -1:
        return -1
    if soundness_check == True:
        return correct_list
    # compute the accuracy based on the target predicates 
    target_pre_acc = calculate_accuracy_from_target(task_name, t_relation, cap_flag)
    logging.info('Accuracy on the %s samples is:'%string_print)
    logging.info(target_pre_acc)
    return target_pre_acc

def return_all_predicate(dataset, predicate = None, sample_walk = False):
    '''
    Return all relations in the task 
    '''
    if predicate == None:
        if sample_walk == False:
            original_data_path = 'deepDFOL/'+dataset+'/data/'+dataset+'.nl'    
        else:
            original_data_path = 'deepDFOL/'+dataset+'/data/'+dataset+'.onl'  
    else:
        if sample_walk == False:
            original_data_path = 'deepDFOL/'+dataset+'/data/'+predicate+'.nl'    
        else:
            original_data_path = 'deepDFOL/'+dataset+'/data/'+predicate+'.onl' 
        # Only use once when check the accuracy for fbk
        # if not os.path.exists(original_data_path):
        #     shutil.copy('deepDFOL/'+dataset+'/data/'+dataset+'.onl' , original_data_path)
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
def open_logic_program_into_list(file_path):
    with open(file_path,'r') as f:
        line = f.readline()
        lp = []
        while line:
            lp.append(line)
            line = f.readline()
        f.close()
    return lp
def write_list_to_logic_program(lp,file_path):
    with open(file_path, 'w') as f:
        for i in lp:
            print(i, file=f, end='')
        f.close()
    return 1
def check_soundness(dataset, test_file, sample_walk = False):
    '''
    Check the accuracy from the generated logic programs. When doing this operation, remove the test on .onl file cause we test the soundness  of rules in all datasets. 
    '''
    with open('deepDFOL/'+dataset+'/result/sound_'+test_file+'.txt', 'w') as f:
        all_relation = return_all_predicate(dataset, None,  sample_walk)
        all_sound = []
        for i in all_relation:
            logging.info("[Check Sound for]")
            logic_path = 'deepDFOL/'+dataset+'/result/'+i+'/'+test_file+'.pl'
            logging.info(logic_path)
            with open(logic_path+'.log', 'w') as f_s:
                print('Success sound', file = f_s)
                f_s.close()
            lp_list = open_logic_program_into_list(logic_path)
            new_list = []
            soundness_list = check_pl(dataset, i, 2, test_file+'.pl', cap_flag = True, sample_walk=sample_walk, soundness_check = True, check_test = False)
            ini_index = 0
            for j in soundness_list:
                if j[0] >= 0.0: # previous soundness filter 0.6
                    all_sound.append(j[0])
                    old_rule = lp_list[ini_index]
                    old_rule = old_rule[:old_rule.index('#')]
                    new_rule = old_rule + '# '+str(j)+'\n'
                    new_list.append(new_rule)
                ini_index += 1
            write_list_to_logic_program(new_list, logic_path)
            if len(all_sound) == 0:
                mean_sound = 0
            else:
                mean_sound = sum(all_sound)/len(all_sound)
            logging.info(all_sound)
            logging.info(mean_sound)
            print(mean_sound, file=f)
            print(all_sound, file=f)
            f.flush()
        f.close()
    logging.info("[Check Soundness Done!]")
    return mean_sound
    

def accuracy_all_relation(dataset, test_file, sample_walk = False):
    '''
    Used for checking all mean accuracy and accuracies on test positive examples on each relation. 
    '''
    all_relation = return_all_predicate(dataset, None,  sample_walk)
    all_acc = []
    all_number = []
    failed_relation = []
    with open('deepDFOL/'+dataset+'/result/all_ac'+test_file+'.txt', 'w') as f:
        for i in all_relation:
            test_res_path = os.path.join('deepDFOL', dataset, 'result', i, 'acc_on_test_'+test_file+'.res')
            logging.info("Check the predicate:")
            logging.info(test_res_path)
            acc_single_relation,number = check_pl(dataset, i, 2, test_file+'.pl', cap_flag = True, sample_walk=sample_walk)
            if acc_single_relation == 0:
                failed_relation.append(i)
            all_acc.append(acc_single_relation)
            all_number.append(number)
            with open(test_res_path, 'w') as f_2:
                if number == 0:
                    single_relation_acc_test = 0
                else:
                    single_relation_acc_test = acc_single_relation / number
                print(single_relation_acc_test, file=f_2)
                print(acc_single_relation,'/',number,file=f_2)
                f_2.close()
            if sum(all_number) == 0:
                mean_acc = 0
            else:
                mean_acc = sum(all_acc)/sum(all_number)
            print(all_acc, '\n' ,all_number,'\n',mean_acc, file=f)
            print("all test instance", sum(all_number), file=f)
            print(failed_relation, len(failed_relation), file= f)
            f.flush()
        f.close()
    logging.info("[Check Acc Done!]")
    return all_acc


if __name__ == "__main__":
    task_name = sys.argv[2] # unml
    t_relation = sys.argv[3]
    
        
    data_path = 'deepDFOL/'+task_name+'/data/'
    result_path = 'deepDFOL/'+task_name+'/result/'

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