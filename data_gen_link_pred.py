'''
@ Description: This code is used to generated the trainable data through end-to-end mothod or we compute the trainable data in a batched manner. The datsets in this code are regarded as large dataset.
@ Updated [1]: This code is useless because the end-to-end trainng manner is desprated.  (2020.04.11)
@ Date: 2022.04.04
@ Author: Kun Gao
@ Version: 1.0
@ Status: Unactivated. 
''' 
# -*- coding: utf-8 -*-
import os
import sys
import time
from pathlib import Path
import csv
import random

#path = os.getcwczd()
path = Path(os.getcwd())
parent_path = path.parent.absolute()
sys.path.append(os.getcwd())
sys.path.append(str(parent_path))    
#sys.path.append(os.getcwd())
import logging
logger = logging.getLogger()
logger.basicConfig = logging.basicConfig(level=logging.DEBUG) # this is my first trial f
import itertools
import pickle 
import numpy as np
import tensorflow as tf
from KnowledgeGraphEmbedding.codes.prediction import prediction_KG
from KnowledgeGraphEmbedding.codes.prediction import back_index

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array

def first_order_features_example(body, head):

    feature = {
        'body': _bytes_feature(serialize_array(body)),
        'head': _bytes_feature(serialize_array(head)),
    }
    
    return tf.train.Example(features=tf.train.Features(feature=feature))

def get_batch_elements(substitutation, batch_size):
    '''
    Generate a batch sized elelmetns from the iterative objects.
    '''
    ini_number = 0
    batch_sub = []
    while ini_number < batch_size:
        try:
            batch_sub.append(next(substitutation))
            ini_number += 1
        except StopIteration:
            return batch_sub
    return batch_sub
        
def get_random_batch_elements(initial_list,substitutation, batch_size, buffer_size):
    '''
    Generate a ramdom batch sized elelmetns from the iterative objects.
    Buffer_size should be larger than the batch_size
    '''
    ini_number = 0

    while ini_number < buffer_size:
        try:
            initial_list.append(next(substitutation))
            ini_number += 1
        except StopIteration:
            return random.shuffle(initial_list)
        
    random_list = random.shuffle(initial_list)
    random_batch = random_list[:batch_size]
    rest_list = random_list[batch_size:]
    
    return random_batch, rest_list


def classifier(all_relation_path, variable_number, original_data_path, t_relation = ''):
    '''
    Get some information about the relation
    '''
    all_rlation = {}
    save_all_relation = {}
    all_predicate = []
    all_object = {}
    relation = {}
    arity_relation = {}
    pro = {}
    with open(all_relation_path,'r') as f:
        new_single_line  = f.readline()        
        while new_single_line:
            one_perd = []
            # retrieval the relation name and two objects
            predicate = new_single_line[0:new_single_line.index(')')]
            if '#' in new_single_line:
                if 'TEST' in new_single_line:
                    new_single_line = f.readline()
                    continue
                if 'T.#' in new_single_line or 'T#' in new_single_line:
                    probability = 1
                else:
                    probability = new_single_line[new_single_line.index('.'):new_single_line.index('#')].replace('.','').replace(' ','').replace('#','')
                    probability = '0.'+probability
                    probability = float(probability) 
            else:
                probability = 1
            single_line = predicate.split('(')
            relation_name = single_line[0]
            the_rest = single_line[1].split(",")
            first_obj = the_rest[0]
            second_obj = the_rest[1]
            one_perd.append(relation_name)
            one_perd.append(first_obj)
            one_perd.append(second_obj)
            one_perd.append(probability)
            all_predicate.append(one_perd)
            if first_obj not in all_object:
                all_object[first_obj] = set([])
            if second_obj not in all_object:
                all_object[second_obj] = set([])
            if relation_name not in all_rlation:
                all_rlation[relation_name] = [set([]),set([])]
            relation[relation_name] = []
            arity_relation[relation_name] = 1 #  initial arity for all relation are 1
            new_single_line = f.readline()
            pro[predicate+')'] = probability
        f.close()
    all_relation_list = list(all_rlation)
    
    
    # Save all predicate name into the file 
    for i in all_relation_list:
        save_all_relation[i] = i
    with open(original_data_path+'/all_relation_dic.dt','wb') as f:
        pickle.dump(save_all_relation,f)
        f.close()
    with open(original_data_path+'/all_relation_dic.txt','w') as f:
        print(str(save_all_relation),file = f)
        f.close()  
    
    for pred in all_predicate:
        first_string = str(all_relation_list.index(pred[0])) + '-1'
        second_string = str(all_relation_list.index(pred[0])) + '-2'
        all_object[pred[1]].add(first_string)
        all_object[pred[2]].add(second_string)

    for pred in all_predicate:
        one_tuple = []
        one_tuple.append(pred[1])
        one_tuple.append(pred[2])
        one_tuple = tuple(one_tuple) 
        relation[pred[0]].append(one_tuple) # {'relation_name': (),(),...,()}

        # check the arity of all predicate 
        if pred[1] != pred[2] and arity_relation[pred[0]] == 1:
            arity_relation[pred[0]] = 2
    

    # Make variable - object dictionary
    variable_objects = {} # TODO this dictionary stores the classification of each variables in the task 
    for i in range(variable_number):
        variable_objects[i] = set([])
    
    target_arity = arity_relation[t_relation]
    if target_arity == 2:     
        for pred in all_predicate:
            if pred[0] == t_relation:
                variable_objects[0].add(pred[1])
                variable_objects[1].add(pred[2])
        # add variable into the rest of the variable 
        for obj in all_object:
            for variable_name in range(2,variable_number):
                variable_objects[variable_name].add(obj)
    elif target_arity == 1: # TODO. The arity of targetr predicate is 1. In order to make both postive and negative predicate, we ask all variables have the same classification 
        for obj in all_object:
            for variable_name in range(variable_number):
                variable_objects[variable_name].add(obj)
                # variable_objects[1].add(obj)
                # variable_objects[2].add(obj)
    
    # Save relation
    with open(original_data_path+'/relation_entities.dt','wb') as f:
        pickle.dump(relation, f)
        f.close()
    # Save the probability of all relational facts
    
                
    with open(original_data_path+'/pro.dt','wb') as f:
        pickle.dump(pro,f)
        f.close()
    with open(original_data_path+'/pro.txt', 'w') as f:
        print(pro,file=f)
        f.close()
    
    return variable_objects, list(relation.keys()), relation, arity_relation, target_arity

def get_all_object(): # This fun is made for country dataset 
    
    all_objects = {}
    objects_1 = [] 
    
    with open(original_data_path+'countries.csv') as c_csv:
        csv_reader = csv.reader(c_csv, delimiter=',')
        line = 0
        for row in csv_reader:
            if line == 0:
                line+=1
                continue
            row = row[0].split(',')[0].lower().replace(' ','_')
            objects_1.append(row)
            line += 1
        c_csv.close()
    print("👉 All countries are:")
    print(objects_1)

    #Assemble all subregions
    objects_2 = []
    with open(original_data_path+'subregions.txt') as f_sub_r:
        sub_r = f_sub_r.read()
        sub_r = sub_r.split('\n')
        objects_2 = sub_r
        f_sub_r.close()    
        
    print("👉 All subrgions are:")
    print(objects_2)
    
    
    objects_3 = []
    with open(original_data_path+'regions.txt') as f_r:
        r = f_r.read()
        r = r.split('\n')
        objects_3 = r
        f_r.close()    
        
    print("👉 All rgions are:")
    print(objects_3)
    
    all_objects[1] = objects_1
    all_objects[2] = objects_2
    all_objects[3] = objects_3
    return all_objects

def get_all_valid_predicate_subsample(all_substitution, relation, all_variable_permination, t_relation, t_index, total_substitutation, total_n_substitutation, buffer_size, batch_size):

    logging.info("Check Valid Predicate in subsample manner")
    '''
    The propositionalization method 2 mentioned by the paper:
    - Check whether there is a preidicate's value is 0 for always, and generated the corresponding valid body predicate. 
    - Different with the original function, this function adapts the subsampling process.  
    '''
    flag_predicates = {}
    relation_index = 0
    relation_name = list(relation.keys())
    target_predicate_index = relation_name.index(t_relation)
    print("Target predicate is")
    print(target_predicate_index)
    for i in (relation):
        flag_predicates[relation_index] = [0] * len(all_variable_permination[i])
        relation_index += 1
    sub_index = 0
    start_time = time.time()

    finished_sub = 0
    initial_list = []
        
    # generate all trainable data in the formate of: x-y: [[S(x),S(y),S(z)]...][[1,1,0,0]^|number of predicates|...      
    while finished_sub < total_n_substitutation:
        random_batch, initial_list = get_random_batch_elements(initial_list, all_substitution, batch_size, buffer_size)
        # Doing substitutions
        for i in random_batch:
            current_relation_index = 0
            start_time = time.time()
            for relation_name in relation:  
                current_data = relation[relation_name]
                current_predicate_index = 0
                current_permination = all_variable_permination[relation_name]
                for j in current_permination:
                    first_variable = j[0]
                    second_variable = j[1]
                    target_tuple = []
                    target_tuple.append(i[first_variable])
                    target_tuple.append(i[second_variable])
                    target_tuple = tuple(target_tuple)
                    if target_tuple in current_data:
                        flag_predicates[current_relation_index][current_predicate_index] = 1
                    current_predicate_index += 1
                current_relation_index += 1
            
            sub_index+=1
            if sub_index % 500 == 0:
                now_time = time.time()
                time_cost = float((now_time - start_time)/60.0)
                print(sub_index, total_substitutation, '%.4f'%float(sub_index*100/total_substitutation) +'%', '%.4f'%time_cost+'/'+'%.2f'%float((time_cost)*(total_substitutation/500))+'mins' , end='\r')   
                start_time = time.time()

        finished_sub += 1



    print("All predicate info is")
    print(flag_predicates)
    valid_index = []
    number_of_predicate = 0
    for i in flag_predicates:
        info = flag_predicates[i]
        for j in info:
            if j == 1 and number_of_predicate != t_index:
                valid_index.append(number_of_predicate)
            number_of_predicate += 1
            
    # Compute whether all boolean variable in the body are zero
    logging.info("Valid predicare index are")
    logging.info(valid_index)
    template = {}
    number_of_predicate = 0
    for i in flag_predicates:
        template[i] = []
        for j in range(len(flag_predicates[i])):
            if flag_predicates[i][j] == 1 and number_of_predicate != t_index:
                template[i].append(j)
            number_of_predicate += 1 
            
    logging.info("Valid template")
    logging.info(template)
    logging.info("Check Valid Predicate Success")
    return valid_index, template


def get_all_valide_predicate(all_substitution, relation, all_variable_permination, t_relation, t_index, total_substitutation):
    logging.info("Check Valid Predicate")
    '''
    The propositionalization method 2 mentioned by the paper:
    - Check whether there is a preidicate's value is 0 for always, and generated the corresponding valid body
        predicate. 
    '''
    flag_predicates = {}
    relation_index = 0
    relation_name = list(relation.keys())
    target_predicate_index = relation_name.index(t_relation)
    print("Target predicate is")
    print(target_predicate_index)
    for i in (relation):
        flag_predicates[relation_index] = [0] * len(all_variable_permination[i])
        relation_index += 1
    sub_index = 0
    start_time = time.time()
    # generate all trainable data in the formate of: x-y: [[S(x),S(y),S(z)]...][[1,1,0,0]^|number of predicates|...]
    for i in all_substitution:  
        current_relation_index = 0
        for relation_name in relation: #! fix at here 
            current_data = relation[relation_name]
            current_predicate_index = 0
            current_permination = all_variable_permination[relation_name]
            for j in current_permination:
                first_variable = j[0]
                second_variable = j[1]
                target_tuple = []
                target_tuple.append(i[first_variable])
                target_tuple.append(i[second_variable])
                target_tuple = tuple(target_tuple)
                if target_tuple in current_data:
                    flag_predicates[current_relation_index][current_predicate_index] = 1
                current_predicate_index += 1
            current_relation_index += 1
        
        sub_index+=1
        if sub_index % 500 == 0:
            now_time = time.time()
            time_cost = float((now_time - start_time)/60.0)
            print(sub_index, total_substitutation, '%.4f'%float(sub_index*100/total_substitutation) +'%', '%.4f'%time_cost+'/'+'%.2f'%float((time_cost)*(total_substitutation/500))+'mins' , end='\r')   
            start_time = time.time()
    
    print("All predicate info is")
    print(flag_predicates)
    valid_index = []
    number_of_predicate = 0
    for i in flag_predicates:
        info = flag_predicates[i]
        for j in info:
            if j == 1 and number_of_predicate != t_index:
                valid_index.append(number_of_predicate)
            number_of_predicate += 1
            
    # Compute whether all boolean variable in the body are zero
    logging.info("Valid predicare index are")
    logging.info(valid_index)
    template = {}
    number_of_predicate = 0
    for i in flag_predicates:
        template[i] = []
        for j in range(len(flag_predicates[i])):
            if flag_predicates[i][j] == 1 and number_of_predicate != t_index:
                template[i].append(j)
            number_of_predicate += 1 
            
    logging.info("Valid template")
    logging.info(template)
    logging.info("Check Valid Predicate Success")
    return valid_index, template



def return_min_max(filename, min_max_file, total_num, show_time = 10000):

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
    
    logging.info('Find the min %f and max %f' % (minnum, maxnum))
    
    with open(min_max_file, 'wb') as f:
        pickle.dump({'min':minnum, 'max':maxnum}, f)

    return {'min':minnum, 'max':maxnum}

def make_tuple_data(predicted_path, all_substitution, relation, relation_variable_permutations, all_predicate_number,args_embedding_model, batch = 2048):
    
    # Generate all trainable data in the formate of: x-y: [[S(x),S(y),S(z)]...][[1,1,0,0]^|number of predicates|...]
    with open(predicted_path,"a") as batch_data_writer:
        round_num = 0

        while True:
            round_num += 1

            batch_sub = get_batch_elements(all_substitution, batch)
            if len(batch_sub) == 0:
                break
            triples = []
            for single_substitution in batch_sub:
                ini_index = [index_list for index_list in range(len(single_substitution))]
                single_substitution = dict(zip(ini_index, single_substitution))
                for relation_name in relation:
                    # All predicates
                    current_relation_predicate =list(relation_variable_permutations[relation_name])
                    for pairs_variable in current_relation_predicate:
                        pairs_variable = list(pairs_variable)
                        symbolic_pairs = [single_substitution.get(a) if single_substitution.get(a) else a for a in pairs_variable]

                        triples.append(back_index(symbolic_pairs[0], relation_name.lower(), symbolic_pairs[1],args_embedding_model))
                        
                        round_num += 1

            
            y_score = prediction_KG(args_embedding_model, triples)
            np.savetxt(batch_data_writer, y_score)
            

            print(str(round_num * 100 /all_predicate_number)[0:6]+'%',end='\r')           
                
        batch_data_writer.close()
    logging.info("Generate All Symbolic Data (3-tuple) and make prediction Success.")
    return 0

def make_all_predicate(relation, relation_variable_permutations):
    '''
    Make all variable permutations into the corresponding file.
    '''
    a_p_a_v = []
    for relation_name in relation:
        for variable_pair in relation_variable_permutations[relation_name]:
            a_p_a_v.append(variable_pair)
    return a_p_a_v
    
    
    
def make_series(predicted_data, tf_data_path, relation, relation_variable_permutations,template,t_relation, all_data_length, nmin, nmax):
    logging.info("Begin Serialize Data:")
    writer = tf.io.TFRecordWriter(tf_data_path)
    time_predicate = 0
    valid_substitution = 0
    # The index of targer predicate
    target_index = -1
    predicate_index = 0
    for relation_name in relation:
        for variable_pair in relation_variable_permutations[relation_name]:
            # begin to save variable arrgement 
            if relation_name == t_relation and variable_pair == (0,1):
                target_index = predicate_index 
            predicate_index += 1
            
    with open(predicted_data, 'r') as nor_file:
        line = nor_file.readline()
        while line:
            # The first constrain mentioned in the paper: if all of the input are 0, then the labels should not be one 
            wrong_flag = True  
            # Used to record all boolean values of the body of the logic program
            all_body_boolean_value = {} 
            relation_index = 0
            body = []
            for relation_name in relation: 
                y_one_data = []
                current_permination = relation_variable_permutations[relation_name]
                for variable_pair in current_permination:
                    if line == None:
                        raise NameError('No enough data in the CSV file!')
                    predicted_value = float(line)
                    normalized_value = (predicted_value - nmin) / (nmax - nmin)
                    y_one_data.append(normalized_value)
                    body.append(np.float64(normalized_value))
                    time_predicate += 1
                    line = nor_file.readline()
                    
                all_body_boolean_value[relation_index]=y_one_data
                relation_index += 1   
            
            

            # Prepare the body
            # body = []
            # for i in all_body_boolean_value:
            #     for j in all_body_boolean_value[i]:
            #         body.append(np.float64(j))
            
            # Compute whether all boolean variable in the body are zero
            check_ind = 0
            for i in all_body_boolean_value:
                label_list=[]
                # Check the template data
                for acq in template[check_ind]:    
                    label_list.append( all_body_boolean_value[i][acq])
                # TODO: Strategy 1: check whether all values in templates are larger than 0.5
                for i in label_list:
                    if i >= 0.5:
                        wrong_flag = False
                    break
                check_ind += 1
            
            if body[target_index] >= 0.5 and wrong_flag == True:
                continue
            
            head = []    
            # Add the target predicare firsr-order feature into the train data
            head.append([np.float64(body[target_index])])
            body.append(np.float64(body[target_index]))
            
            tf_example = first_order_features_example(body,head)
            writer.write(tf_example.SerializeToString())
                
            valid_substitution += 1
            if time_predicate % 5000 == 0:
                print(time_predicate, all_data_length,  str(time_predicate*100/all_data_length)[0:6]+'%', end="\r")
        
        
        nor_file.close()
    
    writer.close()
    logging.info("Generate Serialize Data Success.")
    
    return valid_substitution, time_predicate

def main(dataset, t_relation = '', path_name = '' , original_data_path= '', variable_depth=1, args_embedding_model=None, actual_sub_precent = 0.2, buffer_size = 5000, batch_size = 1000):

    relation_name = []
    variable_number = variable_depth + 2 
    variable_objects, relation_name, relation, arity_relation, target_arity = classifier(original_data_path+t_relation +'.nl', variable_number, original_data_path+t_relation, t_relation)
    

    #template = {} # ! Define the template 
    #template[0] = [0,4] # (0,2),(1,0),(1,2),(2,0),(2,1)
    #template[1] = [1] # (0,1),(0,2),(1,0),(1,2),(2,0),(2,1) neighbo(z,y) never show be one after the substitutation 
    # Assemb all the countries 
    variable_class = {}     
    # cityLocIn(x,y)  x->countries y->region z->subregion, cities, and regions. 
    # Then possible predicate include: locatedIn[c_2,c_2],locatedIn[c_1,c_2], locatedIn[c_1,c_3], locatedIn[c_2,c_3] neighbor[c_1,c_1]
    
    for i in variable_objects:
        variable_class[i] = list(variable_objects[i])
        
    ALL_objects = []
    for i in variable_class:
        l = variable_class[i]
        for j in l:
            if j not in ALL_objects:
                ALL_objects.append(j)

    entities_dic_name_number = {}  #sue:0 
    
    for i in ALL_objects:
        entities_dic_name_number[i] = ALL_objects.index(i) 
    print('👉 The second dictionary is', entities_dic_name_number)
    
    print("👉 All relation in the dataset")
    print(relation)
    
    # train the neural predicates
    variable_value = variable_class.values()


    all_substitution = itertools.product(*variable_value)
    
    # Compute the number of all substitutation
    all_substitution_number = 1
    for i in variable_value:
        all_substitution_number *= len(i)
    
    total_n_substitutation = int(actual_sub_precent * all_substitution_number)
    
    # The possible predicate correspodnnig each predicates 
    relation_variable_permutations = {}
    for i in arity_relation:
        if arity_relation[i] == 1:
            empty_list = []
            for variable_index in range(variable_number):
                empty_list.append((variable_index, variable_index))
            relation_variable_permutations[i] = empty_list
        else:
            relation_variable_permutations[i] = list(itertools.permutations(range(variable_number), 2))

    # The index of the target predicate 
    find_index = 0
    for i in range(len(relation_name)):
        for j in relation_variable_permutations[relation_name[i]]:
            if arity_relation[t_relation] == 1 and j == (0,0) and i == relation_name.index(t_relation):
                t_index = find_index
                target_predicate = t_relation+"(X,X)"
            elif arity_relation[t_relation] == 2 and j == (0,1) and i == relation_name.index(t_relation):
                t_index = find_index
                target_predicate = t_relation+"(X,Y)"
            find_index += 1

    print('The index of target predicate are:')
    print(t_index)

    print('👉 all variables')
    print(relation_variable_permutations)
    with open(path_name+'data/' + t_relation+'/relation_variable.dt','wb') as f:
        pickle.dump(relation_variable_permutations, f)
        f.close()
    with open(path_name+'data/' + t_relation+'/relation_variable.txt','w') as f:
        print(relation_variable_permutations, file=f)
        f.close()    
    
    #TODO: redesign the data saver when stroing the smaller datsets
    #begin at here
    train_label = {}
    
    for i in range(len(relation)):
        train_label[i] = []
    
    # Used to store the Boolean label for the taget relational predicate
    target_predicate_index = len(relation)  
    train_label[target_predicate_index] = []
        
    print("👉 Pre-operation on the label dataset:", train_label)
    
    try:
        with open(path_name + 'data/'+t_relation+'/valid_index.dt', "rb") as f:
            res = pickle.load(f)
            valid_index = res['valid_index']
            template = res['template']
            print(res)
            f.close()
    except FileNotFoundError:
        # # Change to sub-sampleing process 
        # valid_index, template = get_all_valid_predicate_subsample(all_substitution = all_substitution, relation = relation, all_variable_permination = relation_variable_permutations, t_relation = t_relation, t_index=t_index, total_n_substitutation = total_n_substitutation, buffer_size = buffer_size, batch_size = batch_size)
        
        #The original version, without subsample process 
        valid_index, template= get_all_valide_predicate(all_substitution=all_substitution,relation = relation, 
                                                all_variable_permination=relation_variable_permutations,
                                                t_relation = t_relation, t_index = t_index, total_substitutation=all_substitution_number)
        
        res = {}
        res['valid_index'] = valid_index
        res['template'] = template
        with open(path_name + 'data/'+t_relation+'/valid_index.dt', 'wb') as f:
            pickle.dump(res, f)
            f.close()
        with open(path_name + 'data/'+t_relation+'/valid_index.txt','w') as f:
            print(str(res), file = f)
            f.close()
        print("Save template succeess")

    # -------------------------------------------------
    all_substitution = itertools.product(*variable_value)
    # all_substitution = itertools.islice(all_substitution, 3)
    
    
    # Compute the all first-order features 
    all_predicate_number = 0 
    for i in relation:
        all_predicate_number += len(relation_variable_permutations[i])
    
    # Begin generating all symbolic 3-tuple data and make prediction
    predicted_path = path_name+'data/'+ t_relation+'/predicted.csv'
    if not os.path.exists(predicted_path):
        logging.info("Begin to Save Predicted Data")
        make_tuple_data(predicted_path, all_substitution, relation, relation_variable_permutations, all_substitution_number*all_predicate_number, args_embedding_model)
        
    # Search the meta-information for min and  max value 
    min_max_data_path = original_data_path + t_relation+'/min_max.dt'
    if not os.path.exists(min_max_data_path):
        logging.info("Begin to get min and max Data")
        d_min_max = return_min_max(predicted_path, min_max_data_path, all_substitution_number * all_predicate_number)
    else: 
        with open(min_max_data_path,'rb') as f:
            d_min_max = pickle.load(f)
    
    # Save TFRecords data
    tf_datasets_path = original_data_path + t_relation+'/data.tfr' 
    if not os.path.exists(tf_datasets_path):
        valid_substitution,time_predicate = make_series(predicted_path, tf_datasets_path, relation, relation_variable_permutations, template, t_relation, all_substitution_number*all_predicate_number, d_min_max['min'], d_min_max['max'])
    
        try:
            # whether all first-order featuer under the all substitutation are equal in the prediction process and serialize process 
            assert all_substitution_number*all_predicate_number == time_predicate
        except:
            # when the prediction process cannot generate complete substitutation for all first-order featuers, generate a part of substitutation.
            # Then, we check whether all [times for substitute first-order featuer under the incomplete substitutation in the serializa process] can devide the number of all first-order features
            assert time_predicate % all_predicate_number == 0
        

        logging.info("Data Number Generated Correctly!")
        
        # Print some meta-information into datasets
        with open(original_data_path + t_relation+'/number_valid_substitution.txt', 'w') as vf:
            vf.write(str(valid_substitution))
            logging.info(str(valid_substitution)+' (valid)/(all) '+str(all_substitution_number*all_predicate_number)+' / '+ str((valid_substitution * 100 / (all_substitution_number*all_predicate_number)))[0:6]+'%' )
            vf.close()
        
    print("👉 Begin to save all variable permutations in all predicates...")
    a_p_a_v = make_all_predicate(relation, relation_variable_permutations)
    with open(original_data_path + t_relation+'/predicate.txt', 'w') as vf:
        vf.write(str(a_p_a_v))
        vf.close()
    with open(original_data_path + t_relation+'/predicate.list', 'wb') as vf:
        pickle.dump((a_p_a_v), vf)
        vf.close()
    print("👉 Save success")
    
    all_obj_file = original_data_path + t_relation+'/all_ent.dt'
    with open(all_obj_file, 'wb') as f:
        pickle.dump(ALL_objects,f)
        f.close()    
        
    # return all the objects in the task 
    return ALL_objects , target_arity, target_predicate




def gen_ete(dataset, target_relation_name, variable_depth,args_embedding_model):
    original_data_path = 'deepDFOL/'+dataset+'/data/'
    path_name = 'deepDFOL/'+dataset+'/'
    _, target_arity, head_pre = main(dataset, t_relation=target_relation_name, path_name=path_name,original_data_path=original_data_path, variable_depth=variable_depth, args_embedding_model= args_embedding_model)
    print("-------Generating Data Success!-------")
    return target_arity, head_pre


if __name__ == "__main__":

    args = sys.argv
    dataset = args[1] 
    target_relation_name = args[2]
    print(dataset, target_relation_name)
    original_data_path = 'deepDFOL/'+dataset+'/data/'
    path_name = 'deepDFOL/'+dataset+'/'
    # target_relation_name = 'locatedIn'
    # data_index = 'S1'

    # classifier(original_data_path+dataset+'.nl', 3)


    ALL_objects = main(dataset,t_relation= target_relation_name, path_name = path_name, original_data_path = original_data_path)
    print("👉 The numbet of all objects in the task:") # 276
    print(len(ALL_objects))