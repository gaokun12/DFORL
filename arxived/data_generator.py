'''
@ Description: This code is used to generated the trainable data through propsoed propositionalization mothod.
@ If the data is large, we can choose the subsample rate. If the data is not large enough, we make 'dt' as the default format. 
@ Date: 2022.04.11
@ Author: Kun Gao
@ Version: 2.0
'''

# -*- coding: utf-8 -*-
import os
import sys

from pathlib import Path
import csv


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


def get_all_valide_predicate(all_substitution, relation, all_variable_permination, t_relation, t_index):
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
        if sub_index % 2000 == 0:
            print(sub_index, flush=True)   
    
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
    print("Valid predicare index are")
    print(valid_index)
    template = {}
    number_of_predicate = 0
    for i in flag_predicates:
        template[i] = []
        for j in range(len(flag_predicates[i])):
            if flag_predicates[i][j] == 1 and number_of_predicate != t_index:
                template[i].append(j)
            number_of_predicate += 1 
            
    print("Valid template")
    print(template)
    return valid_index, template
            
            
def main(dataset, t_relation = '', path_name = '' , original_data_path= '', variable_depth=1, large = False):

    relation_name = []
    variable_number = variable_depth + 2 
    variable_objects, relation_name, relation , arity_relation, target_arity = classifier(original_data_path+t_relation +'.nl', variable_number, original_data_path+t_relation, t_relation)
    

    #template = {} # ! Define the template 
    #template[0] = [0,4] # (0,2),(1,0),(1,2),(2,0),(2,1)
    #template[1] = [1] # (0,1),(0,2),(1,0),(1,2),(2,0),(2,1) neighbo(z,y) never show be one after the substitutation 
    # Assemb all the countries 
    variable_class = {}     # cityLocIn(x,y)  x->countries y->region z->subregion, cities, and regions. 
    # Then possible predicate include: locatedIn[c_2,c_2],locatedIn[c_1,c_2], locatedIn[c_1,c_3], locatedIn[c_2,c_3] neighbor[c_1,c_1]
    for i in variable_objects:
        variable_class[i] = list(variable_objects[i])
        
    # ! debug at here
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

    if large == False:
        all_substitution = list(itertools.product(*variable_value))
    else:
        all_substitution = itertools.product(*variable_value)
    
    
    # The possible predicate correspodnnig each predicates 
    relation_variable_permutations = {}
    for i in arity_relation:
        if arity_relation[i] == 1:
            empty_list = []
            for variable_index in range(variable_number):
                empty_list.append((variable_index, variable_index))
            relation_variable_permutations[i] = empty_list
        else:
            relation_variable_permutations[i] = list(itertools.permutations(range(variable_number), 2 ))

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
    
    # Save all variable permutations into the corresponding file
    a_p_a_v = []

    try:
        with open(path_name + 'data/'+t_relation+'/valid_index.dt', "rb") as f:
            res = pickle.load(f)
            valid_index = res['valid_index']
            template = res['template']
            print(res)
            f.close()
    except FileNotFoundError:
        valid_index, template= get_all_valide_predicate(all_substitution=all_substitution,relation = relation, 
                                                all_variable_permination=relation_variable_permutations,
                                                t_relation = t_relation, t_index = t_index)
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

    if large == True:
        all_substitution = itertools.product(*variable_value)

    # open the probabilistic table 
    with open(path_name+'data/'+ t_relation+'/pro.dt',"rb") as f:
        pro = pickle.load(f)
        f.close()
    
    if large == True:
        datasets_path = original_data_path + t_relation+'/data.tfr' 
        writer = tf.io.TFRecordWriter(datasets_path)
    
    
    times_substitution = 0 
    valid_substitution = 0
    for single_substitution in all_substitution: 
        '''
        - ALL substitutions based on the all objects;
        - Generate all trainable data in the formate of: x-y: [[S(x),S(y),S(z)]...][[1,1,0,0]^|number of predicates|...]
        '''
        relation_index = 0
        # The first constrain mentioned in the paper: if all of the input are 0, then the labels should not be one 
        wrong_flag = True  
        
        all_body_boolean_value = {} # used to record all boolean values of the body of the logic program
            
        for relation_name in relation: #
            y_one_data = []
            data = relation[relation_name] # Read all relational data 
            current_permination = relation_variable_permutations[relation_name]
            for variable_pair in current_permination:
                string_tuple = []
                
                # begin to save variable arrgement 
                if times_substitution == 0:
                    a_p_a_v.append(variable_pair)
                for m in variable_pair:
                    string_tuple.append(single_substitution[m])
                string_tuple = tuple(string_tuple) # like (sue, dinana)...
                if string_tuple in data:
                    # build the current symbolic relational data 
                    current_fact = relation_name+'('+string_tuple[0] +',' + string_tuple[1]+')'
                    prob_value = pro[current_fact]
                    y_one_data.append(prob_value)
                else:
                    y_one_data.append(0)
            
            all_body_boolean_value[relation_index]=y_one_data
            relation_index += 1   
        times_substitution += 1

        # Compute whether all boolean variable in the body are zero
        check_ind = 0
        for i in all_body_boolean_value:
            label_list=[]
            for acq in template[check_ind]:     #Check the tempalte data
                label_list.append( all_body_boolean_value[i][acq])
            if 1 in label_list:
                # ! change to false to check whether accuracy of neural predicate are improved
                wrong_flag = False 
                break
            check_ind += 1
        # target predicate is ancester(x,y) -> ancester(0,1) in the embedding point of view 
        target_tuple = []
        # append the subtitution value corresponding to the variable x
        target_tuple.append(single_substitution[0]) 
        # append the subtitution value corresponding to the variable y
        target_tuple.append(single_substitution[1]) 
        # The symbolic format data correspinding to the variables x and y
        target_tuple = tuple(target_tuple) 

        if target_tuple in relation[t_relation] and wrong_flag == True:
            continue
        
        body = []
        head = []
        
        # TODO STep 1: Add all non-target predicate into train dataset  
        for i in all_body_boolean_value:
            if large == False:
                train_label[i].append(all_body_boolean_value[i])
            else:
                for j in all_body_boolean_value[i]:
                    body.append(np.float64(j))

        
        # TODO Step 2: Add the target predicare firsr-order feature into the train data
        if target_tuple in relation[t_relation]:
            current_fact = t_relation+'('+target_tuple[0] +',' + target_tuple[1]+')'
            current_fact_pro = pro[current_fact]
            if large == False:
                train_label[relation_index].append([current_fact_pro])
            else:
                head.append([np.float64(current_fact_pro)])
                body.append(np.float64(current_fact_pro))
        else:
            if large == False:
                train_label[relation_index].append([0])
            else:
                head.append([np.float64(0)])
                body.append(np.float64(0))
        
        if large == True:
            tf_example = first_order_features_example(body,head)
            writer.write(tf_example.SerializeToString())
            
        valid_substitution += 1
        if times_substitution % 5000 == 0:
            print(times_substitution)

    if large == True:
        writer.close()
    

    # Print some meta-information into datasets
    with open(original_data_path + t_relation+'/number_valid_substitution.txt', 'w') as vf:
        vf.write(str(valid_substitution))
        logging.info(str(valid_substitution)+' (valid)/(all) '+str(times_substitution))
        vf.close()
        
    print("👉 Begin to save all variable permutations in all predicates...")
    with open(original_data_path + t_relation+'/predicate.txt', 'w') as vf:
        vf.write(str(a_p_a_v))
        vf.close()
    with open(original_data_path + t_relation+'/predicate.list', 'wb') as vf:
        pickle.dump((a_p_a_v), vf)
        vf.close()
    print("👉 Save success")

    


    # ? Before the following code, the data in the x is [[1,2,3],[] (all combination of the variable)]
    # ? And the label is {0:[[ the value of first relation under current per],1:[the value for the #2 relation], 3: [the values of target predicticates]]
    # ? After the fillowing code, the data in the label is [[values of #1 relation ],[values of #2]], the label is [[the value of target predicates]]
    
    if large == False:
        train_label[len(relation)+1] = train_label[len(relation)]
    
        print("👉 The generated data is:")
        print(len(train_label), len(train_label[0]), len(train_label[1]), len(train_label[2]) )
    
        final_x = []
        for i in range(len(train_label) - 1):
            final_x.append(train_label[i])

        new_train_label = train_label[len(train_label) - 1]
    

        x_file = original_data_path + t_relation+'/x_train.dt' 
        y_file = original_data_path + t_relation+'/y_train.dt' 

    
        with open(x_file, 'wb') as xf:
            pickle.dump(final_x,xf)
            xf.close()
        with open(y_file, 'wb') as yf:
            pickle.dump(new_train_label,yf)
            yf.close()

    
    all_obj_file = original_data_path + t_relation+'/all_ent.dt'
    with open(all_obj_file, 'wb') as f:
        pickle.dump(ALL_objects,f)
        f.close()    
        
    # return all the objects in the task 
    return ALL_objects , target_arity, target_predicate


    


def gen(dataset, target_relation_name, variable_depth, large = False):
    original_data_path = 'NLP/'+dataset+'/data/'
    path_name = 'NLP/'+dataset+'/'
    _, target_arity, head_pre = main(dataset, t_relation=target_relation_name, path_name=path_name,original_data_path=original_data_path, variable_depth=variable_depth, large = large)
    print("-------Generating Data Success!-------")
    return target_arity, head_pre


if __name__ == "__main__":

    args = sys.argv
    dataset = args[1] 
    target_relation_name = args[2]
    print(dataset, target_relation_name)
    original_data_path = 'NLP/'+dataset+'/data/'
    path_name = 'NLP/'+dataset+'/'
    # target_relation_name = 'locatedIn'
    # data_index = 'S1'

    # classifier(original_data_path+dataset+'.nl', 3)


    ALL_objects = main(dataset,t_relation= target_relation_name, path_name = path_name, original_data_path = original_data_path)
    print("👉 The numbet of all objects in the task:") # 276
    print(len(ALL_objects))