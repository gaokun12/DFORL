'''
@ Description: This code aims to simplifies the data size in the originial dataset. The algorithm is motivated from the multiple-hop traverse algorithm. 
@ Author: Kun Gao. 
@ Data: 2022/04/16. 
@ Version: 1.0.
'''
import numpy as np 
from data_generator import show_process
import os
import time
import random 
import pickle
import shutil
import logging
logger = logging.getLogger()
logger.basicConfig = logging.basicConfig(level=logging.DEBUG)

def get_all_relation_entites(data_path, relation_entities_data_path):
    entities = set()
    relations = set()
    data_tuple = set()
    with open(data_path, 'r') as f:
        line = f.readline()
        while line:
            if '#TEST' in line:
                line = f.readline()
                continue
                # line = line[:line.index("#")]
            else:
                line = line[:line.index('\n')]
            relation = line[:line.index('(')]
            first_entitiy = line[line.index('(') + 1 : line.index(',')]
            second_entity = line[line.index(',') + 1 : line.index(')')] 
            relations.add(relation)
            entities.add(first_entitiy)
            entities.add(second_entity)
            data_tuple.add((relation, first_entitiy, second_entity))
            line = f.readline()
    predicate_data_dic = {}
    for i in relations:
        predicate_data_dic[i] = []
    for i in data_tuple:
        predicate_data_dic[i[0]].append((i[1],i[2]))
    with open(relation_entities_data_path, 'wb') as f:
        pickle.dump((list(relations), list(entities), list(data_tuple),predicate_data_dic), f)
        f.close()
    return (list(relations), list(entities), list(data_tuple),predicate_data_dic)

def get_graph_matrix(relations, entities, data, path):
    '''
    The large version for generating all data in the matrix
    '''
    number_relations = len(relations)
    number_entities = len(entities)
    all_matrix = [ [[[] for _ in range(number_entities)] for i in range(number_relations)] for _ in range(2) ]
    
    for i in data:
        relation = i[0]
        first_o = i[1]
        second_o = i[2]
        first_o_index = entities.index(first_o)
        second_o_index = entities.index(second_o)
        relation_index = relations.index(relation)
        all_matrix[0][relation_index][first_o_index].append(second_o_index)
        all_matrix[1][relation_index][second_o_index].append(first_o_index)
    with open(path, 'wb') as f:
        pickle.dump(all_matrix, f)
        f.close
    print('Build Matrices Success')    
    return all_matrix
    
    
# def get_graph_matrix(relations, entities, data, path):
    
#     all_matrix = np.zeros((len(relations),len(entities), len(entities)), dtype=bool)

#     for i in data:
#         relation = i[0]
#         first_o = i[1]
#         second_o = i[2]
#         first_o_index = entities.index(first_o)
#         second_o_index = entities.index(second_o)
#         relation_index = relations.index(relation)
#         all_matrix[relation_index][first_o_index][second_o_index] = True
#     with open(path, 'wb') as f:
#         pickle.dump(all_matrix, f)
#         f.close
#     print('Build Matrices Success')    
#     return all_matrix

def print_sample_data_to_file(sub_data_path, sub_data):
    with open(sub_data_path, 'w') as f:
        for i in list(sub_data):
            symbolic = i[0]+'('+i[1]+','+i[2]+")."
            print(symbolic, file=f)
    return 0

def traverse_from_atom(predicate_name,all_matrix, sub_data_path, predicate_data_dic, relations, entities,percent = 0.1,sampled_predicate_info_path = '', ap_mode = False):
    '''
    @ Description: The second version of sampling data. 
    @ The main idea: Choose an atom, and make the constraints in the atom as the begining, then walk with these two nodes in one step. Then collect the overlapped nodes. 
    @ ❌ Limitation: The function now only support the variable depth is equal with 1. 
    '''
    target_pairs = predicate_data_dic[predicate_name]
    if ap_mode == False:
        # Read information about whether the predicate is sampled
        try:
            with open(sampled_predicate_info_path, 'rb') as f:
                sampled_predicate = pickle.load(f)
                f.close()
        except FileNotFoundError:
            sampled_predicate = {}
            for i in target_pairs:
                sampled_predicate[i] = 0
        unsampled_predicate = []
        for i in sampled_predicate:
            if sampled_predicate[i] == 0:
                unsampled_predicate.append(i)
        if len(unsampled_predicate) == 0:
            raise ValueError("All examples are considered")
        length_random = int(percent * len(target_pairs))
        if length_random >= len(unsampled_predicate):
            length_random = len(unsampled_predicate)
        if length_random <= 5:
            logging.warning('The current length:')
            logging.warning(length_random)
            logging.warning("Increase the sampling rate")
            return -1
    else:
        unsampled_predicate = list(target_pairs)
        length_random = int(percent * len(target_pairs))
    random_target_pairs = random.sample(unsampled_predicate, k = length_random)
    if ap_mode == False:
        for i in random_target_pairs:
            sampled_predicate[i] = 1
    # Print sampled predicate 
    logging.info("unconsidered before sample/current sample/all")
    logging.info((len(unsampled_predicate),len(random_target_pairs), len(target_pairs)))
    # Update sampled predicate 
    if ap_mode == False:
        with open(sampled_predicate_info_path, 'wb') as f:
            pickle.dump(sampled_predicate,f)
            f.close()
    # Begin to sample
    ini_time = time.time()
    all_nodes = []
    current_process = 0
    for pairs in random_target_pairs:
        #print the process
        current_process += 1
        if current_process % 50 == 0:
            ini_time = show_process(50, length_random, current_process, ini_time, note = '[Sampling Time]')
        
        first_nodes  = pairs[0]
        second_nodes = pairs[1]
        first_all_nodes = traverse_destination_nodes(entities,[first_nodes], relations, all_matrix)
        sub_data_second = traverse_with_conditions(entities, [second_nodes], first_all_nodes, relations, all_matrix)
        
        seconde_all_nodes = traverse_destination_nodes(entities,[second_nodes], relations, all_matrix)
        sub_data_first = traverse_with_conditions(entities, [first_nodes], seconde_all_nodes, relations, all_matrix)
        
        if len(sub_data_first)==0 and len(sub_data_second) == 0:
            continue 
        if sub_data_first == sub_data_second == {(predicate_name,first_nodes,second_nodes)}:
            continue 
        all_nodes.append((predicate_name,first_nodes,second_nodes))
        all_nodes.extend(sub_data_second)
        all_nodes.extend(sub_data_first)
    
    no_duplicated_nodes = []
    for i in all_nodes:
        if i not in no_duplicated_nodes:
            no_duplicated_nodes.append(i)
    print_sample_data_to_file(sub_data_path, no_duplicated_nodes)
        
    return 0 

def traverse_with_conditions(entities, random_start_node, condition_nodes, relations, all_matrix, hop_number=0):
    
    '''
    Return an nodes list through walk algorithm from the start
    '''
    sub_data = set()
    for start_node in random_start_node:
        queue = []
        already_visit = list()
        queue.append((start_node,0))
        already_visit.append(start_node)
        while len(queue) > 0:
            current_node = queue[0][0]
            current_hop = queue[0][1]
            next_walk_number = current_hop + 1
            if current_hop > hop_number:
                break
            for r in relations:
                r_index = relations.index(r)
                entity_index = entities.index(current_node)
                second_one_index = all_matrix[0][r_index][entity_index]
                for second_index in second_one_index:
                    second_node = entities[second_index]
                    if second_node in condition_nodes:
                        sub_data.add((r, current_node, second_node))
                    if second_node not in already_visit:
                        already_visit.append(second_node)
                        queue.append((second_node, next_walk_number))

                first_one_index = all_matrix[1][r_index][entity_index]
                for first_index in first_one_index:
                    first_node = entities[first_index]
                    if first_node in condition_nodes:
                        sub_data.add((r, first_node, current_node))    
                    if first_node not in already_visit:
                        already_visit.append(first_node)
                        queue.append((first_node,next_walk_number))
            queue.pop(0)
    return sub_data
    
    

def traverse_destination_nodes(entities, random_start_node, relations, all_matrix, hop_number=0):
    '''
    Return an nodes list through walk algorithm from the start
    '''
    sub_data = set()
    for start_node in random_start_node:
        queue = []
        already_visit = list()
        queue.append((start_node,0))
        already_visit.append(start_node)
        while len(queue) > 0:
            current_node = queue[0][0]
            current_hop = queue[0][1]
            next_walk_number = current_hop + 1
            if current_hop > hop_number:
                break
            for r in relations:
                r_index = relations.index(r)
                entity_index = entities.index(current_node)
                second_one_index = all_matrix[0][r_index][entity_index]
                for second_index in second_one_index:
                    second_node = entities[second_index]
                    sub_data.add((r, current_node, second_node))
                    if second_node not in already_visit:
                        already_visit.append(second_node)
                        queue.append((second_node, next_walk_number))

                first_one_index = all_matrix[1][r_index][entity_index]
                for first_index in first_one_index:
                    first_node = entities[first_index]
                    sub_data.add((r, first_node, current_node))
                    if first_node not in already_visit:
                        already_visit.append(first_node)
                        queue.append((first_node,next_walk_number))
            queue.pop(0)
    return already_visit

def traverse(entities, sub_data_path, relations, all_matrix, number_start_node, hop_number, target_relation):
    '''
    The traversal algorithm in the paper. 
    - entities: all entities list;
    - sub_data_path: the path to store the sub dataset;
    - relations: a list of all relations;
    - all_matrix: the graph matrix; 
    - number_start_node: the number of nodes which are picked randomly for the first nodes to walk;
    - hop_number: the number of maximal walk steps. hop_number = variable depth + 1
    '''
    random_start_node = random.sample(entities, k = number_start_node)
    sub_data = set()
    for start_node in random_start_node:
        # Check whether the node and the target relation are in the database 
        if all_matrix[0][relations.index(target_relation)][entities.index(start_node)] == [] and all_matrix[1][relations.index(target_relation)][entities.index(start_node)] == []:
            new_node = random.sample(entities, k = 1)
            random_start_node.append(new_node[0])
            continue
        logging.info('start from:')
        logging.info(start_node)
        queue = []
        already_visit = set()
        queue.append((start_node,0))
        already_visit.add(start_node)
        while len(queue) > 0:
            # print(len(queue))
            current_node = queue[0][0]
            current_hop = queue[0][1]
            next_walk_number = current_hop + 1
            if current_hop > hop_number:
                break
            for r in relations:
                r_index = relations.index(r)
                entity_index = entities.index(current_node)
                second_one_index = all_matrix[0][r_index][entity_index]
                # second_one_index = np.where(all_matrix[r_index][entities.index(current_node),:]==True)[0]
                # print(second_one_index)
                for second_index in second_one_index:
                    second_node = entities[second_index]
                    sub_data.add((r, current_node, second_node))
                    if second_node not in already_visit:
                        already_visit.add(second_node)
                        queue.append((second_node, next_walk_number))
                # for second_index in range(len(entities)):
                #     if all_matrix[r][entities.index(current_node)][second_index] == 1:
                #         print(second_index)
                #         second_node = entities[second_index]
                #         sub_data.add((r, current_node, second_node))
                #         if second_node not in already_visit:
                #             already_visit.add(second_node)
                #             queue.append((second_node, next_walk_number))
                first_one_index = all_matrix[1][r_index][entity_index]
                # first_one_index = np.where(all_matrix[r_index][:,entities.index(current_node)]==True)[0]
                # print(first_one_index)
                for first_index in first_one_index:
                    first_node = entities[first_index]
                    sub_data.add((r, first_node, current_node))
                    if first_node not in already_visit:
                        already_visit.add(first_node)
                        queue.append((first_node,next_walk_number))
                # for first_index in range(len(entities)):
                #     if all_matrix[r][first_index][entities.index(current_node)] == 1:
                #         first_node = entities[first_index]
                #         sub_data.add((r, first_node, current_node))
                #         if first_node not in already_visit:
                #             already_visit.add(first_node)
                #             queue.append((first_node,next_walk_number))
            queue.pop(0)
    print_sample_data_to_file(sub_data_path, sub_data)
    return 0
    
def get_samle_dataset(data_set_name, predicate_name, number_starting_nodes ,variable_depth, ap_mode = False):
    logging.info("Begin to sample data... The precent of considered predicate is")
    logging.info(number_starting_nodes)
    # read the entity and the relation from the data
    data_path = os.path.join('deepDFOL',data_set_name,'data',predicate_name+'.onl')
    if not os.path.exists(data_path):
        shutil.copy(os.path.join('deepDFOL',data_set_name,'data',data_set_name+'.nl'), data_path) 
    relation_entities_data_path = os.path.join('deepDFOL', data_set_name, 'data', data_set_name+'.rnd')
    try: 
        with open(relation_entities_data_path, 'rb') as f:
            (relations , entities, data, predicate_data_dic) = pickle.load(f)
            f.close()
    except FileNotFoundError:
        (relations , entities, data, predicate_data_dic) = get_all_relation_entites(data_path, relation_entities_data_path)
        
    sub_data_path = os.path.join('deepDFOL', data_set_name, 'data', predicate_name+'.nl')
    all_matrix_path = os.path.join('deepDFOL', data_set_name, 'data', data_set_name+'.matrix')
    sampled_predicate_info_path = os.path.join('deepDFOL', data_set_name, 'data','sampled_pred',predicate_name+'.sampled')
    if not os.path.exists(os.path.join('deepDFOL', data_set_name, 'data','sampled_pred')):
        os.mkdir(os.path.join('deepDFOL', data_set_name, 'data','sampled_pred'))
    try: 
        with open(all_matrix_path, 'rb') as f:
            all_matrix = pickle.load(f)
            f.close()
    except FileNotFoundError:
        all_matrix = get_graph_matrix(relations, entities, data, all_matrix_path)
    
    # traverse(entities, sub_data_path, relations, all_matrix, number_starting_nodes, variable_depth+1, predicate_name)
    resample_flag = traverse_from_atom(predicate_name, all_matrix, sub_data_path, predicate_data_dic, relations, entities,number_starting_nodes,sampled_predicate_info_path, ap_mode = ap_mode)
    if resample_flag == -1:
        raise NameError('Sampling number too small')
    logger.info('Sampling Success')
    return 0


if __name__ == '__main__':
    data_set_name =  'locatedIn_S3_sub'
    predicate_name =  'locatedIn'
    get_samle_dataset(data_set_name, predicate_name,3,2)
