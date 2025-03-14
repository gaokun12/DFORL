# -*- coding: utf-8 -*-
import os
from re import M
import sys
import numpy as np
from pathlib import Path

from tensorflow.python.keras.backend import dtype, variable

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
import tensorflow as tf
import numpy as np
import argparse
import itertools
import pickle


def _parse_function(example_proto):
    feature =  {'body': tf.io.FixedLenFeature([], tf.string), 
            'head':tf.io.FixedLenFeature([], tf.string)
            }
    example = tf.io.parse_single_example(example_proto, feature)
    example_body = example['body']
    example_body = tf.io.parse_tensor(example_body, tf.double)
    # example_body = tf.reshape(example_body,[1,-1])
    
    example_head = example['head']
    example_head = tf.io.parse_tensor(example_head, tf.double)
    example_head = tf.reshape(example_head, [-1,])

    one = tf.constant([1.0])
    zero = tf.constant([0.0])
    example_body = (example_body, one)
    if prior != False:
        example_head = (example_head, one, one, zero, zero, example_head, one, one, zero, one )
    else: 
        example_head = (example_head, one, one, zero, example_head, one, one, zero, one)
    
    return example_body, example_head



def t_main(task_name, t_relation, alpha=10, learning_rate=0.001, target_arity = 2, max_epochs = 1000,n_rules = 4, prior_knowledge = False, variable_depth = 1, larger = False, batch_size = 32, verbose = 2):

    nID.LAYERS = 4 #ly 
    nID.OUTPUT = {}
    
    n_rules = n_rules # The number of rules in the nerual logic programs

    task_path = 'deepDFOL/'+task_name+'/result/'
    task_data_path = 'deepDFOL/'+task_name+'/data/'
    
    folder_name = task_path + t_relation
    data_folder = task_data_path + t_relation

    variable_arity = 2
    variable_number = variable_depth + variable_arity
    
    # Open entity files 
    with open(data_folder+'/all_ent.dt', 'rb') as f:
        entities = pickle.load(f)
        f.close()
    
    print("👉 The number of all entities is", len(entities))
    global prior
    with open(data_folder+'/all_relation_dic.dt','rb') as f:
        relations = pickle.load(f)
        f.close()
    
    try:
        with open(data_folder+'/prior_knowledge.dt', 'rb') as f:
            prior = pickle.load(f) 
            flag = not np.any(prior)
            if flag:
                print('prior_knowledge is empty')
                prior_knowledge = False
            else:
                print('prior_knowledge is not empty')
            f.close()
    except FileNotFoundError:
        prior_knowledge = False
        
    prior = prior_knowledge
    target_index = 0 # the index is 0 or 6 
    
    entities_dic = {}
    index_dic = 0 
    for i in entities:
        entities_dic[index_dic] = i 
        index_dic += 1
    print("The dictionary about entities",entities_dic)
    
    
    if larger == True:
        dataset = tf.data.TFRecordDataset(task_data_path+t_relation+'/data.tfr')
        dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        
        for i in dataset.take(1):
            body_sample = i[0]
            number_all_predicate = int(tf.shape(body_sample[0])[0].numpy())
            print(i[0]) # [[1*13],[1*1]]
            print(i[1]) # [[1*1]...[1*1]]

        with open(task_data_path+t_relation+'/number_valid_substitution.txt') as f:
            total_number = f.readline()
            total_number = int(total_number)
            f.close()
        train_number = int(0.9 * total_number)
        
        # For true randomness, we set the shuffle buffer to the full dataset size.
        # dataset = dataset.cache() # Add from smaller data. (Less than 1GB)
        dataset = dataset.shuffle(2048) # I think the buffer size can not be the whole dataset. 
        
        train_data = dataset.take(train_number)
        valid_data = dataset.skip(train_number)
        
        #  Batch after shuffling to get unique batches at each epoch.
        train_data = train_data.batch(batch_size) 
        train_data = train_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        
        valid_data = valid_data.batch(batch_size)
        valid_data = valid_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        
        #for i in train_data:
        #    print(i[0])
        #    print(i[1])
    
    else:

        # extract the datasaets
        x_file = task_data_path+t_relation+'/x_train.dt' 
        y_file = task_data_path+t_relation+'/y_train.dt'

        
        with open(x_file, 'rb') as xf:
            x = pickle.load(xf)
            xf.close()
        with open(y_file, 'rb') as yf:
            label = pickle.load(yf)
            yf.close()
            
        # Combine all predicates in all relations 
        x = np.concatenate(x,axis=1)
        #  x_test = np.concatenate(x_test,axis=1)

        number_all_predicate = int(np.shape(x)[1])
        
        new_x = []
        # new_x_test = []
        
        new_x.append(x)
        # new_x_test.append(x_test)
        
        y = []
        # y_test = []

        y.append(np.array(label))
        # y_test.append(np.array((label_test)))
            
        #Make more data for sptial constrains and one_sum constrains
        shape_data = len(label) # Get the first shape of the training data

        y.append(np.ones((int(shape_data), 1),dtype= float))
        y.append(np.ones((int(shape_data), 1),dtype = float))  # Make label for the final label data
        y.append(np.zeros((int(shape_data), 1), dtype= float))
        if prior_knowledge != False:
            y.append(np.zeros((int(shape_data), 1 ), dtype= float)) # We do not know the explicit number of rows, hence we use 1 
        
        #TODO for aux 
        y.append(np.array(label))
        y.append(np.ones((int(shape_data), 1),dtype= float))
        y.append(np.ones((int(shape_data), 1),dtype = float))  # Make label for the final label data
        y.append(np.zeros((int(shape_data), 1), dtype= float))
        y.append(np.ones((int(shape_data), 1), dtype= float))
        # ? Total 9 y 
        
        new_x.append(np.ones((int(shape_data), 1 ),dtype= float)) 
        
        
        label = y 
        x = new_x
        
        
        print('::> Length of input x',len(x), 'length of label is', len(label))


    # Define neural logic program according to the DLFIT
    print("🔧 Begin to define neural logic programs")
    # define the input layer
    nID.input_model(number_all_predicate=number_all_predicate)
    
    
    with open(data_folder+'/valid_index.dt','rb') as f:
        res = pickle.load(f)
        f.close()
    template = res['valid_index']   # The valid index, inclduing the target predicate 
    
    nID.neural_logic_program(label = t_relation, n_rules = n_rules, alpha = alpha, predicate_path=data_folder, template=template,target_index=target_index, only_logic_program=True, target_arity = target_arity, prior_knowledge = prior_knowledge, variable_number = variable_number)
    print("::> Define logic programs successfully. All output layers are",nID.OUTPUT)
    print()
    
    # Define model 
    print("::> Define the model")
    nID.build_model(figure_path=folder_name)
    print("::> Build model scuuess!")
    print()
    
    # Train the KB
    print("::> Begin to tain the defined multilabeled model respresenting neural predicates")

    if larger == True:
        nID.train(train_data,None,max_epochs = max_epochs, learning_rate = learning_rate, batch_size = None, model_path=folder_name + '/model', figure_path=folder_name, verbose=verbose, patience=2, validation_split=0.0, loss_weight = [0.2,0.2,0.2,1,0.2,0.2,0.2,1], relation_number = len(relations), prior_knowledge = prior_knowledge, validation_data = valid_data)
    else:
        nID.train(x,label,max_epochs = max_epochs, learning_rate = learning_rate, batch_size = batch_size,model_path=folder_name + '/model', figure_path=folder_name, verbose=verbose, patience=2, validation_split=0.2, loss_weight = [0.2,0.2,0.2,1,0.2,0.2,0.2,1], relation_number = len(relations), prior_knowledge = prior_knowledge)
    
    return 0

if __name__ == "__main__":
    
    task_name = sys.argv[1] 
    t_relation = sys.argv[2]
        
    t_main(task_name, t_relation)