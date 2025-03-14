# this code defines multilabel neural predicates
# -*- coding: utf-8 -*-
'''
set variable depth as k, and set all variables V in the datasets
V = 2 + k (2 indicates the number of two prime variables x and y)


set the input of neural networks, including two embeddings represetning two objects
set A(|V|, 2) + |V| as the number of output nodes of neural predicates

We regard multilabel neural networks as the neural predicate
'''
import os
import pickle
from re import A
import sys
import matplotlib.pyplot as plt 
from pathlib import Path

from numpy.core.records import array


#path = os.getcwczd()

path = Path(os.getcwd())
parent_path = path.parent.absolute()
print('Parent path is:')
print(parent_path)
sys.path.append(os.getcwd())
sys.path.append(str(parent_path))
print(sys.path)
print("----")
#sys.path.append(os.getcwd())
import logging
logger = logging.getLogger()
logger.basicConfig = logging.basicConfig(level=logging.DEBUG)
import tensorflow as tf
import tensorflow.keras as keras 
import tensorflow.keras.backend as K
import numpy as np
import DLFIT as DLFIT
import argparse
import itertools


# There is only one neural predicate in the current implementation

# set the number of layers in each neural predicate

LAYERS = 4 #default value 
CONSTANTS = {} # I think this is useless untill we do some constrains on the embeddings of the objects in the task 
PREDICATE = {} # The PREDICATE dictionary record all output from neural predicates 
OUTPUT = {} # The OUTOUT dictionary record all output layer from the both neural predicates and neural logic programs

# set trainable embeddings for variables
# def constant(label,value=None, min_value=None, max_value=None):
#     label = "ltn_constant_"+label
#     if value is not None:
#         result = tf.constant(value,name=label) 
#     else:
#         result = tf.Variable(tf.random.uniform(
#                 shape=(1,len(min_value)),
#                 minval=min_value,
#                 maxval=max_value,name=label))
#     result.doms = []
#     CONSTANTS[label] = result
#     return result


# def variable(label,number_of_features_or_feed):
#     if type(number_of_features_or_feed) is int:
#         result = tf.placeholder(dtype=tf.float32,shape=(None,number_of_features_or_feed),name=label)
#     elif isinstance(number_of_features_or_feed,tf.Tensor):
#         result = tf.identity(number_of_features_or_feed,name=label)
#     else:
#         result = tf.constant(number_of_features_or_feed,name=label)
#     result.doms = [label]
#     return result

def input_model(number_all_predicate):
    global input_layer 
    global input_2 
    global model_input
    input_layer = tf.keras.layers.Input(shape = (number_all_predicate,))
    input_2 = keras.layers.Input(shape = (1,))
    model_input = []
    model_input.append(input_layer)
    model_input.append(input_2)
    print("🌠 The input for pure neural logic programs build successfully!")
    return 0 
    
    
# set embedding layers for the objects
def define_embedding( single_input_length, embedding_size, number_arity = 2, variable_depth = 1):
    global flatten_layer 
    global input_layer
    global number_of_variable
    global input_2
    global model_input
    number_of_variable = number_arity + variable_depth

    # Define the input of the model
    input_layer = tf.keras.layers.Input(shape= (number_of_variable,), name='Input_Layer_pred')
    input_2 = keras.layers.Input(shape=(1,))
    # Assemble all input layer 
    model_input = []
    model_input.append(input_layer)
    model_input.append(input_2)
    embedding_layer = tf.keras.layers.Embedding(input_dim = single_input_length, output_dim = embedding_size, input_length=number_of_variable)(input_layer)
    # Connect the constants and their embeddings at here...
    
    flatten_layer = tf.keras.layers.Flatten()(embedding_layer)
    print("[⭐️ nID⭐️] Construct embedding layer successfully")
    return 0

#set binary neural predicates 
def binary_predicate(label='', t_relation=False, n_layer=LAYERS, pred_definition=None,):
    global flatten_layer
    number_of_predicate = len(list(itertools.permutations(range(number_of_variable),2))) # no unary predicates
    if t_relation == True:
        number_of_predicate = number_of_predicate - 1
    
    print('[⭐️ nID⭐️] The number of predicates for [',label,'] is',number_of_predicate) 
    tem_hidden_layer = tf.keras.layers.Dense(20, activation = 'relu', name = label+'_HL_'+str(0))(flatten_layer)
    if pred_definition is None:
        # define layer and weight parameters
        for i in range(n_layer-1):
            tem_hidden_layer = tf.keras.layers.Dense(20, activation = 'relu', name = label+'_HL_'+str(i+1))(tem_hidden_layer)
        output_layer = tf.keras.layers.Dense(number_of_predicate, activation = 'sigmoid', name = label+'_OL')(tem_hidden_layer)
    else: 
        print("[⭐️ nID⭐️] The predicate has been defined, and it cannot be defined again!")
        return 
    
    PREDICATE[label] = output_layer 
    OUTPUT[label] = output_layer 
    print('[⭐️ nID⭐️] Construct neural predicate for [',label,'] successfully') 
    return 0

# Define the neural logic programs with same head fitsr-order features. Like the RNN model, bulid a recurrent between input and output
def neural_logic_program(label='', n_rules=4, alpha = 10, predicate_path='', 
                         template = None, target_index=None, 
                         only_logic_program = False, target_arity = 2, 
                         prior_knowledge = False, variable_number = 3):
    global logic_program
    p_all_r = []
    if only_logic_program == False: 
        # Combine the values from all predicates into one vector in shape [|number of all predicates|,]
        for i in PREDICATE:
            p_all_r.append(PREDICATE[i])
        p_all_r = tf.concat(p_all_r, axis=1)
    else:
        p_all_r = input_layer
            

    logic_program = DLFIT.inference(n_rules= n_rules, alpha= alpha,data_path = predicate_path ,path=predicate_path+'/predicate.list', 
                                    template=template, target_index = target_index, 
                                    arity=target_arity, l_name= 'Original_layer',
                                    prior_knowledge = prior_knowledge,
                                    variable_number = variable_number) # define a logic_program obejct
    
    # Compute the infernece layer 
    logic_program_output = logic_program(p_all_r, fun_index = 1)
    inference_layer = logic_program_output['inference_output']
    final_predicate = logic_program.fuzzy_or(inference_layer) # inference layer 
    OUTPUT[label] = final_predicate
    
    if target_index != None and only_logic_program == False:
        predicted_output = logic_program.target_predicate_predited_neural(p_all_r)
        OUTPUT[label+'directly_pred'] = predicted_output
    # Compute the one loss layer
    #logic_program_output_2 = logic_program.one_sum(input_2)
    #one_loss_sum = logic_program_output_2
    one_sum_output = logic_program(input_2, fun_index = 2)
    OUTPUT[label+'one_loss'] = one_sum_output['one_sum_loss'] # ❓ A output layer for custimized loss , a float number 
    
    #sptial_output = logic_program.spatial_unity() # Add sptial constrains
    #OUTPUT[label+'sptial'] = sptial_output 
    
    sptial_output, sptial_output_2 = logic_program.spatial_unity(input_2)
    OUTPUT[label+'basic'] = sptial_output
    OUTPUT[label + 'occurance'] = sptial_output_2
    
    if prior_knowledge != False:
        OUTPUT[label+'prior_cos'] = logic_program(input_2, fun_index = 3)['cos_prior']
    
    
    # TODO Tis code is added in the recernt version 
    auxiliary_layer = DLFIT.auxiliary_inference( n_rules = n_rules,alpha = alpha,
                                                data_path = predicate_path, path=predicate_path+'/predicate.list',
                                                template=template, target_index = target_index,arity=target_arity, 
                                                l_name= 'AUX_layer', addtional_aux_num=2,
                                                aux_prior_knowledge = prior_knowledge, variable_number= variable_number)
    aux_res = auxiliary_layer(inputs = p_all_r,input_all_one= input_2)
    OUTPUT['aux_inf'] = aux_res[0]
    OUTPUT['aux_one_sum'] = aux_res[1]
    OUTPUT['aux_basic_loss'] = aux_res[2]
    OUTPUT['aux_occurance_loss'] = aux_res[3]
    OUTPUT['aux_dissimiliarity_loss'] = aux_res[4]
    return 0 

def build_model(figure_path = None):
    global model_output
    global model 
    # Define the model 


    print("[⭐️ nID ⭐️] All output layer is")
    for i in OUTPUT:
        print(OUTPUT[i])

    # try the ombination output
    #new_output = tf.concat(list(OUTPUT.values()), axis= 1)
    
    
    model_output = OUTPUT.values() # A list includes all the output layers which we have defined before 
    model = tf.keras.Model(inputs = model_input,  outputs = model_output)
        
    # Print the model summary
    print("[⭐️ nID ⭐️] Model summary is:")
    print(model.summary())
    
    # print the details of the model
    # print('[⭐️ nID⭐️] Print model...')
    # tf.keras.utils.plot_model(model, to_file=figure_path+'/model_plot.png', show_shapes=True, show_layer_names=True)
    # print('[⭐️ nID⭐️] Print the details of model success!')
    

    return 0
    


# set trainable function to train the neural predcates in the PREDICATE dictionary
def train(x, y, max_epochs=10000, learning_rate= 0.02, decay_rate= 1, model_path = None ,verbose = 1, patience = 100, figure_path = None, batch_size = 32, validation_split = 0.2 ,loss_weight=None, relation_number = None, prior_knowledge = False, validation_data = None):
    
    # print('All constants include',CONSTANTS) # we do not need constant layer when we already set the Embedding layer.    
    # Define the custom loss function 
    # DLFIT_one_sum_loss = DLFIT.inference.loss_sum_one # ? Fix at here
    

    # Define the metrics and training algorithms
    print("[⭐️ nID⭐️] Define the loss and metrics...")
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate,decay_steps=10000,decay_rate=decay_rate) # it is not decay now
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule) # Using Adam as the optimization function. We can try SGD later.
    metrics_ce = []
    #metrics_ce.append(tf.keras.metrics.Precision())
    #metrics_ce.append(tf.keras.metrics.Recall())
    metrics_ce.append(tf.keras.metrics.BinaryAccuracy())
    # Define the loss list 
    loss_list = []
    #loss_list.append("mean_squared_error")
    loss_list.append('binary_crossentropy')  # Let predicate of predicate and target predicad are trained thourgh CrossEntropy  
    loss_list.append('mean_squared_error') # Let one_sum and sptical loss are trained through MSE 
    loss_list.append('mean_squared_error') 
    loss_list.append('mean_squared_error') 
    if prior_knowledge != False:
        loss_list.append('mean_squared_error')
    # TODO For Auxiliary Predicate 
    loss_list.append('binary_crossentropy')  # Let predicate of predicate and target predicad are trained thourgh CrossEntropy  
    loss_list.append('mean_squared_error') # Let one_sum and sptical loss are trained through MSE 
    loss_list.append('mean_squared_error') 
    loss_list.append('mean_squared_error') 
    loss_list.append('mean_squared_error') 

    model.compile(loss=loss_list, optimizer=opt,  metrics=metrics_ce,loss_weights=loss_weight) # ⭐️ check whether is right. I guess the threshhold in the metrics computing function is 0.5
    

    
    # Define the callback functions for 
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
    
    # cir_learning = CurriculumLearning(logic_program, validation_data, relation_number)
    
    # combine y 
    # y = np.concatenate(y, axis=1)
    print("[⭐️ nID⭐️] Fit model on training data:")
    
    history = model.fit(
        x,
        y,
        batch_size=batch_size,
        epochs=max_epochs,
        validation_split = validation_split,
        verbose=verbose,
        callbacks=[es],
        validation_data=validation_data,
    )
    
    attributes = list(history.history.keys())
    print('[⭐️ nID⭐️] The attributes in the model include',attributes)
    
    # Save the model
    if model_path != None:
        model.save(model_path)
        print("[⭐️ nID⭐️] Save model successfully")
    
    # Polt the metrics in the model
    if figure_path != None:
        plot_metrics(attributes, history, figure_path)
        
    return 0
    
    
def plot_metrics(attributes, history, figure_path):
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    
    plt.title('loss of all NLP')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss','val_loss'], loc='right')
    plt.savefig(figure_path+'/loss.png')
    plt.clf()   
    print("[⭐️ nID⭐️] Plot metrices successfully! ")
    return 0

def test_model(x, model_path = None):
    if model_path != None:
        model = keras.models.load_model(model_path)
    print(len(list(model.layers)))
    for i in range(1, len(model.layers)):
        layer_output = tf.keras.backend.function([model.layers[0].input], [model.layers[i].output])
        layer_output = layer_output(x)
        print("[⭐️ nID⭐️] Test output from the",i,'th is',layer_output)
    return 0 
    
def prediction(model, true_value_dictionary):
    # with open(data_path+'/valid_index.dt','rb') as f:
    #     valid_index = pickle.load(f)
    #     f.close()
    # v_index = valid_index['valid_index']
    # template = valid_index['template']
    output = model.predict(true_value_dictionary)
    print("[⭐️ nID⭐️] The prediction is:")
    print(output)
    return output

def extract_symbolic_logic_programs(relation = [], data_path = '', 
                                    result_path = '', model = None, threshold = 0.0,
                                    head_pre = '', number_of_variable = 3):
    with open(data_path+'/valid_index.dt','rb') as f:
        valid_index = pickle.load(f)
        f.close()
    with open(data_path+'/relation_variable.dt','rb') as f:
        relation_variable_permutations = pickle.load(f)
        f.close()
    v_index = valid_index['valid_index']
    template = valid_index['template']
    reconstructed_model = model
    
    print("[⭐️ nID⭐️ ] The number of all layer is:") 
    print(len(reconstructed_model.layers))
    
    layer = reconstructed_model.get_layer(name = 'Original_layer')
    all_layer_in_original = layer.get_weights()
    layer_weights_o = all_layer_in_original[0] # The first element in this list is the trainable 
    
    layer_aux_ini = reconstructed_model.get_layer(name = 'AUX_layer')
    all_layer_aux = layer_aux_ini.get_weights()
    layer_aux = all_layer_aux[0]
    layer_aux_mean_sum = np.mean(layer_aux,axis=2)
    layer_aux_final = np.transpose(layer_aux_mean_sum)
    
    
    layer_weights = np.concatenate((layer_weights_o, layer_aux_final) , axis=1)
    
    print(layer_weights)
    
    # set the variable in the dic to X Y Z and additional variable to W1 W2 W3...
    dic = {0:'X', 1:'Y', 2:'Z'}
    basic_number = 3
    additional_variable_name_pool = ['W','M','N','T']
    for i in range(3,number_of_variable):
        dic[i] = additional_variable_name_pool[i-basic_number]
    
    # # save the variable dic
    # with open(data_path+'/variableinfo.txt', 'w') as f:
    #     print(dic.keys(), file = f)
    #     f.close()
    # with open(data_path+'/variableinfo.dt', 'wb') as f:
    #     pickle.dump(dic.keys(), file = f)
    #     f.close()

    target_predicate_tuple=[]
    # According to the index, reshow the symbolic information stored in the following list
    all_variable_symbolic = []
    for item in relation_variable_permutations:
        head_symbol = item
        body_list = relation_variable_permutations[item]
        for pair_variable in body_list:
            first_variable = dic[pair_variable[0]]
            second_variable = dic[pair_variable[1]]
            latter = '('+first_variable+','+second_variable+')'
            target_predicate_tuple.append((head_symbol,latter))
    
    
    
    target_predicate = []
    for item in range(len(target_predicate_tuple)):
        if item in v_index:
            target_predicate.append(target_predicate_tuple[item][0]+target_predicate_tuple[item][1])
        

    #extract the symbolic logic format 
    max_value_tensor = np.amax(layer_weights)
    if max_value_tensor < threshold:
        print("No acceptable elements in the training dataset according to the threshold.")
        return -1, layer_weights
    statisfied_index = np.transpose((layer_weights >= threshold).nonzero())
    # print(statisfied_index)
    satisfied_list = []
    for i in statisfied_index:
        one = []
        one.append(i[1])
        one.append(i[0])
        satisfied_list.append(one)
    
    satisfied_list = sorted(satisfied_list, key=lambda tup: tup[0])    
    # print(satisfied_list)
    
    try:
        number_of_rule = satisfied_list[-1][0] + 1 
    except IndexError:
        return -2,-2 
    information_logic = {}
    
    for i in range(number_of_rule):
        information_logic[i] = []
    
    for i in satisfied_list:
        f_i = i[0]
        s_i = i[1]
        information_logic[f_i].append(s_i)
    print("[⭐️ nID ⭐️] predicate index after the logic index")
    print(information_logic)
    rules = []
    for i in information_logic:
        rule_one = head_pre + ' :- '
        for item in information_logic[i]:
            rule_one += target_predicate[item] + '& '
        rules.append(rule_one)
    # print(rules)
    
    return rules, layer_weights


class CurriculumLearning(keras.callbacks.Callback):
    """Stop training when the precision and recall of the logic rules are both one. 
    The sumbolic methods (algorithm)
    Arguments:
    The weight
    """

    def __init__(self,logic_program_c,validation_data, relation_number):
        super(CurriculumLearning, self).__init__()
        self.logic_program = logic_program_c
        self.validation_data = validation_data
        self.template_predicate_index = None
        self.relation_number = relation_number

    def on_train_begin(self, logs=None):
        self.presicion_rule = 0
        self.recall_rule = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        self.template_predicate_index = self.logic_program.return_template_predi_index()
        print("The template indexs are:")
        print(self.template_predicate_index)


    def on_epoch_begin(self, epoch, logs=None): # change to the end
        # Show all validation 
        print("Validation data")
        print(self.validation_data)
        val_data = self.validation_data[1]
        empty_one = []
        for i in range(self.relation_number):
            empty_one.append(val_data[i])
        val_data = tf.concat(empty_one, axis = 1)
        val_label = self.validation_data[1][self.relation_number]
        print(val_data, val_label)
        # get the label of the validation data in the corresponding indexes
        if self.template_predicate_index != None:
            valid_predicate_data = tf.gather(val_data, self.template_predicate_index, axis=1)
            print("Valid_data")
            print(valid_predicate_data)
        else:
            print("Set logic template:")
            
        # Extract the logic program using the defined method
        layer = model.get_layer(index = -6)
        layer_weights = layer.get_weights()  
        print(layer_weights)
        
        
        
        # Make predictin in the defined method in the DLFIT class or redefined ?
        
        # Compute the predision and recall of the generated symboloc program 
                
        # Begin to extract the logic program from the learned matrices 
        # rules= extract_symbolic_logic_programs(relation=['locatedIn','neigibor'],head_pre='locatedIn(x,y)',
        #                                         dataset = data_path, model = '/result/model',
        #                                         number_of_variable=3, threshold = 0.)
        # print("::> All rules extracted from neural logic prgoram are:")
        # for i in rules:
        #     print(i)
        
        
        # current = logs.get("loss")
        # if np.less(current, self.best):
        #     self.best = current
        #     self.wait = 0
        #     # Record the best weights if current results is better (less).
        #     self.best_weights = self.model.get_weights()
        # else:
        #     self.wait += 1
        #     if self.wait >= self.patience:
        #         self.stopped_epoch = epoch
        #         self.model.stop_training = True
        #         print("Restoring model weights from the end of the best epoch.")
        #         self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
