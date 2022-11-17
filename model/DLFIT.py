import itertools
import tensorflow as tf 
import tensorflow.keras as keras
from tensorflow.python.keras.backend import shape
from tensorflow.python.ops.gen_batch_ops import batch

import pickle
import numpy as np

#TODO: 1. Delete all codes relatived with the original version which not includs target_index 
# Define the layer of D-LFIT:
# The input of all layers is a vector embedding all predicates from all relations in tasks. 
class inference(keras.layers.Layer):     # unit is the default number of different logic programs for single relation predicate  
    '''
    - self.target_index is a integer to show th location of the target predicates
    - self.templtae show the tuple of the templates ! need to change to template_predi_index, the latter one
    is more accurate
    '''
    def __init__(self, n_rules, alpha ,data_path, path  , 
                 template , target_index, arity , l_name, 
                 prior_knowledge= False, variable_number = 3):
        self.rules = n_rules
        self.alpha = alpha
        self.path = path 
        # self.template = template
        self.template_predi_index = template
        self.template_predi = []
        self.target_index = target_index 
        self.arity = arity
        self.prior_knowledge = prior_knowledge 
        self.variable_number = variable_number
        if prior_knowledge != False:
            with open(data_path + '/prior_knowledge.dt', 'rb') as f:
                self.prior_knowledge = pickle.load(f)
                f.close()
            self.prior_knowledge_number = np.shape(self.prior_knowledge)[1]
            
        super(inference, self).__init__(name = l_name)
        
    # Define the customized sigmoid-like activation function
    def sigmoid_like(self, x):
        y = tf.math.divide(1,(1+tf.math.exp(-1 * self.alpha *(x))))  
        return y
    
    def build(self, input_shape): 
        # Make embeddings for all predicates in the output of the neural predicate layers
        with open(self.path,'rb') as vf:
            all_predicates = pickle.load(vf)
            vf.close()
        print("[⭐️ np] All predicates in the task are:", all_predicates)

        # if self.template_predi_index != None:
        # TODO Load the valid index of predicates
        for i in self.template_predi_index:
            self.template_predi.append(all_predicates[i])
        print("⭐️ The indexes are:")
        print(self.template_predi_index)
        print("⭐️ Selected predicate according to the template are")
        print(self.template_predi)
        
        if type(self.prior_knowledge) is not bool:
            # This operation just increaset the loss value from the begining stage 
            # ! we plan to set this weight value very large 
            self.prior_weights = tf.Variable(initial_value=self.prior_knowledge, trainable=False, name = 'prior') 
            self.w = tf.Variable(
                initial_value=tf.random.truncated_normal(shape=(len(self.template_predi_index), self.rules), dtype="float32"),
                trainable=True,
                constraint=lambda t: tf.clip_by_value(t, 0, 1), name = 'rules_weights')
        else:
            self.w = tf.Variable(
                initial_value=tf.random.truncated_normal(shape=(len(self.template_predi_index), self.rules), dtype="float32"),
                trainable=True,
                constraint=lambda t: tf.clip_by_value(t, 0, 1), name='rules_weights'
            )
        # else: 
        #     self.w = tf.Variable(
        #         initial_value=tf.random.truncated_normal(shape=(input_shape[-1]-1, self.rules), dtype="float32"), # The node expecting the target predicates
        #         trainable=True,
        #         constraint=lambda t: tf.clip_by_value(t, 0, 1)
        #     )
        # TODO Perpare for the embeedding for the template predicates 
        embedddings_o_p = []
        if self.template_predi_index == None:
            valid_pred = all_predicates
        else:
            valid_pred = self.template_predi
        
        for i in range(len(valid_pred)):
            item = valid_pred[i]
            one_label = []
            # make basic and occurrence embeddings for each valid first-order features
            for variable_index in range(self.variable_number):
                if item[0] == variable_index or item[1] == variable_index: # Check whether variable x is in the tuple item (0,2) <-> (x,z)
                    one_label.append(1)
                else:
                    one_label.append(0)
            # if item[0] == 0 or item[1] == 0: # Check whether variable x is in the tuple item (0,2) <-> (x,z)
            #     one_label.append(1)
            # else:
            #     one_label.append(0)
            # if item[0] == 1 or item[1] == 1: # Check whether variable y is in the tuplte item 
            #     one_label.append(1)
            # else:
            #     one_label.append(0)
            # if item[0] == 2 or item[1] == 2: # Check whether variable z is in the tuplte item 
            #     one_label.append(1)
            # else:
            #     one_label.append(0)
            embedddings_o_p.append(one_label)
        print("[⭐️ np] All emnbeddings for output predicate are:")   
        print(embedddings_o_p)
        self.e_o_p = tf.concat(np.array(embedddings_o_p, dtype=np.float32), 0)
        self.e_o_p = tf.Variable(initial_value=self.e_o_p, trainable=False)
        print("[⭐️ np] The shape of the embeddings of all predicates after concation:")
        print(self.e_o_p)
        self.e_o_p = tf.reshape(self.e_o_p, [self.variable_number,-1])
        print("[⭐️ np] The shape of the embeddings of all predicates after transpose:")
        print(self.e_o_p)
        super().build(input_shape)
        return 0 
        
    
    def sptial_activation(self, x):
        a = 1
        b = 10
        c = 1
        y =  c * tf.math.exp(a-b*tf.math.square(x-1))
        return y
            
    def fuzzy_or(self, inputs):
        neg_inputs = 1 - inputs
        pro_neg_input = tf.math.reduce_prod(neg_inputs, 1)
        predict_value = 1 - pro_neg_input
        predict_value = tf.reshape(predict_value, [-1, 1])
        return predict_value
        
    
    def spatial_unity(self, inputs):
        '''
        Define the sptial unity:
        - 1. Ensure that the variable x and y appear in the body
        - 2. Ensure the occurrence of variable z in the body cannnot be one 
        '''
        # Do multiple between parameters and its embeddings. The first arity         
        weights = self.w 
        weights = tf.transpose(weights)
        weights = tf.expand_dims(weights,1)
        weighted_embeddings = weights * self.e_o_p
        constrain_x_and_y = weighted_embeddings[:,0:self.arity,:] # keep the first arity embeddings 
        constrain_rest = weighted_embeddings[:,self.arity:,:] # The latter embeddigns 
        neg_weighted_embedddings = 1-constrain_x_and_y
        reduce_pro = tf.math.reduce_prod(neg_weighted_embedddings, 2)
        fuzzy_or =  1 - reduce_pro # This result is the fuzzy or between all predicate for same boolean variable and
        print(fuzzy_or)
        fuzzy_and = tf.math.reduce_prod(fuzzy_or, 1) # This result is the fuzzy and between all Boolean variable in the all rules 
        print(fuzzy_and)
        fuzzy_and = tf.reshape(fuzzy_and, [-1, self.rules])
        # all_rule_fuzzy_and = tf.math.reduce_prod(fuzzy_and, 0)
        # all_rule_fuzzy_and = tf.reshape(all_rule_fuzzy_and, [-1,1])
        print("[⭐️ np]: The final of sptial output")
        print(fuzzy_and)
        sptial_output = tf.matmul(inputs, fuzzy_and, name='sptial_utility') 

        # The second arity 
        the_sum_all_rest = tf.math.reduce_sum(constrain_rest,2)
        sptial_activate = self.sptial_activation(the_sum_all_rest)
        sum_sptial_activate = tf.math.reduce_sum(sptial_activate, 1)
        sptial_activate_transpose = tf.reshape(sum_sptial_activate, [-1, self.rules])
        sptial_output_2 = tf.matmul(inputs, sptial_activate_transpose, name='sptial_utility_rest_variable')
        
        return sptial_output, sptial_output_2
        
    def target_predicate_predited_neural(self,inputs): # predicted by the neural predicates
        print(inputs)
        target_predicate_predicted = tf.gather(inputs, self.target_index, axis = 1, name='direct_predi')
        target_predicate_predicted=tf.reshape(target_predicate_predicted, [-1,1],name='direct_predi')
        return target_predicate_predicted
    
    def call(self, inputs,fun_index):  #! Upate at here when logic templates are used.
        logic_program_layer = {}
        if fun_index == 1:
            # if self.template_predi_index == None:
            #     if self.target_index == None:
            #         t_head = inputs[:,:self.target_index]
            #         t_tail = inputs[:,self.target_index+1:]
            #         inputs = tf.concat([t_head, t_tail], axis = 1)
            #         logic_program_layer['inference_output'] = self.sigmoid_like(tf.matmul(inputs,self.w,name='inference_with_tem')-1)
            #     else:
            #         logic_program_layer['inference_output'] = self.sigmoid_like( tf.matmul(inputs, self.w,name='infer') - 1 ) # ! change at here
            # else:
            template_predicate = tf.gather(inputs, self.template_predi_index, axis=1)
            print("⭐️ The original output of neural predicate are:")
            print(inputs)
            print("⭐️ The cutten output after logic tem[late are")
            print(template_predicate)
            logic_program_layer["inference_output"] = self.sigmoid_like( tf.matmul(template_predicate, self.w) -1)
        if fun_index == 2:
            '''
            Make the sum of all weights in a single rule be one. 
            '''
            sum_tem = tf.reduce_sum(self.w, 0)
            sum_tem = tf.reshape(sum_tem, [1,-1])
            print(sum_tem.shape)
            logic_program_layer['one_sum_loss'] = tf.matmul(inputs, sum_tem, name='one_sum')    
            print("[⭐️ np]: Shape of one_sum",logic_program_layer['one_sum_loss'] )     

        if fun_index == 3:
            all_combination_rows = list(itertools.product(range(self.prior_knowledge_number), range(self.rules)))          
            cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
            dissimi = []
            for j in all_combination_rows:
                y_1 = tf.reshape(self.prior_weights[:,j[0]], [1,-1])
                y_2 = tf.reshape(self.w[:,j[1]], [1,-1])
                diss = tf.reshape(tf.convert_to_tensor(cosine_loss(y_1, y_2)), [1])
                dissimi.append(diss)
            dissmi_tf = tf.concat(dissimi,axis=0)
            # y_1 = all_combination_rows[:,0]
            # y_2 = all_combination_rows[:,1]
            # cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
            dissimilarity = tf.reshape( dissmi_tf,[1,-1])
            final_dis = tf.matmul(inputs, dissimilarity)
            logic_program_layer['cos_prior'] = final_dis # To MSE (cos_prior-0)^2
        return logic_program_layer
    
    def get_config(self):
        return {"inference_weights": self.w.numpy(), "sptial_embeddings":self.e_o_p.numpy()}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def return_template_predi_index(self):
        return self.template_predi_index
        
    
class auxiliary_inference(inference):
    def __init__(self, addtional_aux_num, aux_prior_knowledge, **kwds):
        self.addtional_aux_num = addtional_aux_num
        
        super().__init__(**kwds)
        
    
    def build(self,input_shape):
        super().build(input_shape=input_shape)
        # Make embeddings for all predicates in the output of the neural predicate layers
        with open(self.path,'rb') as vf:
            all_predicates = pickle.load(vf)
            vf.close()
        print("[⭐️ np] All predicates in the task are:", all_predicates)

        if self.template_predi_index != None:
            for i in self.template_predi_index:
                self.template_predi.append(all_predicates[i])
            print("⭐️ The indexes are:")
            print(self.template_predi_index)
            print("⭐️ Selected predicate according to the template are")
            print(self.template_predi)
            self.w = tf.Variable(
                initial_value=tf.random.truncated_normal(shape=(self.rules, len(self.template_predi_index), self.addtional_aux_num), dtype="float32"),
                trainable=True,
                constraint=lambda t: tf.clip_by_value(t, 0, 1)
            )
        else: 
            self.w = tf.Variable(
                initial_value=tf.random.truncated_normal(shape=(self.rules, len(self.template_predi_index), self.addtional_aux_num), dtype="float32"),
                trainable=True,
                constraint=lambda t: tf.clip_by_value(t, 0, 1)
            )
        return 0

    
    def spatial_unity(self, x):
        '''
        Define the sptial unity:
        - 1 BASIC CONSTRAIN. Ensure that the variable x and y appear in the body. 
            - Update: Ensure the variable in the head apper in the body 
        - 2 OCCURANCE CONSTRAIN. Ensure the occurrence of variable z in the body cannnot be one 
            - Update: Ensure the rest variable not  in the head connot be one. 
            - The index in the embedding of all predicate are fixed x:0 y:1 z:2
        '''
        # Do multiple between parameters and its embeddings. The first arity         
        weights = tf.reduce_mean(self.w, axis=2) 
        weights = tf.expand_dims(weights,1)
        weighted_embeddings = weights * self.e_o_p
        constrain_x_and_y = weighted_embeddings[:,0:self.arity,:] # keep the first arity embeddings 
        constrain_rest = weighted_embeddings[:,self.arity:,:] # The latter embeddigns 
        neg_weighted_embedddings = 1-constrain_x_and_y
        reduce_pro = tf.math.reduce_prod(neg_weighted_embedddings, 2)
        fuzzy_or =  1 - reduce_pro # This result is the fuzzy or between all predicate for same boolean variable and
        print(fuzzy_or)
        fuzzy_and = tf.math.reduce_prod(fuzzy_or, 1) # This result is the fuzzy and between all Boolean variable in the all rules 
        print(fuzzy_and)
        fuzzy_and = tf.reshape(fuzzy_and, [-1, self.rules])
        # all_rule_fuzzy_and = tf.math.reduce_prod(fuzzy_and, 0)
        # all_rule_fuzzy_and = tf.reshape(all_rule_fuzzy_and, [-1,1])
        print("[⭐️ np]: The final of sptial output")
        print(fuzzy_and)
        sptial_output = tf.matmul(x, fuzzy_and, name='sptial_utility') 

        # The second arity 
        the_sum_all_rest = tf.math.reduce_sum(constrain_rest,2)
        sptial_activate = self.sptial_activation(the_sum_all_rest)
        sum_sptial_activate = tf.math.reduce_sum(sptial_activate, 1)
        sptial_activate_transpose = tf.reshape(sum_sptial_activate, [-1, self.rules])
        sptial_output_2 = tf.matmul(x, sptial_activate_transpose, name='sptial_utility_rest_variable')
        
        return sptial_output, sptial_output_2
    
    def max_similarity(self, list_weights, one_input):
        '''
        In the keras API, the similarity = 1, then the vector is different 
        '''
        shape_weight = list_weights.get_shape().as_list()
        number_AB_rule = shape_weight[0]
        number_AH_rule = shape_weight[2]
        all_combination_rows = list(itertools.combinations(range(number_AH_rule),2))
        cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
        dissimi = []
        for i in range(number_AB_rule):
            for j in all_combination_rows:
                y_1 = tf.reshape(list_weights[i,:,j[0]],[1,-1])
                y_2 = tf.reshape(list_weights[i,:,j[1]],[1,-1])
                diss = tf.reshape(tf.convert_to_tensor(cosine_loss(y_1, y_2)), [1])
                dissimi.append(diss)
        dissmi_tf = tf.concat(dissimi,axis=0)
        # y_1 = all_combination_rows[:,0]
        # y_2 = all_combination_rows[:,1]
        # cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
        dissimilarity = tf.reshape( dissmi_tf,[1,-1])
        final_dis = tf.matmul(one_input, dissimilarity)
        return final_dis
    
    
    def DLFIT_inference(self, x):
        template_predicate = tf.gather(x, self.template_predi_index, axis=1, name='aux_inf')
        average_sum_aux = tf.reduce_mean(self.w, axis=2)
        average_sum_aux_reshape = tf.reshape(average_sum_aux, [-1,self.rules])
        fuzzy_or_value = self.sigmoid_like(tf.matmul(template_predicate, average_sum_aux_reshape)  -1)
        fuzzy_or_value = super(auxiliary_inference, self).fuzzy_or(fuzzy_or_value)
        return fuzzy_or_value

    
    def call(self, inputs,input_all_one):
        # The inference computation, the sptial computation and the one sum computation 
        logic_inference = self.DLFIT_inference(inputs)
        basic_constrains, occurance_constrains  = self.spatial_unity(input_all_one)
    
        # Make the sum of all weights in a single rule be one.
        weights = tf.reduce_mean(self.w, axis = 2)    
        sum = tf.reduce_sum(weights,axis=1)
        sum_tem = tf.reshape(sum, [-1,self.rules])
            
        #print(sum_tem.shape)
        one_sum_constrains = tf.matmul(input_all_one, sum_tem, name='one_sum')    
        #print("[⭐️ np]: Shape of one_sum",one_sum_constrains)
        
        # TODO Compute the dissimilarity between all rows, we will return the meaned sum of all value, we can call the keras loss API 
        # TODO The similiarity should be 1  (most dissimilar)
        similiarity = self.max_similarity(self.w, input_all_one) # should be 1 in shape[None, 5]
        
        
        # TODO ONE_SUM_CONSTRAIN and BASIC_CONSTRAIN should eauql with one, OCCURSNCE_CONSTRAIN shoule be zeros
        return [logic_inference, one_sum_constrains, basic_constrains, occurance_constrains, similiarity]
    
    
    def get_config(self):
        return {"inference_weights": self.w.numpy(), "sptial_embeddings":self.e_o_p.numpy()}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def return_template_predi_index(self):
        return self.template_predi_index


class make_prediction_using_logic_program(keras.layers.Layer):
    def __init__(self, logic_program_dic ):
        super(make_prediction_using_logic_program, self).__init__()
        self.logic_program_dic = logic_program_dic
        
    def call(self):
        return 0
