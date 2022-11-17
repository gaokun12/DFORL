# -*- coding: utf-8 -*-
import os
import re
import sys
from pathlib import Path
from time import time
from typing_extensions import final
path = Path(os.getcwd())
parent_path = path.parent.absolute()
sys.path.append(os.getcwd())
sys.path.append(str(parent_path))
import logging
logger = logging.getLogger()
logger.basicConfig = logging.basicConfig(level=logging.DEBUG)
from DFOL.model.data_generator import gen
from DFOL.model.train import t_main
from DFOL.model.predict_extract import  get_best_logic_programs, check_pl, return_all_predicate, accuracy_all_relation
import argparse
import time
import pickle
from pyDatalog import pyDatalog 
from DFOL.model.tools.pre_data_gen import data_gen
from DFOL.model.tools.lessthan_data_gen_pb import data_gen_lessthan
from DFOL.model.tools.lessthan_data_gen_mis import data_gen_lessthan_mis
'''
The code in this folder are the original version. (without any attempts to solve the long logic rules)
'''

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--g',
                    help="generate dataset", type=int, default=0)
# parser.add_argument('-t', '--t',
                    # help="Train model?", type=int, default = 0)
parser.add_argument('-d', '--d',
                    help="dataset name", type=str, default = '')
parser.add_argument('-ap', '--ap',
                    help="generate for all predicate", type=int, default=0)
parser.add_argument('-p', '--p',
                    help="predicate name", type=str, default='')
# parser.add_argument('-i', '--i',
                    # help="check indicators", type=int, default = 0)
# parser.add_argument('-c', '--c',
                    # help="check best logic program", type=int, default = 0)
# parser.add_argument('-mp', '--mp',
                    # help="Make prdiction", type=int, default = 0)
parser.add_argument('-cur', '--cur',
                    help="curriculum learning", type=int, default = 0)
parser.add_argument('-amg', '--amg',
                    help="amg training", type=int, default = 0)
parser.add_argument('-mis', '--mis',
                    help="mislabelling training", type=int, default = 0)
parser.add_argument('-checkpl', '--checkpl',
                    help="onlycheckfromlogic", type=int, default = 0)
parser.add_argument('-vd', '--vd',
                    help="variable_depth", type=int, default = 1)
parser.add_argument('-ft', '--ft',
                    help="final threshold", type=float, default = 0.3)
parser.add_argument('-alpha', '--alpha',
                    help="alpha", type=int, default = 10)
parser.add_argument('-tfn', '--testfilename',
                    help="test program file name", type=str, default = 'best')
parser.add_argument('-cap', '--check_acc_p',
                    help="check accuracy for all predicates", type=int, default = 0)


args = parser.parse_args()

# predefine all arguemsnts
g = bool(args.g)
# t = bool(args.t)
# i = bool(args.i)
# c = bool(args.c)
ap = bool(args.ap)
d = args.d
p = args.p
# mp = bool(args.mp)
cur = bool(args.cur)
amg = bool(args.amg)
mis = bool(args.mis)
checkpl = bool(args.checkpl)
variable_depth = args.vd
final_threshold = args.ft
alpha = args.alpha
test_file = args.testfilename
cap = bool(args.check_acc_p)

print('ARGS:', args)

# def all_predicate_run(dataset_name):
#     all_pred = return_all_predicate(dataset_name,dataset_name)
#     return all_pred


def data_assemble(dataset, predicate, variable_depth):
    target_arity, head_pre = gen(dataset,predicate, variable_depth)
    meta_info = {'target_arity':target_arity, "head_pre": head_pre}
    with open("DFOL/"+dataset+'/data/'+predicate +"/meta_info.dt",'wb') as f:
        pickle.dump(meta_info,f)
        f.close()
    with open("DFOL/"+dataset+'/data/'+predicate +"/meta_info.txt",'w') as f:
        print(meta_info, file=f)
        f.close()
    return 0 

        
if g == True:
    data_assemble(d,p,variable_depth)
    
# if t == True:
#     with open("DFOL/"+d+'/data/'+p +"/meta_info.dt",'rb') as f:
#         meta_info = pickle.load(f)
#         f.close()
#     target_arity = meta_info["target_arity"]
#     t_main(d,p, alpha = alpha, learning_rate= 0.001, target_arity = target_arity, max_epochs=2000, n_rules=5, variable_depth=variable_depth) # 2000
# if c == True: 
#     with open("DFOL/"+d+'/data/'+p +"/meta_info.dt",'rb') as f:
#         meta_info = pickle.load(f)
#         f.close()
#     head_pre = meta_info["head_pre"]
#     target_arity = meta_info["target_arity"]
#     get_best_logic_programs(d,p,head_pre,target_arity, variable_depth)
# if i == True:
#     # with open("DFOL/"+d+'/data/'+p +"/meta_info.dt",'rb') as f:
#     #     meta_info = pickle.load(f)
#     #     f.close()
#     # target_arity = meta_info["target_arity"]
#     check_Hits(d,p, test_file)
# if mp == True:
#     prediction(d,p)

def curriculum_learning(dataset, predicate):
    ini_alpha = alpha
    learning_times = 5
    time_train = 0
    while time_train < learning_times:
        with open("DFOL/"+dataset+'/data/'+predicate +"/meta_info.dt",'rb') as f:
            meta_info = pickle.load(f)
            f.close()
        target_arity = meta_info["target_arity"]
        # Train the mdoel 
        t_main(dataset,predicate, alpha = ini_alpha, learning_rate= 0.001, target_arity = target_arity, max_epochs=2000, n_rules=5, prior_knowledge= True, variable_depth=variable_depth) # 2000
        # Do extract and save the best parameters 
        head_pre = meta_info["head_pre"]
        target_arity = meta_info["target_arity"]
        acc = get_best_logic_programs(dataset,predicate,head_pre,target_arity,variable_depth, final_threshold=final_threshold)
        if acc == -2:
            ini_alpha = int(ini_alpha/1.2)
            continue
        if acc >= 0.997:
            break
        time_train+=1
    return acc

if cur == True:
    curriculum_learning(d,p)
    
if ap == True:
    all_pred = return_all_predicate(d)
    
    for current_predicate in all_pred:
        data_path = 'DFOL/'+d+'/data/'+current_predicate
        result_path = 'DFOL/'+d+'/result/'+current_predicate
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        # generate trainig data
        training_data_path = data_path + '/meta_info.dt'
        if not os.path.exists(training_data_path):
            data_assemble(d,current_predicate,variable_depth)
        logic_program_path = result_path + '/acc_pred.txt'
        if not os.path.exists(logic_program_path):
            curriculum_learning(d, current_predicate)
        
def ambiguous():
    # only support the succ and lessthan dataset because of the data geneator python file 
    stand_derivation = [3,2.5,2,1.5,1,0.5]
    res = []

    for single_derivation in stand_derivation:
        with open("DFOL/"+d+'/result/'+p+"/ambi.txt",'a') as f:
            data_gen_lessthan(single_derivation,d)
            data_assemble(d,p,variable_depth)
            # remove the prior knoeledge 
            if os.path.exists("DFOL/"+d+'/data/'+p+"/prior_knowledge.dt"):
                os.remove("DFOL/"+d+'/data/'+p+"/prior_knowledge.dt")
            else:
                print("The file does not exist")
            if os.path.exists("DFOL/"+d+'/result/'+p+"/best.pl"):
                os.remove("DFOL/"+d+'/result/'+p+"/best.pl")
            else:
                print("The file does not exist")
            acc = curriculum_learning(d,p)
            res.append((single_derivation, acc))
            print(res, file=f)
            print(res)
            f.close()
    return 0
if amg == True:
    ambiguous()

def mislabelling():
    mis_rates = [0.95,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
    res = []
    for mis_rate in mis_rates:
        with open("DFOL/"+d+'/result/'+p+"/mis.txt",'a') as f:
            data_gen_lessthan_mis(mis_rate, d)
            data_assemble(d,p,variable_depth)
            # remove the prior knoeledge 
            if os.path.exists("DFOL/"+d+'/data/'+p+"/prior_knowledge.dt"):
                os.remove("DFOL/"+d+'/data/'+p+"/prior_knowledge.dt")
            else:
                print("The file does not exist")
            if os.path.exists("DFOL/"+d+'/result/'+p+"/best.pl"):
                os.remove("DFOL/"+d+'/result/'+p+"/best.pl")
            else:
                print("The file does not exist")
            acc = curriculum_learning(d,p)
            res.append((mis_rate, acc))
            print(res, file=f)
            print(res)
            f.close()
    return 0 

if mis == True:
    mislabelling()

if checkpl == True:
    with open("DFOL/"+d+'/data/'+p +"/meta_info.dt",'rb') as f:
        meta_info = pickle.load(f)
        f.close()
    target_arity = meta_info["target_arity"]
    check_pl(d,p,target_arity,file_name=test_file+'.pl')    

if cap == True:
    accuracy_all_relation(d, test_file)

# nations alpha = 3 learning_rate = 0.001 / 0.01
#umls alpha = 10, learning_rate = 0.001
