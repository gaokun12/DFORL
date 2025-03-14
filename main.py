'''
@ Description: The code in this folder are the version based on the IJCAI submission. The main extension is that we add an process to learn from larger datasets in an end-to-end manner 
@ Version: 2.0
@ Author: Kun Gao
'''
# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Do not use GPU
import sys
from pathlib import Path
path = Path(os.getcwd())
parent_path = path.parent.absolute()
sys.path.append(os.getcwd())
sys.path.append(str(parent_path))
sys.path.append(os.path.join(os.getcwd(),'KnowledgeGraphEmbedding','codes'))
import logging
logger = logging.getLogger()
logger.basicConfig = logging.basicConfig(level=logging.DEBUG)
from data_generator import gen
from train import t_main
from predict_extract import check_MRR_Hits, get_best_logic_programs, check_soundness, check_pl, return_all_predicate, accuracy_all_relation
import argparse
from datetime import datetime
import shutil
import pickle
from pyDatalog import pyDatalog 
from tools.pre_data_gen import data_gen
from tools.lessthan_data_gen_pb import data_gen_lessthan
from tools.lessthan_data_gen_mis import data_gen_lessthan_mis
from sample import get_samle_dataset

def set_parser():
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
    parser.add_argument('-ind', '--indicator',
                        help="check MRR and HITS indicators", type=int, default = 0)
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
                        help="check the covered test or train target positive examples accuracy from logic programs stored in best.nl file ", type=int, default = 0)
    parser.add_argument('-checktrain', '--checktrain',
                        help="check the test by default. Test from train, set the value to 1", type=int, default = 0)
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
    parser.add_argument('-bs', '--batch_size',
                        help="the batch size og the training process", type=int, default = 64)
    parser.add_argument('-ver', '--verbose',
                        help="Preview mode when using Keras to train the data: Optional value: 0,1,2. In addition, 1 indicates that model is minimal to show step; And 2 indicates that model is minimal to show batch", type=int, default = 2)
    parser.add_argument('-lar', '--lar',
                        help="whether pick the substitutations randomly when the data is large. When this flag is open, we nend to decide the values for percent flag", type=int, default = 0)
    parser.add_argument('-percent', '--subtitution_percent',
                        help="When dealing with a large dataset, how many substitutation are considered through random sampling", type=float, default = 1)
    parser.add_argument('-ranb', '--random_batch',
                        help="When sampling, the batch for a random pick", type=int, default = 5000)
    parser.add_argument('-bufs', '--buffer_size',
                        help="When sampling, the buffer size", type=int, default = 9000)
    parser.add_argument('-walk_c', '--sample_walk_check',
                        help="Whether the random walk sample algorithm is applied to the model. When this flag is open, we test the logic program based on the data in the original whole data which store in '.onl' file. Otherwies, we compute the accuracy of logic program based on the database stored in '.nl' file, which is a sub walk dataset.", type=int, default = 0)
    parser.add_argument('-walk_n', '--sample_starting_number',
                        help="Whether perform the sample algorithm. This code only perform before generating data process.", type=float, default = 0)
    parser.add_argument('-learning_rate', '--learning_rate',
                        help="The learning rate.", type=float, default = 0.001)
    parser.add_argument('-focus', '--focus',
                        help="The focus mode on, the model explore only the target predicate.", type=int, default = 0)
    parser.add_argument('-bap', '--bap', help="Use focus mode on and generate all logic program. This manner generate LPs in bottom up manner.", type=int, default = 0)
    parser.add_argument('-sodche', '--sodnesschecker', help="check the soundness of all LPs on whole datasets.", type=int, default = 0)

    args = parser.parse_args()
    print('ARGS:', args)
    return args

def data_assemble(dataset, predicate, variable_depth, large = False, sub_per = 0.2, buffer_size = 9000, random_size = 5000):
        target_arity, head_pre = gen(dataset,predicate, variable_depth, large, sub_per, buffer_size = buffer_size, random_size = random_size)
        if target_arity == -1 and head_pre == -1:
            logging.warning("Data generation on checking valid predicate time is too long!")
            raise NameError('RUNTOOLONG')
        meta_info = {'target_arity':target_arity, "head_pre": head_pre}
        with open("deepDFOL/"+dataset+'/data/'+predicate +"/meta_info.dt",'wb') as f:
            pickle.dump(meta_info,f)
            f.close()
        with open("deepDFOL/"+dataset+'/data/'+predicate +"/meta_info.txt",'w') as f:
            print(meta_info, file=f)
            f.close()
        logging.info("Generate all trainable data success with variable depth:")
        logging.info(variable_depth)
        return 0 
    
def set_logger(args, project_folder="deepDFOL"):
    '''
    Write logs to checkpoint and console
    '''
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    father_dir = os.path.join(os.getcwd(),project_folder,args.d,'result',args.p)
    
    if not os.path.exists(father_dir):
        os.makedirs(father_dir)
    
    log_file = os.path.join(father_dir, 'train.log')
    
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='a'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    return 0


def curriculum_learning(dataset, predicate,learning_rate=0.001, alpha = 10, variable_depth = 1, lar = False, bs = 64, verbose = 2, final_threshold = 0.3, sample_walk = False, learning_times = 5):
    ini_alpha = alpha
    time_train = 0
    while time_train < learning_times:
        with open("deepDFOL/"+dataset+'/data/'+predicate +"/meta_info.dt",'rb') as f:
            meta_info = pickle.load(f)
            f.close()
        target_arity = meta_info["target_arity"]
        # Train the mdoel 
        t_main(dataset,predicate, alpha = ini_alpha, learning_rate= learning_rate, target_arity = target_arity, max_epochs=2000, n_rules=5, prior_knowledge= True, variable_depth=variable_depth, larger = lar, batch_size = bs, verbose = verbose)
        # last learning rate 0.001
        # Do extract and save the best parameters 
        head_pre = meta_info["head_pre"]
        target_arity = meta_info["target_arity"]
        acc = get_best_logic_programs(dataset,predicate,head_pre,target_arity,variable_depth, final_threshold=final_threshold, sample_walk = sample_walk)
        if acc == -2:
            ini_alpha = int(ini_alpha/1.2)
            continue
        if acc >= 0.997:
            break
        time_train+=1
    return acc

def ambiguous(d, p, variable_depth):
    # only support the succ and lessthan dataset because of the data geneator python file 
    stand_derivation = [3,2.5,2,1.5,1,0.5]
    res = []

    for single_derivation in stand_derivation:
        with open("deepDFOL/"+d+'/result/'+p+"/ambi.txt",'a') as f:
            data_gen_lessthan(single_derivation,d)
            data_assemble(d,p,variable_depth)
            # remove the prior knoeledge 
            if os.path.exists("deepDFOL/"+d+'/data/'+p+"/prior_knowledge.dt"):
                os.remove("deepDFOL/"+d+'/data/'+p+"/prior_knowledge.dt")
            else:
                print("The file does not exist")
            if os.path.exists("deepDFOL/"+d+'/result/'+p+"/best.pl"):
                os.remove("deepDFOL/"+d+'/result/'+p+"/best.pl")
            else:
                print("The file does not exist")
            acc = curriculum_learning(d,p)
            res.append((single_derivation, acc))
            print(res, file=f)
            print(res)
            f.close()
    return 0

def mislabelling(d, p, variable_depth, ):
    mis_rates = [0.95,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
    res = []
    for mis_rate in mis_rates:
        with open("deepDFOL/"+d+'/result/'+p+"/mis.txt",'a') as f:
            data_gen_lessthan_mis(mis_rate, d)
            data_assemble(d,p,variable_depth)
            # remove the prior knoeledge 
            if os.path.exists("deepDFOL/"+d+'/data/'+p+"/prior_knowledge.dt"):
                os.remove("deepDFOL/"+d+'/data/'+p+"/prior_knowledge.dt")
            else:
                print("The file does not exist")
            if os.path.exists("deepDFOL/"+d+'/result/'+p+"/best.pl"):
                os.remove("deepDFOL/"+d+'/result/'+p+"/best.pl")
            else:
                print("The file does not exist")
            acc = curriculum_learning(d,p)
            res.append((mis_rate, acc))
            print(res, file=f)
            print(res)
            f.close()
    return 0 

def check_sampled_statue(d,p):
    sampled_predicate_path = os.path.join('deepDFOL', d, 'data','sampled_pred',p+'.sampled')
    if not os.path.exists(sampled_predicate_path):
        return False
    with open(sampled_predicate_path, 'rb') as f:
        sample = pickle.load(f)
        f.close()
    statue = list(sample.values())
    if 0 in statue:
        return False
    
    return True


def checkpl_mode(predicate, sample_walk_check, d, test_file, check_test ):
        try :
            with open("deepDFOL/"+d+'/data/'+predicate +"/meta_info.dt",'rb') as f:
                meta_info = pickle.load(f)
                f.close()
            target_arity = meta_info["target_arity"]
        except:
            print("No meta info on this predicate:",predicate)
            logging.warning("Set the arity of predicate to 2")
            target_arity = 2
        
        acc = check_pl(d,predicate,target_arity,file_name=test_file+'.pl', sample_walk = sample_walk_check,check_test = check_test)  
        with open(os.path.join('deepDFOL', d, 'result',predicate, 'acc_on_test.res'),'a') as f:
            print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), file = f)
            print(acc, file = f)
            f.close()
        return acc


def focus_mode(predicate, d, start_number,variable_depth, lar, sub_per, buffer_size, random_batch, learning_rate, alpha, bs, verbose, final_threshold, sample_walk_check ,break_flag = False, break_time = 5):
        data_path = 'deepDFOL/'+d+'/data/'+predicate
        result_path = 'deepDFOL/'+d+'/result/'+predicate
        # build the data folder 
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        # build the result folder
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        statue = check_sampled_statue(d,predicate)
        while statue == False:
            if start_number > 1:
                raise ValueError("Over maximum sample number!")
            try:
                get_samle_dataset(d,predicate, start_number, variable_depth)
            except:
                start_number = 2.5 * start_number
                continue
            removed_path = 'deepDFOL/'+d+'/data/'+predicate+'/'
            shutil.rmtree(removed_path)
            os.mkdir(removed_path)
            try:
                data_assemble(d,predicate,variable_depth, lar, sub_per, buffer_size, random_batch)
            except:
                start_number = start_number / 2
                continue 
            curriculum_learning(d,predicate, learning_rate,alpha, variable_depth, lar, bs, verbose, final_threshold, sample_walk = sample_walk_check, learning_times = 2)
            statue = check_sampled_statue(d,predicate)
            if break_flag == True:
                break_time -= 1
            if break_time == 0:
                return 1
        logging.info('all target predicate are considered')
        return 0


def main(args):
    # predefine all arguemsnts
    g = bool(args.g)
    # t = bool(args.t)
    indicator = bool(args.indicator)
    # c = bool(args.c)
    ap = bool(args.ap)
    d = args.d
    p = args.p
    # mp = bool(args.mp)
    cur = bool(args.cur)
    amg = bool(args.amg)
    mis = bool(args.mis)
    learning_rate = args.learning_rate
    checkpl = bool(args.checkpl)
    variable_depth = args.vd
    final_threshold = args.ft
    alpha = args.alpha
    test_file = args.testfilename
    cap = bool(args.check_acc_p)
    lar = bool(args.lar)
    bs = args.batch_size
    # ete = bool(args.endtend)
    verbose = args.verbose
    sub_per = args.subtitution_percent
    sample_walk_check = bool(args.sample_walk_check)
    start_number = args.sample_starting_number
    focus = bool(args.focus)
    check_test = bool(1- args.checktrain)
    bottom_up_ap = bool(args.bap)
    sound_checker = bool(args.sodnesschecker)
    # if ete == True:
        # lar = True

    set_logger(args)

    arg_dic = vars(args)
    non_zero_args = {}
    zero_args = {}
    for i in arg_dic:
        if arg_dic[i] == 0:
            zero_args[i] = arg_dic[i]
        else:
            non_zero_args[i] = arg_dic[i]
    
    logging.info('ARGS:\n' + str(non_zero_args) + '\n \n' + str(zero_args))
    if sound_checker == True:
        check_soundness(d, test_file, sample_walk_check)

    # args_embedding_model = parse_args(["-init","KnowledgeGraphEmbedding/models/RotatE_"+ d +"_0"])
    if start_number != 0 and ap == False and focus == False and bottom_up_ap == False:
        get_samle_dataset(d,p,start_number, variable_depth)
        

    if g == True:
        facts_path = 'deepDFOL/'+d+'/data/'+p+'.nl'
        all_fact_path = 'deepDFOL/'+d+'/data/'+d+'.nl'
        # build the facts file 
        if not os.path.exists(facts_path):
            shutil.copy(all_fact_path, facts_path) 
        data_assemble(dataset = d, predicate = p, variable_depth = variable_depth,large = lar, sub_per =  sub_per, buffer_size = args.buffer_size, random_size =  args.random_batch)
    # if t == True:
    #     with open("deepDFOL/"+d+'/data/'+p +"/meta_info.dt",'rb') as f:
    #         meta_info = pickle.load(f)
    #         f.close()
    #     target_arity = meta_info["target_arity"]
    #     t_main(d,p, alpha = alpha, learning_rate= 0.001, target_arity = target_arity, max_epochs=2000, n_rules=5, variable_depth=variable_depth) # 2000
    # if c == True: 
    #     with open("deepDFOL/"+d+'/data/'+p +"/meta_info.dt",'rb') as f:
    #         meta_info = pickle.load(f)
    #         f.close()
    #     head_pre = meta_info["head_pre"]
    #     target_arity = meta_info["target_arity"]
    #     get_best_logic_programs(d,p,head_pre,target_arity, variable_depth)
    # if i == True:
    #     # with open("deepDFOL/"+d+'/data/'+p +"/meta_info.dt",'rb') as f:
    #     #     meta_info = pickle.load(f)
    #     #     f.close()
    #     # target_arity = meta_info["target_arity"]
    #     check_Hits(d,p, test_file)
    # if mp == True:
    #     prediction(d,p)


    if cur == True:
        curriculum_learning(d,p,learning_rate, alpha ,variable_depth, lar, bs, verbose, final_threshold, sample_walk_check)
        logging.info("Finish Training.")
        
    if ap == True:
        all_pred = list(return_all_predicate(d))
        stop_singal = [0]*len(all_pred)
        all_start_number = [start_number]*len(all_pred)
        while 0 in stop_singal:
            for current_predicate in all_pred:
                logging.info(current_predicate)
                current_start_number = all_start_number[all_pred.index(current_predicate)]
                data_path = 'deepDFOL/'+d+'/data/'+current_predicate
                result_path = 'deepDFOL/'+d+'/result/'+current_predicate
                if current_start_number != 0:
                    facts_path = 'deepDFOL/'+d+'/data/'+current_predicate+'.onl'
                else:
                    facts_path = 'deepDFOL/'+d+'/data/'+current_predicate+'.nl'
                all_fact_path = 'deepDFOL/'+d+'/data/'+d+'.nl'
                # build the data folder 
                if not os.path.exists(data_path):
                    os.mkdir(data_path)
                # build the result folder
                if not os.path.exists(result_path):
                    os.mkdir(result_path)
                # build the facts file 
                if not os.path.exists(facts_path):
                    shutil.copy(all_fact_path, facts_path) 
                #Begin to subsample 
                if current_start_number != 0 and not os.path.exists('deepDFOL/'+d+'/data/'+current_predicate+'.nl'):
                    get_samle_dataset(d,current_predicate,current_start_number, variable_depth, ap_mode =  True)
                # generate trainig data
                training_data_path = data_path + '/meta_info.dt'
                if not os.path.exists(training_data_path): 
                    try:
                        # I think the time over limit situation only appears on large KBs.
                        data_assemble(d,current_predicate,variable_depth, lar, sub_per, args.buffer_size, args.random_batch)
                    except NameError:
                        all_start_number[all_pred.index(current_predicate)] = current_start_number / 2 
                        os.remove('deepDFOL/'+d+'/data/'+current_predicate+'.nl')
                        continue 
                logic_program_path = result_path + '/acc_pred.txt'
                if not os.path.exists(logic_program_path):
                    curriculum_learning(d,current_predicate, learning_rate,alpha, variable_depth, lar, bs, verbose, final_threshold, sample_walk = sample_walk_check)
                stop_singal[all_pred.index(current_predicate)] = 1 
                logging.info(str(current_predicate)+' success!')
    
    if amg == True:
        ambiguous(d, p, variable_depth)


    if mis == True:
        mislabelling(d, p, variable_depth)


        
    if checkpl == True:
        checkpl_mode(p,sample_walk_check, d, test_file, check_test)
        
    if cap == True:
        '''
        check the accuracy of the test facts. For our current ckeck strategy: if the facts can meet any statistic rules in the .l file, then the test fact can be satisfied. The reuslt is equal with HINTS@10. 
        '''
        accuracy_all_relation(d, test_file, sample_walk=sample_walk_check)

    if indicator == True:
        '''
        Check the MRR and HITNS value for a dataset after generating all logic programs according to each predicates in the task 
        '''
        check_MRR_Hits(d, test_file, sample_walk_check)
    

    if focus == True:
        focus_mode(p, d, start_number, variable_depth, lar, sub_per, args.buffer_size, args.random_batch, learning_rate, alpha, bs, verbose, final_threshold, sample_walk_check)
        
    if bottom_up_ap == True:
        all_pred = list(return_all_predicate(d))
        for current_predicate in all_pred:
            if os.path.exists(os.path.join('deepDFOL', d, 'result',current_predicate, 'acc_pred.txt')):
                logging.info('Current predicate build success')
                logging.info(current_predicate)
                continue
            data_path = 'deepDFOL/'+d+'/data/'+current_predicate
            result_path = 'deepDFOL/'+d+'/result/'+current_predicate
            # build the data folder 
            if not os.path.exists(data_path):
                os.mkdir(data_path)
            # build the result folder
            if not os.path.exists(result_path):
                os.mkdir(result_path)
            logging.info('Build current predicate LPs begin...')
            logging.info(current_predicate)
            if not os.path.exists('deepDFOL/'+d+'/data/'+current_predicate+'.onl'):
                shutil.copy('deepDFOL/'+d+'/data/'+d+'.nl','deepDFOL/'+d+'/data/'+current_predicate+'.onl')
            # acc = checkpl_mode(current_predicate, True, d, test_file, check_test)
            # if acc >= 0.9:
                # logging.info('Current predicate build success')
                # logging.info(current_predicate)
                # continue
            inner_focus_time = 5
            outer_stand_time = 2
            for out in range(outer_stand_time):
                try:
                    focus_mode(current_predicate, d, start_number, variable_depth, lar, sub_per, args.buffer_size, args.random_batch, learning_rate, alpha, bs, verbose, final_threshold, sample_walk_check, break_flag = True, break_time =  inner_focus_time)
                except:
                    continue
                # acc = checkpl_mode(current_predicate, True, d, test_file, check_test)
                # logging.info(acc)
                # if acc >= 0.9:
                    # break
            logging.info('Current predicate build success')
            logging.warning(current_predicate)
            
        
    # nations alpha = 3 learning_rate = 0.001 / 0.01
    #umls alpha = 10, learning_rate = 0.001
    return 0 

if __name__ == '__main__':
    main(set_parser())