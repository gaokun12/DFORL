import numpy as np
from numpy import random

def genrate_ran_num(stand,positive_negative = 1):
    pro = random.normal(positive_negative,stand)
    latter_str = str(pro)[1:-1]
    if pro >= 1:
        # pro = 1
        latter_str = 'T'
    elif pro <= 0:
        latter_str = '.000'
        
    return latter_str

def lessthan(stand_devriation=0.1):

    data = []
    for i in range(10):
        if i+1 != 10:
            data.append('succ('+ str(i) +',' + str(i+1)+ ').' + genrate_ran_num(stand_devriation,1)+'#')
            data.append('succ('+ str(i+1) +',' + str(i)+ ').' + genrate_ran_num(stand_devriation,0)+'#-')
            
        for j in range(10):
            if i<j:
                data.append('lessthan('+ str(i) +',' + str(j)+ ').'+ genrate_ran_num(stand_devriation,1)+'#')
            else:
                data.append('lessthan('+ str(i) +',' + str(j)+ ').'+genrate_ran_num(stand_devriation,0)+'#-')

    file_path = 'DFOL/lessthan_pb/data/'
    with open(file_path + 'lessthan.nl', 'w') as f:
        for i in range(len(data)):
            print(data[i], file=f)
        
        f.close()
    return 0

def pre(stand_devriation):
    data = []
    for i in range(10):
        if i+1 != 10:
            data.append('succ('+ str(i) +',' + str(i+1)+ ').' + genrate_ran_num(stand_devriation,1)+'#')
            data.append('succ('+ str(i+1) +',' + str(i)+ ').' + genrate_ran_num(stand_devriation,0)+'#-')
            data.append('pre('+ str(i+1) +',' + str(i)+ ').' + genrate_ran_num(stand_devriation,1)+'#')
            data.append('pre('+ str(i) +',' + str(i+1)+ ').' + genrate_ran_num(stand_devriation,0)+'#-')

    file_path = 'DFOL/pre_pb/data/'
    with open(file_path + 'pre.nl', 'w') as f:
        for i in range(len(data)):
            print(data[i], file=f)
        f.close()
    return 0

def member(stand_devriation):
    facts = []
    example = []
    int_elemetns = [4,3,2,1]
    all_slides = []
    index = 0
    while index < len(int_elemetns):
        first_part = int_elemetns[0:index]
        second_part = int_elemetns[index:len(int_elemetns)]
        all_slides.append(second_part)
        index += 1

    for i in all_slides:
        for j in all_slides:
            if j[0] == i[0]-1:
                new_i = str(i).replace(', ','>')
                new_j = str(j).replace(', ','>')
                facts.append('cons('+new_i+','+new_j+').'+ genrate_ran_num(stand_devriation,1)+'#')
    for i in all_slides:
        new_i = str(i).replace(', ','>')
        facts.append('value('+new_i+','+str(i[0])+').'+ genrate_ran_num(stand_devriation,1)+'#')
    for i in int_elemetns:
        for j in all_slides:
            new_j = str(j).replace(', ','>')
            new_i = str(i).replace(', ','>')
            if i in j:
                example.append('member('+new_i+','+new_j+').'+ genrate_ran_num(stand_devriation,1)+'#')
            else:
                example.append('member('+new_i+','+new_j+').' + genrate_ran_num(stand_devriation,0)+'#-')
                
    file_path = 'DFOL/member_pb/data/'
    with open(file_path + 'member.nl', 'w') as f:    
        for i in facts:
            print(i, file=f)
        for i in example:
            print(i, file=f)
        f.close()    
    
    return 0

def sonof(stand_devriation):
    facts_old = ['father(a,b)','father(a,c)','father(d,e)','father(d,f)','father(g,h)','father(g,i)','brother(b,c)','brother(c,b)','brother(e,f)','sister(f,e)','sister(h,i)','sister(i,h)']
    example_old = ['sonof(b,a)','sonof(c,a)','sonof(e,d)','sonof(f,d)','sonof(h,g)','sonof(i,g)']
    example = []
    facts = []
    objects = ['a','b','c','d','e','f','g','h','i']
    for i in objects:
        for j in objects:
            if i == j :
                continue
            new_predicate = 'sonof('+i+','+j+')'
            if new_predicate not in example_old:
                example.append(new_predicate+ '.'+genrate_ran_num(stand_devriation, 0)+'#-')
    
    for i in facts_old:
        facts.append(i+ '.' + genrate_ran_num(stand_devriation,1)+'#')
    for i in example_old:
        example.append(i+ '.'+genrate_ran_num(stand_devriation,1)+'#')
        
        
    file_path = 'DFOL/sonof_pb/data/'
    with open(file_path + 'sonof.nl', 'w') as f:    
        for i in facts:
            print(i, file=f)
        for i in example:
            print(i, file=f)
        f.close()
        
    
    return 0


def con(stand_devriation):
    facts_old = ['edge(a,b)','edge(b,c)','edge(c,d)','edge(b,a)']
    example_old = ['con(a,b)','con(b,c)','con(c,d)','con(b,a)','con(a,c)','con(a,d)','con(a,a)','con(b,d)','con(b,a)','con(b,b)']
    example = []
    facts = []
    objects = ['a','b','c','d','e','f']
    for i in objects:
        for j in objects:
            if i == j :
                continue
            new_predicate = 'con('+i+','+j+')'
            if new_predicate not in example_old:
                example.append(new_predicate+ '.' + genrate_ran_num(stand_devriation,0)+'#-')
    
    for i in facts_old:
        facts.append(i+ '.'+ genrate_ran_num(stand_devriation,1)+'#')
    for i in example_old:
        example.append(i+ '.'+ genrate_ran_num(stand_devriation,1)+'#')
        

                
    file_path = 'DFOL/con_pb/data/'
    with open(file_path + 'con.nl', 'w') as f:    
        for i in facts:
            print(i, file=f)
        for i in example:
            print(i, file=f)
        f.close()
        
    return 0 

def connected(stand_devriation):
    facts_old = ['edge(a,b)','edge(b,a)','edge(b,d)','edge(d,b)','edge(d,e)','edge(e,d)','edge(c,c)']
    example_old = ['connected(a,b)','connected(b,a)','connected(d,e)','connected(e,d)','connected(b,a)','connected(b,d)','connected(d,b)','connected(c,c)']
    example = []
    facts = []
    objects = ['a','b','c','d','e']
    for i in objects:
        for j in objects:
            if i == j :
                continue
            new_predicate = 'connected('+i+','+j+')'
            if new_predicate not in example_old:
                example.append(new_predicate+ '.'+ genrate_ran_num(stand_devriation, 0)+'#-')
    
    for i in facts_old:
        facts.append(i+ '.'+ genrate_ran_num(stand_devriation,1)+'#')
    for i in example_old:
        example.append(i+ '.'+ genrate_ran_num(stand_devriation,1)+'#')
        
                
    file_path = 'DFOL/connected_pb/data/'
    with open(file_path + 'connected.nl', 'w') as f:    
        for i in facts:
            print(i, file=f)
        for i in example:
            print(i, file=f)
        f.close()
    
    return 0

def data_gen_lessthan(stand, dataset):
    if dataset == 'lessthan_pb':
        lessthan(stand)
    elif dataset == 'pre_pb':
        pre(stand)
    elif dataset == 'member_pb':
        member(stand)
    elif dataset == 'sonof_pb':
        sonof(stand)
    elif dataset == 'con_pb':
        con(stand)
    elif dataset == 'connected_pb':
        connected(stand)
    else:
        print('DO NOT FIND THE FILE')
    return 0



