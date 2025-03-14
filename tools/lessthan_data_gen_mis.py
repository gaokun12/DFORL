import numpy as np
from numpy import random as rd
import random
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pand



def lessthan(mis_rate=1):

    fact = []
    example = []

    
    for i in range(10):
        if i+1 != 10:
            fact.append('succ('+ str(i) +',' + str(i+1)+ ').' + 'T'+'#')       # 'T' means ture value. '#' means we apply a probability value
        
            
        for j in range(10):
            if i<j:
                example.append('lessthan('+ str(i) +',' + str(j)+ ').'+'T'+'#')
            else:
                example.append('lessthan('+ str(i) +',' + str(j)+ ').'+'.000'+'#-') # '.000' means false value, we won't test them in the test datset, hence we add '-' symbol. 
            
    total_length = len(example)
    number_mutant = int(mis_rate * total_length)
    mutanted_index = random.sample(range(0, total_length), number_mutant)
    print(total_length)
    print(mutanted_index)
    ini_index = 0
    new_example = []
    for i in example:
        if ini_index in mutanted_index:
            if 'T' in i:
                i = i.replace("T","000") 
            elif '-' in i:
                i = i.replace("000","T")
            i += '@' # For the mutanted_index, we add '@' symbol to indicated it 
        new_example.append(i)
        ini_index += 1

    file_path = 'DFOL/lessthan_mis/data/'
    with open(file_path + 'lessthan.nl', 'w') as f:
        for i in fact:
            print(i, file=f)
        for i in new_example:
            print(i, file=f)
        f.close()
            
    return 0

def pre(mis_rate):
    
    fact = []
    example = []
    
    for i in range(10):
        if i+1 != 10:
            fact.append('succ('+ str(i) +',' + str(i+1)+ ').' + 'T'+'#')       # 'T' means ture value. '#' means we apply a probability value
            example.append('pre('+ str(i+1) +',' + str(i)+ ').' + 'T'+'#') 
            # '.000' means false value, we won't test them in the test datset, hence we add '-' symbol. 
            example.append('pre('+ str(i) +',' + str(i+1)+ ').' + '.000'+'#-') 
            
            
    total_length = len(example)
    number_mutant = int(mis_rate * total_length)
    mutanted_index = random.sample(range(0, total_length), number_mutant)
    print(total_length)
    print(mutanted_index)
    ini_index = 0
    new_example = []
    for i in example:
        if ini_index in mutanted_index:
            if 'T' in i:
                i = i.replace("T","000") 
            elif '-' in i:
                i = i.replace("000","T")
            i += '@' # For the mutanted_index, we add '@' symbol to indicated it 
        new_example.append(i)
        ini_index += 1

    file_path = 'DFOL/pre_mis/data/'
    with open(file_path + 'pre.nl', 'w') as f:
        for i in fact:
            print(i, file=f)
        for i in new_example:
            print(i, file=f)
        f.close()
            
    return 0

def member(mis_rate):
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
                facts.append('cons('+new_i+','+new_j+').'+ 'T'+'#')
    for i in all_slides:
        new_i = str(i).replace(', ','>')
        facts.append('value('+new_i+','+str(i[0])+').'+ 'T'+'#')
    for i in int_elemetns:
        for j in all_slides:
            new_j = str(j).replace(', ','>')
            new_i = str(i).replace(', ','>')
            if i in j:
                example.append('member('+new_i+','+new_j+').'+ 'T'+'#')
            else:
                example.append('member('+new_i+','+new_j+').' + '.000'+'#-')

    total_length = len(example)
    number_mutant = int(mis_rate * total_length)
    mutanted_index = random.sample(range(0, total_length), number_mutant)
    print(total_length)
    print(mutanted_index)
    ini_index = 0
    new_example = []
    for i in example:
        if ini_index in mutanted_index:
            if 'T' in i:
                i = i.replace("T","000") 
            elif '-' in i:
                i = i.replace("000","T")
            i += '@' # For the mutanted_index, we add '@' symbol to indicated it 
        new_example.append(i)
        ini_index += 1

                
    file_path = 'DFOL/member_mis/data/'
    with open(file_path + 'member.nl', 'w') as f:    
        for i in facts:
            print(i, file=f)
        for i in new_example:
            print(i, file=f)
        f.close()

    return 0    
def sonof(mis_rate):
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
                example.append(new_predicate+ '..000'+'#-')
    
    for i in facts_old:
        facts.append(i+ '.T'+'#')
    for i in example_old:
        example.append(i+ '.T'+'#')
        
    total_length = len(example)
    number_mutant = int(mis_rate * total_length)
    mutanted_index = random.sample(range(0, total_length), number_mutant)
    print(total_length)
    print(mutanted_index)
    ini_index = 0
    new_example = []
    for i in example:
        if ini_index in mutanted_index:
            if 'T' in i:
                i = i.replace("T","000") 
            elif '-' in i:
                i = i.replace("000","T")
            i += '@' # For the mutanted_index, we add '@' symbol to indicated it 
        new_example.append(i)
        ini_index += 1

                
    file_path = 'DFOL/sonof_mis/data/'
    with open(file_path + 'sonof.nl', 'w') as f:    
        for i in facts:
            print(i, file=f)
        for i in new_example:
            print(i, file=f)
        f.close()
        
    return 0

def con_mis(mis_rate):
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
                example.append(new_predicate+ '..000'+'#-')
    
    for i in facts_old:
        facts.append(i+ '.T'+'#')
    for i in example_old:
        example.append(i+ '.T'+'#')
        
    total_length = len(example)
    number_mutant = int(mis_rate * total_length)
    mutanted_index = random.sample(range(0, total_length), number_mutant)
    print(total_length)
    print(mutanted_index)
    ini_index = 0
    new_example = []
    for i in example:
        if ini_index in mutanted_index:
            if 'T' in i:
                i = i.replace("T","000") 
            elif '-' in i:
                i = i.replace("000","T")
            i += '@' # For the mutanted_index, we add '@' symbol to indicated it 
        new_example.append(i)
        ini_index += 1

                
    file_path = 'DFOL/con_mis/data/'
    with open(file_path + 'con.nl', 'w') as f:    
        for i in facts:
            print(i, file=f)
        for i in new_example:
            print(i, file=f)
        f.close()
        
    return 0 

def connected_mis(mis_rate):
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
                example.append(new_predicate+ '..000'+'#-')
    
    for i in facts_old:
        facts.append(i+ '.T'+'#')
    for i in example_old:
        example.append(i+ '.T'+'#')
        
    total_length = len(example)
    number_mutant = int(mis_rate * total_length)
    mutanted_index = random.sample(range(0, total_length), number_mutant)
    print(total_length)
    print(mutanted_index)
    ini_index = 0
    new_example = []
    for i in example:
        if ini_index in mutanted_index:
            if 'T' in i:
                i = i.replace("T","000") 
            elif '-' in i:
                i = i.replace("000","T")
            i += '@' # For the mutanted_index, we add '@' symbol to indicated it 
        new_example.append(i)
        ini_index += 1

                
    file_path = 'DFOL/connected_mis/data/'
    with open(file_path + 'connected.nl', 'w') as f:    
        for i in facts:
            print(i, file=f)
        for i in new_example:
            print(i, file=f)
        f.close()
    
    return 0

def data_gen_lessthan_mis(mis_rate, dataset):
    if dataset == 'lessthan_mis':
        lessthan(mis_rate)
    elif dataset == 'pre_mis':
        pre(mis_rate)
    elif dataset == 'member_mis':
        member(mis_rate)
    elif dataset == 'sonof_mis':
        sonof(mis_rate)
    elif dataset == 'con_mis':
        con_mis(mis_rate)
    elif dataset == 'connected_mis':
        connected_mis(mis_rate)
    else:
        print('DO NOT FIND THE FILE')
    return 0

