# import itertools as it
# a = it.product([1,2],[3,4,5])
# flag = 0

# def get_batch_elements(substitutation, batch_size):
#     '''
#     Generate a gatch sized elelmetns from the iterative objects.
#     '''
#     ini_number = 0
#     batch_sub = []
#     while ini_number < batch_size:
#         try:
#             batch_sub.append(next(substitutation))
#             ini_number += 1
#         except StopIteration:
#             return batch_sub
#     return batch_sub
# sa = get_batch_elements(a, 4)
# print(sa)
# sa = get_batch_elements(a, 4)
# print(sa)
# sa = get_batch_elements(a, 4)
# print(sa)

import itertools as it
import random
random.seed(70)

def ramdom_buffer(buffer,seed,size):
    
    
    
a = [1,4]
b = ['a','b','c']
dd = it.random_product(a,b,repeat=3)
print(dd)

