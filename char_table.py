import numpy as np
import pandas as pd
from sympy.combinatorics import Permutation, PermutationGroup

char_m = np.array([[1,10,15,20,30,20,24],
            [1,1,1,1,1,1,1],
          [1,-1,1,1,-1,-1,1],
          [4,2,0,1,0,-1,-1],
          [4,-2,0,1,0,1,-1],
          [6,0,-2,0,0,0,1],
          [5,1,1,-1,-1,1,0],
          [5,-1,1,-1,1,-1,0]])

col = ['(1)','(12)','(12)(34)','(123)','(1234)','(123)(45)','(12345)']
idx = ['size','1','2','3','4','5','6','7']
char_table = pd.DataFrame(data=char_m,columns=col,index=idx)
print(char_table)

def conj_size(p, n):
    t = 1
    for i in range(len(p[0])):
        t = t * np.math.factorial(p[1][i]) * p[0][i]**p[1][i]
    return np.math.factorial(n)/t

# print(conj_size([[1],[5]],5))

def inner_product(size, chi1, chi2):
    s = 0
    order = sum(size)
    for i in range(len(size)):
        s += size[i] * chi1[i] * chi2[i]
    return s/order

def delta(size, chi):
    return inner_product(size, chi, chi)


chi_new_1 = [15, -3, -1, 0, 1, 0, 0]
chi_new_2 = [21, 3, 5, 0, -1, 0, 1]
print(delta(char_m[0], chi_new_1))  
print(delta(char_m[0], chi_new_2))
print(inner_product(char_m[0], char_m[7], chi_new_1))
print('haha')
# for i in range(1, len(char_m)):
#     print(inner_product(char_m[0], char_m[i], chi_new_2))
    
print(2*np.cos(np.pi*0.4)**2+2*np.cos(np.pi*0.8)**2)