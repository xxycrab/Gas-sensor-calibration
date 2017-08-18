#-*-coding:utf8-*-

import math
import numpy as np

def sigmoid(x):
    try:
        ans = math.exp(-x)
    except OverflowError:
        return 0
    return 1/(1+ans)

def diff_sigmoid(x):
    return (1-1/(1+math.exp(-x)))*(1/(1+math.exp(-x)))

def diff_tanh(x):
    return 1-x**2

def sgn(x):
    if x>0 :
        return 1
    else :
        return -1

def list_sgn(l):
    return map(sgn,l)

def tansig(X):
    return 2/(1+np.exp(-2*X))-1