# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 17:09:29 2021

@author: Aman Bhanse , nitin
"""

import numpy as np

def tanh(x):
    print(x.shape)
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

