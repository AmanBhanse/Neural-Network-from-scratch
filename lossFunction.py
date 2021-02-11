# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 17:12:06 2021

@author: Aman Bhanse , nitin
"""

import numpy as np

def mse(y_true , y_pred):
    return np.mean(np.power(y_true - y_pred , 2 ))   #((y_pred - actual value)^2)/2

def mse_prime(y_true , y_pred):
    return 2.0*(y_pred - y_true)/y_true.size



