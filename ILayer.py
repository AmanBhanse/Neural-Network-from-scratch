# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 16:42:55 2021

@author: Aman Bhanse , nitin
"""


#abstract class
class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward_propagation(self , input):
        raise NotImplementedError
        
    def backward_propagation(self , input):
        raise NotImplementedError
            


