# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 17:20:19 2021

@author: Aman Bhanse , nitin
"""

import ILayer
import LayerTypes
import lossFunction
import activationFunction
import pickle

class Model:
    def __init__(self):
        self.layers = [] #list of layer
        self.loss = None
        self.loss_prime = None
        
    def add(self , layer):
        self.layers.append(layer)
        
    def use(self , loss , loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime
        
    
    def predict(self , input_data):
        n_examples = len(input_data)
        result=[]
        
        #print("In predict function")
        #print("Examples:" , n_examples)
        
        for i in range(n_examples):
            output= input_data[i]
            
            for layer in self.layers:
                #print(output.shape)
                output = layer.forward_propagation(output)
            result.append(output)
            
        return result
        
    def fit(self , x_train , y_train , epochs , learning_rate):
        # number of examples
        samples = len(x_train)
        print("Training Examples:" , samples)
        
        for i in range(epochs):  #dataset iteration
            err = 0
            for j in range(samples):    #example iteration
                
                output = x_train[j]
                
    
                
                #forward prop
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                #debugging
#                print(output)
#                print("output shape:" , output.shape)
#                print("output type:" ,type(output))
                
                #debuggnig ends
                
                #compute the loss
                err += self.loss(y_train[j] , output)
                
                #backpropagation
                error = self.loss_prime(y_train[j] , output)  # loss derivative at y = y_train[j]
                
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error , learning_rate)
            
            err = err/samples
            print('EPOCH %d/%d error=%f' %(i+1, epochs , err))
            
            
    def summary(self):
        print("no of layers:", len(self.layers))
        
    
    def saveModel(self,name): #Serialization 
        pickle.dump(self , open(name+".dat" , "wb"))
        
        
        
            
    
            