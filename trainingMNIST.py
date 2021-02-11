# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 19:21:27 2021

@author: Aman Bhanse
"""

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib
from model import Model
from LayerTypes import FCLayer
from activationLayer import ActivationLayer
from activationFunction import tanh , tanh_prime
from lossFunction import mse , mse_prime


def mnist_data_loader(path):
    dataset = pd.read_csv(path)
    return dataset

def mnist_x_y_split(dataset):
    y = dataset['label']
    y = pd.DataFrame(y, columns=['label'])
    
    #since the output is 10 node so we have to encode them    
    y_encoded = pd.get_dummies(y.label)
    y_encoded = y_encoded.values.astype('float32')
    
    
    x = dataset.drop('label' , axis=1)  
    
    #since the input is single image so its x's dimension must be (n_example , 1 , n_pixels)
    x = x.values.reshape(x.shape[0] , 1,28*28)  
    x = x.astype('float32')

    return x,y_encoded



def mnist_model():
    
    model = Model()
    
    model.add(FCLayer(28*28 , 100))
    
    model.add(ActivationLayer(tanh , tanh_prime))
    
    model.add(FCLayer(100,50))
    
    model.add(ActivationLayer(tanh, tanh_prime))
    
    model.add(FCLayer(50, 10)) 
    
    model.add(ActivationLayer(tanh, tanh_prime))
    
    model.use(mse , mse_prime)
    
    return model    



def main():
    dataset = mnist_data_loader('mnist.csv')
    x , y = mnist_x_y_split(dataset)
    model = mnist_model()
    model.fit(x[0:9000] , y[0:9000], epochs=35 , learning_rate=0.01)
    
    model.saveModel("TrainedmodelV2")
    
    #pred = model.predict(x[9001: 9010])
    #print(pred)
    #print(y[9001: 9010])
    #x1 = x[9001]
    #x1.reshape(28,28)
    #plt.imshow(x1 , cmap=matplotlib.cm.binary , interploation='nearest')
if __name__ == '__main__':
    main()
    

    


