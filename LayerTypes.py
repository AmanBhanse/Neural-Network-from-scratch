# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 16:49:40 2021

@author: Aman Bhanse , nitin
"""

from ILayer import Layer
import numpy as np



class FCLayer(Layer):   #syntax for inheritance , inheriting Layer absbtract class
    def __init__(self , input_size , output_size):
        self.weights = np.random.rand(input_size , output_size) - 0.5
        self.bias = np.random.rand(1 , output_size) - 0.5
        
    
    def forward_propagation(self , input_data):
        self.input = input_data
        self.output = np.dot(self.input , self.weights) + self.bias
        return self.output
    
    def backward_propagation(self , output_error , learning_rate):
        input_error = np.dot(output_error , self.weights.T)
        weight_error = np.dot(self.input.T , output_error)
        bias_error = 1*output_error
        
        #updating the parameters
        self.weights = self.weights - learning_rate*weight_error
        self.bias = self.bias - learning_rate*bias_error
        return input_error
    
class Conv2d(Layer):
      def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.conv_filter = np.random.randn(filter_size, filter_size)/(filter_size * filter_size)
    
      def image_region(self,image): #generator function
        height, width = image.shape
        self.image = image
        for j in range(height - self.filter_size + 1):
          for k in range(width - self.filter_size +1):
            image_patch = image[j:(j+self.filter_size),k:(k+self.filter_size)]
            yield image_patch, j, k
    
      def forward_propagation(self,image):
        self.input = image
        height, width = image.shape
        conv_out = np.zeros((height - self.filter_size + 1, width - self.filter_size+1))
        for image_path, i,j in self.image_region(image):
          conv_out[i,j]= np.sum(image_path * self.conv_filter, axis=(0,1))
        
        self.output = conv_out
        return conv_out
    
      def backward_propagation(self, dL_dout, learning_rate):
        dL_dF_params = np.zeros(self.conv_filter.shape)
        for image_patch, i,j in self.image_region(self.image):
            dL_dF_params += image_patch*dL_dout[i,j]
    
        #filter params update
        self.conv_filter -= learning_rate*dL_dF_params
        return dL_dF_params

class Flatten(Layer):
    def forward_propagation(self, input_data):
        self.input = input_data
        self._shape = input_data.shape
        self.output = input_data.reshape(1 , input_data.shape[0]*input_data.shape[1])
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return output_error.reshape(self._shape)

    
