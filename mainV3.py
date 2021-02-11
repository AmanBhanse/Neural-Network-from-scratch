# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 17:38:08 2021

@author: Aman Bhanse
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 13:02:58 2021

@author: Aman Bhanse
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 20:40:14 2021

@author: Aman Bhanse
"""

import argparse
import cv2
import matplotlib.pyplot as plt  
import numpy as np
import pickle
from os import system, name


#To use the argparser in python syntax is run FILE_NAME_WITHOUT_EXTENSION arguements
#for this example use run main test.jpg 
# for standalone i m using auto-py-to-exe


def imagePreprocessingSingleImage(img):
    img = cv2.resize(img , (28,28))
    img= img.astype("float32") #this was important 
    img = img/255
    img = img.reshape(1,28*28)
    return img


def loadTrainedModel(name): #deserailization
    return pickle.load(open(name+".dat" , "rb"))
    
def clear(): 
  
    # for windows 
    if name == 'nt': 
        _ = system('cls') 
  
    # for mac and linux(here, os.name is 'posix') 
    else: 
        _ = system('clear')
        
def predictForUserImage():
        path = input("Enter the path: ")
    
    
        img = cv2.imread(path , cv2.IMREAD_GRAYSCALE)
        
        
        #need to implement
        model = loadTrainedModel("Trainedmodel")
        
        
        if(type(img) is np.ndarray ): # condition to check is inputed properly or not
            
            
            img = imagePreprocessingSingleImage(img)
            
            
            #model's predict function accept the array of test img not single image (n_example , 1 , n_pixels)
            
            batch = []
            batch.append(img)
            
            prediction = model.predict(batch) # prediction is 3d list dim 1 = example , dim 2 = example's prediction values dim 3 = digit probability
            prediction = prediction[0][0]
            max_index =0
            max_prop = prediction[0]
            
            print("Probabilities=")
            
            for index , value in enumerate(prediction):
                print(index , "'s Confidence:\t", value)
                if(value > max_prop):
                    max_prop = value
                    max_index = index
            
            print("Final Prediction:" , max_index , "| Confidence:", max_prop)
            
            
        else:
            print("Path provided by user is wrong")
        
    
def main():
    print("---------------------------------------------------------------")
    print("|W E L C O M E   T O   H A N D   D I G I T   P R E D I C T O R|")
    print("---------------------------------------------------------------")    
        
    while True:
        
        print("\nMENU\na. User's image prediction\nb. Clear output\nc. About us\nd. Quit" )       
        choice = input(" >>>")
        choice = choice.lower()
        if choice == 'a':
            predictForUserImage()
        
        elif choice == 'b':
            clear()
            print("---------------------------------------------------------------")
            print("|W E L C O M E   T O   H A N D   D I G I T   P R E D I C T O R|")
            print("---------------------------------------------------------------")  
        elif choice == 'c':
            print("Project is developed by Aman Bhanse and Nitin Berwal\nThis project comes under © Siemens")
        elif choice == 'd':
            break
        
        else :
            print("Invalid choice, please try again")
            
        
    print("Thank you for using our product!")
    
    
    
if __name__ == "__main__":
    main()



