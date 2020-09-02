import numpy as np # linear algebra
import tensorflow
import cv2 #opencv library
import os

from tensorflow.keras.preprocessing.image import img_to_array, load_img


class Processes:
    
    def __init__(self, folder):
        self.folder=folder
        #list all images in given folder path
        images = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        
        train_files = []
        y_train = []
        i=0
        
        for image in images:
            train_files.append(image)
            label_image = image.find("_")
            y_train.append(image[0:label_image])
            
        print("Files in train_files: %d" % len(train_files))
        
        # Original Dimensions
        image_width = 128
        image_height = 128     
        self.dataset = np.ndarray(shape=(len(train_files), image_height, image_width),
                             dtype=np.float32)
        
        for _file in train_files:
            # image is loaded
            img = load_img(folder + "/" + _file)
            #fix the image sizes
            img.thumbnail((image_width, image_height))
            # Convert to Numpy Array
            arrayI = img_to_array(img)  
            #convert images to grey level  
            grayscaleI = cv2.cvtColor(arrayI, cv2.COLOR_BGR2GRAY)
            # Normalize the images with linear normalization [0 , 255] to [-1 , 1]
            pixelMax=255
            pixelMin=0
            newMax=1
            newMin=0
            newI = (grayscaleI - pixelMin) * (newMax - newMin) / (pixelMax - pixelMin)  + newMin
            self.dataset[i] = newI
            i += 1
        
        
        
        from sklearn.model_selection import train_test_split
        
        #Splitting 
        X_train, X_test, y_train, y_test = train_test_split(self.dataset, y_train, test_size=0.2, random_state=33)
        print("Train set size: {0},Test set size: {1}".format(len(X_train), len(X_test)))
        self.X_train=X_train
        self.X_test=X_test
        print("-----------------")
        #print("max:{0} , min: {1}".format(X_train.max(), X_train.min()))
            
            
            