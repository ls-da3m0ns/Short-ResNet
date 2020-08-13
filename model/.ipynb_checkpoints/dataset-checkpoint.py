import tensorflow as tf 
import numpy as np
from sklearn.utils import shuffle

class Dataset(object):
    
    def __init__(self,normalize = True):
        print("\nInitializing Dataset ... \n")
        
        self.normalize = normalize 
        self.train_batch_index = 0
        self.test_batch_index = 0
        
        (train_images,train_labels),(test_images,test_labels) = tf.keras.datasets.fashion_mnist.load_data()
        
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels
        
        self.num_train = train_images.shape[0]
        self.num_test = test_images.shape[0]
        
        self.initial_min_val = train_images[0].min()
        self.initial_max_val = train_images[0].max()
        
        self.num_classes = train_labels[0].max() + 1
        
        self.height, self.width = train_images.shape[1],train_images.shape[2]
        
        try: self.channels = train_images.shape[3]
        except: self.channels = 1
        
        if normalize:
            train_images = (train_images - self.initial_min_val) / (self.initial_max_val - self.initial_min_val)
            test_images = (test_images - self.initial_min_val) / (self.initial_max_val - self.initial_min_val)
        
        print("Dataset Info :")
        print("Shape height: {} width:{} channel: {} ".format(self.height,self.width, self.channels))
        print("No of clases : {}".format(self.num_classes))
        if normalize:
            print("Initial min max values {} {} new min max values {} {}".format(self.initial_min_val,self.initial_max_val,train_images[0].min(),train_images[0].max()))
        
    def labels2vector(self,labels):
        label_vector = []
        for label in range(labels.shape[0]):
            tmp_vector = np.eye(self.num_classes)[labels[label]]
            label_vector.append(tmp_vector)
            
        label_vector = np.array(label_vector)    
        
        return label_vector
    
    def next_train_batch(self,batch_size = 1,terminator = False):
        start = self.train_batch_index 
        if (self.train_batch_index + batch_size >= self.num_train):
            end = self.num_train 
            terminator = True
        else : 
            end = self.train_batch_index + batch_size
            self.train_batch_index = self.train_batch_index + batch_size
            
            
        images,labels = self.train_images[start:end],self.train_labels[start:end]
        images = np.expand_dims(images,axis=3)
        labels = self.labels2vector(labels)
        
        return images, labels, terminator
        
    def next_test_batch(self,batch_size = 1,terminator=False):
        start = self.test_batch_index 
        if (self.test_batch_index + batch_size >= self.num_test):
            end = self.num_test
            terminator = True
        else : 
            end = self.test_batch_index + batch_size
            self.test_batch_index = self.test_batch_index + batch_size
        
        images , labels = self.test_images[start:end],self.test_labels[start:end]
        images = np.expand_dims(images,axis=3)
        labels = self.labels2vector(labels)
        
        return images,labels, terminator
        
        
        