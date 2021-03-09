# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 16:21:09 2021

@author: 20182372
"""

import sklearn.metrics as sk
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}   
import tensorflow as tf

import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPool2D, Dropout, BatchNormalization, GlobalMaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow import keras


# the size of the images in the PCAM dataset
IMAGE_SIZE = 96

def analysis(model,val_gen):
    val_predict = model.predict_generator(val_gen)
    val_y = val_gen.classes
    val_pred = []
    
    for i in range(len(val_y)):
        if val_predict[i] <= 0.5:
            val_pred.append(0)
        else :
            val_pred.append(1)
    
    cm = sk.confusion_matrix(val_y,val_pred)
    disp = sk.ConfusionMatrixDisplay(cm)
    disp.plot(cmap=plt.cm.get_cmap('Greys_r'))
    
    #ppv and npv
    TN = cm[0][0]
    FN = cm[1][0]
    TP = cm[1][1]
    FP = cm[0][1]
               
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)
    
    print(PPV,NPV)  
    return


def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32):

     # dataset parameters
     TRAIN_PATH = os.path.join(base_dir, 'train+val', 'train')
     VALID_PATH = os.path.join(base_dir, 'train+val', 'valid')

     RESCALING_FACTOR = 1./255
     
     # instantiate data generators
     datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)

     train_gen = datagen.flow_from_directory(TRAIN_PATH,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=train_batch_size,
                                             class_mode='binary')

     val_gen = datagen.flow_from_directory(VALID_PATH,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=val_batch_size,
                                             class_mode='binary',
                                             shuffle=False)
     
     return train_gen, val_gen



def get_model(kernel_size=(3,3), pool_size=(4,4), first_filters=32, second_filters=64):
    
     # build the model
     model = Sequential()

     model.add(Conv2D(first_filters, kernel_size, activation = 'relu', padding = 'same', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
     model.add(MaxPool2D(pool_size = pool_size)) 

     model.add(Conv2D(second_filters, kernel_size, activation = 'relu', padding = 'same'))
     model.add(MaxPool2D(pool_size = pool_size))

     model.add(Flatten())
     model.add(Dense(64, activation = 'relu'))
     model.add(Dense(1, activation = 'sigmoid'))
     
    
     # compile the model
     model.compile(SGD(lr=0.01, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])

     return model



# ROC analysis
def ROC_curve(model, val_gen):
    
    # obtain labels and predictions
    val_predict = model.predict_generator(val_gen)
    val_y = val_gen.classes
    
    # compute ROC curve and ROC area
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    fpr[0], tpr[0], _ = roc_curve(val_y, val_predict[:,0])
    roc_auc[0] = auc(fpr[0], tpr[0])
    
    # plot the ROC curve and show the area
    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',
             lw=lw, label='ROC curve (AUC = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
    return