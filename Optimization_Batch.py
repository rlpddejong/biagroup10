
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}   
import tensorflow as tf

import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LeakyReLU
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow import keras

import matplotlib.pyplot as plt


from sklearn.metrics import roc_curve, auc


# the size of the images in the PCAM dataset
IMAGE_SIZE = 96

def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32):

     # dataset parameters
     TRAIN_PATH = os.path.join(base_dir, 'train')
     VALID_PATH = os.path.join(base_dir, 'valid')

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
 
 
def get_model(kernel_size=(3,3), pool_size=(4,4), first_filters=32, second_filters=64, third_filters = 128):
     
     act = LeakyReLU(alpha=0.1)
        
     # build the model
     model = Sequential()
     
     model.add(Conv2D(first_filters, kernel_size, activation = act, padding = 'same', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
     model.add(Conv2D(first_filters, kernel_size, activation = act, padding = 'same'))
     model.add(Conv2D(first_filters, kernel_size, activation = act, padding = 'same'))
     model.add(MaxPool2D(pool_size = pool_size)) 
     
     model.add(Conv2D(second_filters, kernel_size, activation = act, padding = 'same'))
     model.add(Conv2D(second_filters, kernel_size, activation = act, padding = 'same'))
     model.add(Conv2D(second_filters, kernel_size, activation = act, padding = 'same'))
     model.add(MaxPool2D(pool_size = pool_size))
     
     model.add(Conv2D(third_filters, kernel_size, activation = act, padding = 'same'))
     model.add(Conv2D(third_filters, kernel_size, activation = act, padding = 'same'))
     model.add(Conv2D(third_filters, kernel_size, activation = act, padding = 'same'))
     model.add(MaxPool2D(pool_size = pool_size))
     
     model.add(Flatten())
     model.add(Dense(256, activation = act))
     model.add(Dense(1, activation = 'sigmoid'))
     
     model.compile(Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-07,amsgrad=False,name="Adam"),loss='binary_crossentropy',metrics=['accuracy'])

     return model 
   
def train_model(model,train_gen,val_gen,name):
    # save the model and weights
    model_name = name
    model_filepath = model_name + '.json'
    weights_filepath = model_name + '_weights.hdf5'

    model_json = model.to_json() # serialize model to JSON
    with open(model_filepath, 'w') as json_file:
        json_file.write(model_json) 


    # define the model checkpoint and Tensorboard callbacks
    checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    tensorboard = TensorBoard(os.path.join('logs', model_name))
    callbacks_list = [checkpoint,tensorboard]


    # train the model
    train_steps = train_gen.n//train_gen.batch_size
    val_steps = val_gen.n//val_gen.batch_size

    history = model.fit_generator(train_gen, steps_per_epoch=train_steps, 
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=3,
                    callbacks=callbacks_list)
    return