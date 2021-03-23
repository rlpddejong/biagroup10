'''
TU/e BME Project Imaging 2021
Convolutional neural network for PCAM
Author: Mitko Veta
'''

# disable overly verbose tensorflow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}   
import tensorflow as tf

#import tensorflow.keras as keras
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout,Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
#from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications import preprocess_input
from tensorflow.keras import optimizers

def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32):
     # dataset parameters
     IMAGE_SIZE = 96
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

def get_model(weigths,dropout='Yes'):
    # the size of the images in the PCAM dataset
    IMAGE_SIZE = 96
    
    restnet = ResNet50(include_top=False, input_shape=(IMAGE_SIZE,IMAGE_SIZE,3))
    
    output = restnet.layers[-1].output
    output = tf.keras.layers.Flatten()(output)
    restnet = Model(restnet.input, output)
    for layer in restnet.layers:
        layer.trainable = False   
    
  #  model = Sequential()
   # model.add(restnet)
    #model.add(Dense(512, activation='relu', input_dim=input_shape))
    #model.add(Dropout(0.3))
    #model.add(Dense(512, activation='relu'))
    #model.add(Dropout(0.3))
    #model.add(Dense(1, activation='sigmoid'))
    restnet.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=2e-5),
                  metrics=['accuracy'])
    
    # print a summary of the model on screen
   # model.summary()

   # model.compile(SGD(lr=0.001, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])
    
    # print a summary of the model on screen
   # model.summary()

    return restnet
    #return restnet.summary()

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
    callbacks_list = [checkpoint, tensorboard]


    # train the model
    train_steps = train_gen.n//train_gen.batch_size
    val_steps = val_gen.n//val_gen.batch_size

    history = model.fit_generator(train_gen, steps_per_epoch=train_steps, 
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=3,
                    callbacks=callbacks_list)

    return

