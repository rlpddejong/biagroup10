import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}   
import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential, model_from_json
from tensorflow.keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D, Dropout, Conv2D, MaxPool2D, LeakyReLU
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.applications import ResNet50V2,ResNet50
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow import keras

import pandas as pd
import sklearn.metrics as sk
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# the size of the images in the PCAM dataset
IMAGE_SIZE = 96

def get_pcam_generators_transfer(base_dir, train_batch_size=128, val_batch_size=128,shuffle=False):
    
     # dataset parameters
     train_path = os.path.join(base_dir, 'train')
     valid_path = os.path.join(base_dir, 'valid')
 
     IMAGE_SIZE = 96
     # instantiate data generators
     datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

     train_gen = datagen.flow_from_directory(train_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=train_batch_size,
                                             class_mode='binary')

     val_gen = datagen.flow_from_directory(valid_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=val_batch_size,
                                             class_mode='binary',
                                             shuffle = shuffle)

     return train_gen, val_gen
    
def get_pcam_generators_des(base_dir, train_batch_size=128, val_batch_size=128):

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


def get_transfer_model(weights):
    # the size of the images in the PCAM dataset
    IMAGE_SIZE = 96
    
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
    
    
    input = Input(input_shape)
    
    # get the pretrained model, cut out the top layer
    pretrained = ResNet50(include_top=False, weights=weights,input_shape=input_shape)
    
    # if the pretrained model it to be used as a feature extractor, and not for
    # fine-tuning, the weights of the model can be frozen in the following way
    # for layer in pretrained.layers:
    #    layer.trainable = False
    
    output = pretrained(input)
    output = GlobalAveragePooling2D()(output)
    output = Dropout(0.5)(output)
    output = Dense(1, activation='sigmoid')(output)
    
    model = Model(input, output)
    
    # note the lower lr compared to the cnn example
    model.compile(SGD(lr=0.001, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])
    
    # print a summary of the model on screen
    model.summary()

    return model

def train_transfer_model(model,train_gen,val_gen,name):
    
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
    
    
    # train the model, note that we define "mini-epochs"
    train_steps = train_gen.n//train_gen.batch_size//20
    val_steps = val_gen.n//val_gen.batch_size//20
    
    # since the model is trained for only 10 "mini-epochs", i.e. half of the data is
    # not used during training
    history = model.fit_generator(train_gen, steps_per_epoch=train_steps,
                        validation_data=val_gen,
                        validation_steps=val_steps,
                        epochs=10,
                        callbacks=callbacks_list)
    
    return

def get_des_model(kernel_size=(3,3), pool_size=(4,4), first_filters=32, second_filters=64, third_filters = 128):
     
     act = LeakyReLU(alpha=0.1)
        
     # build the model
     model = Sequential()
     
     model.add(Conv2D(first_filters, kernel_size, activation = act, padding = 'same', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
     model.add(Conv2D(first_filters, kernel_size, activation = act, padding = 'same'))
     model.add(Conv2D(first_filters, kernel_size, activation = act, padding = 'same'))
     model.add(MaxPool2D(pool_size = pool_size)) 
     model.add(Dropout(0.1))
        
     model.add(Conv2D(second_filters, kernel_size, activation = act, padding = 'same'))
     model.add(Conv2D(second_filters, kernel_size, activation = act, padding = 'same'))
     model.add(Conv2D(second_filters, kernel_size, activation = act, padding = 'same'))
     model.add(MaxPool2D(pool_size = pool_size))
     model.add(Dropout(0.1))
    
     model.add(Conv2D(third_filters, kernel_size, activation = act, padding = 'same'))
     model.add(Conv2D(third_filters, kernel_size, activation = act, padding = 'same'))
     model.add(Conv2D(third_filters, kernel_size, activation = act, padding = 'same'))
     model.add(MaxPool2D(pool_size = pool_size))
     model.add(Dropout(0.1))
        
     model.add(Flatten())
     model.add(Dense(256, activation = act))
     model.add(Dense(1, activation = 'sigmoid'))
     
     model.compile(Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-07,amsgrad=False,name="Adam"),loss='binary_crossentropy',metrics=['accuracy'])

     return model 
    
def train_des_model(model,train_gen,val_gen,name):
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
                    epochs=10,
                    callbacks=callbacks_list)
    return    

def load_model(jsonpath,weightpath):

    json_file = open(jsonpath, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json,custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU})

    # load weights into new model
    model.load_weights(weightpath)
    
    return model

def ROC_curve(model, val_gen):
    
    # obtain labels and predictions
    val_predict = model.predict_generator(val_gen)
    val_y = val_gen.classes
    
    # compute ROC curve and ROC area
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    fpr[0], tpr[0], thresholds = roc_curve(val_y, val_predict[:,0])
    roc_auc[0] = auc(fpr[0], tpr[0])
    
    # plot the ROC curve and show the area
    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='k',
             lw=lw, label='ROC curve (AUC = %0.3f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='grey', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig('ROC.eps', format='eps', dpi=1200)
    plt.show()
    
    
    
    i = np.arange(len(tpr[0])) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr[0]-(1-fpr[0]), index=i), 'threshold' : pd.Series(thresholds, index=i)})
    roc_t = roc.loc[(roc.tf-0).abs().argsort()[:1]]
    
    return list(roc_t['threshold'])

def analysis(model,val_gen,thr=0.5):
    val_predict = model.predict_generator(val_gen)
    val_y = val_gen.classes
    val_pred = []
    
    for i in range(len(val_y)):
        if val_predict[i] < thr:
            val_pred.append(0)
        else :
            val_pred.append(1)
    
    cm = sk.confusion_matrix(val_y,val_pred)
    disp = sk.ConfusionMatrixDisplay(cm)
    disp.plot(cmap=plt.cm.get_cmap('Greys_r'))
    
    plt.savefig('Confusion_matrix.eps', format='eps', dpi=600)
    
    #ppv and npv
    TN = cm[0][0]
    FN = cm[1][0]
    TP = cm[1][1]
    FP = cm[0][1]
               
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)
    
    print('Positive predictive value =',PPV)
    print('Negative predictive value =',NPV)
    return