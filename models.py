#!/usr/bin/env python
# coding: utf-8

# In[10]:


from keras.models import Model,load_model
from keras.layers import Conv2D,TimeDistributed,LSTM,Flatten,Input,Dense,Dropout,MaxPool2D,BatchNormalization,MaxPooling2D,merge,Activation,add,concatenate,GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU


# In[13]:


class ModelType:
    RGB = 0
    FLOW = 1
    FULL = 2


# In[24]:


class ModelFactory:
    def __init__(self,flowpath='trained_models/flowlstm.h5',rgbpath='trained_models/rgblstm.h5',fullpath='trained_models/rgbflow.h5',trained=True):
        self.flowpath = flowpath
        self.rgbpath = rgbpath
        self.fullpath = fullpath
        self.trained = trained
    def getModel(self,model):
       
        if model is ModelType.RGB:
            return self.__rgbmodel()
        elif model is ModelType.FLOW:
            return self.__flowmodel()
        else:
            return self.__fullmodel()#
        
        
    def __rgbmodel(self):
        if self.trained:
            return load_model(self.rgbpath)
        else:
            return __rgb()
        
        
    def __flowmodel(self):
        if self.trained:
            return load_model(self.flowpath)
        else:
            return __flow()
        
        
    def __fullmodel(self):
        if self.trained:
            return load_model(self.fullpath)
        else:
            return __full() # full training should be done after seperate rgb and flow trainings
        
    def __full(self):
    
        rgbmodel = load_model(self.rgbpath)
        flowmodel = load_model(self.flowpath)

        for layer in rgbmodel.layers:
            layer.trainable = False
        for layer in flowmodel.layers:
            layer.trainable = False

        conc = concatenate([rgbmodel.layers[-2].output,flowmodel.layers[-2].output])#concatenate lstm outputs

        x = Dense(2048)(conc)
        x = Dropout(0.4)(x)
        x = Dense(2048)(x)

        pred = Dense(27,activation='softmax')(x)

        fullmodel = Model(inputs=[flowmodel.input,rgbmodel.input],outputs=pred ,name='rgb_flow_lstm')
        #fullmodel.summary()
        return fullmodel
    
    def __rgb(self):

        rgbinput = Input((10,150,100,3),name='rgbinput')

        x = TimeDistributed(Conv2D(kernel_size=(3,3),filters=64,padding='same'))(rgbinput)
        x = TimeDistributed(LeakyReLU(0.01))(x)
        x = TimeDistributed(BatchNormalization())(x)

        x = TimeDistributed(Conv2D(kernel_size=(3,3),filters=64,padding='same'))(x)
        x = TimeDistributed(LeakyReLU(0.01))(x)
        x = TimeDistributed(BatchNormalization())(x)

        x= TimeDistributed(MaxPooling2D((2,2),strides=(2,2),data_format='channels_last'))(x)

        x = TimeDistributed(Conv2D(kernel_size=(3,3),filters=128,padding='same'))(x)
        x = TimeDistributed(LeakyReLU(0.01))(x)
        x = TimeDistributed(BatchNormalization())(x)

        x= TimeDistributed(MaxPooling2D((2,2),strides=(2,2),data_format='channels_last'))(x)

        x = TimeDistributed(Conv2D(kernel_size=(3,3),filters=256,padding='same'))(x)
        x = TimeDistributed(LeakyReLU(0.01))(x)
        x = TimeDistributed(BatchNormalization())(x)

        x = TimeDistributed(Conv2D(kernel_size=(3,3),filters=512,padding='same'))(x)
        x = TimeDistributed(LeakyReLU(0.01))(x)
        x = TimeDistributed(BatchNormalization())(x)

        x = TimeDistributed(Conv2D(kernel_size=(3,3),filters=512,padding='same'))(x)
        x = TimeDistributed(LeakyReLU(0.01))(x)
        x = TimeDistributed(BatchNormalization())(x)

        x= TimeDistributed(MaxPooling2D((2,2),strides=(2,2),data_format='channels_last'))(x)

        x = TimeDistributed(Conv2D(kernel_size=(3,3),filters=1024,padding='same'))(x)
        x = TimeDistributed(LeakyReLU(0.01))(x)
        x = TimeDistributed(BatchNormalization())(x)

        x = TimeDistributed(Conv2D(kernel_size=(3,3),filters=1024,padding='same'))(x)
        x = TimeDistributed(LeakyReLU(0.01))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(GlobalAveragePooling2D())(x)
        x =  TimeDistributed(Dropout(0.4))(x)

        x = TimeDistributed(Dense(1024))(x)
        x =  TimeDistributed(Dropout(0.4))(x)

        x = LSTM(1024)(x)


        pred = Dense(27,activation ='softmax')(x)


        rgbmodel = Model(inputs=rgbinput,outputs=pred,name='rgb_model')
        rgbmodel.compile(Adam(0.0001),loss='categorical_crossentropy',metrics=['categorical_accuracy'])
        #rgbmodel.summary()
        return rgbmodel

    def __flow(self):

        flowinput = Input((10,150,100,2),name='flowinput')

        x = TimeDistributed(Conv2D(kernel_size=(3,3),filters=64,padding='same'))(flowinput)
        x = TimeDistributed(LeakyReLU(0.01))(x)
        x = TimeDistributed(BatchNormalization())(x)

        x = TimeDistributed(Conv2D(kernel_size=(3,3),filters=64,padding='same'))(x)
        x = TimeDistributed(LeakyReLU(0.01))(x)
        x = TimeDistributed(BatchNormalization())(x)

        x= TimeDistributed(MaxPooling2D((2,2),strides=(2,2),data_format='channels_last'))(x)

        x = TimeDistributed(Conv2D(kernel_size=(3,3),filters=128,padding='same'))(x)
        x = TimeDistributed(LeakyReLU(0.01))(x)
        x = TimeDistributed(BatchNormalization())(x)

        x= TimeDistributed(MaxPooling2D((2,2),strides=(2,2),data_format='channels_last'))(x)

        x = TimeDistributed(Conv2D(kernel_size=(3,3),filters=256,padding='same'))(x)
        x = TimeDistributed(LeakyReLU(0.01))(x)
        x = TimeDistributed(BatchNormalization())(x)

        x = TimeDistributed(Conv2D(kernel_size=(3,3),filters=512,padding='same'))(x)
        x = TimeDistributed(LeakyReLU(0.01))(x)
        x = TimeDistributed(BatchNormalization())(x)

        x = TimeDistributed(Conv2D(kernel_size=(3,3),filters=512,padding='same'))(x)
        x = TimeDistributed(LeakyReLU(0.01))(x)
        x = TimeDistributed(BatchNormalization())(x)

        x= TimeDistributed(MaxPooling2D((2,2),strides=(2,2),data_format='channels_last'))(x)

        x = TimeDistributed(Conv2D(kernel_size=(3,3),filters=1024,padding='same'))(x)
        x = TimeDistributed(LeakyReLU(0.01))(x)
        x = TimeDistributed(BatchNormalization())(x)

        x = TimeDistributed(Conv2D(kernel_size=(3,3),filters=1024,padding='same'))(x)
        x = TimeDistributed(LeakyReLU(0.01))(x)
        x = TimeDistributed(BatchNormalization())(x)


        x = TimeDistributed(GlobalAveragePooling2D())(x)
        x =  TimeDistributed(Dropout(0.4))(x)

        x = TimeDistributed(Dense(1024))(x)
        x =  TimeDistributed(Dropout(0.4))(x)

        x = LSTM(1024)(x)

        pred = Dense(27,activation ='softmax')(x)


        flowmodel = Model(inputs=flowinput,outputs=pred,name='flow_model')

        flowmodel.compile(Adam(0.0001),loss='categorical_crossentropy',metrics=['categorical_accuracy'])
        #flowmodel.summary()
    
        return flowmodel


# In[ ]:




