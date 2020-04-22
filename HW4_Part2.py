# -*- coding: utf-8 -*-
"""
Name: Sourav Yadav
Assignment 4
AID: A20450418
Spring 2020
Deep Learning

"""
import keras
import numpy as np
from keras.datasets import cifar10
from keras import optimizers
from keras.models import load_model
from keras import models
from keras import layers
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import losses

#Load Data
(train_data,train_labels),(test_data,test_labels)=cifar10.load_data()
train_data=train_data.reshape((50000,32,32,3))
train_data=train_data/255.0
test_data=test_data.reshape((10000,32,32,3))
test_data=test_data/255.0
train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)

x_train,x_val,y_train,y_val = train_test_split(train_data,train_labels,test_size = 0.2)


from keras import layers
from keras import models

model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),input_shape=(32,32,3)))
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(64,(3,3)))
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(64,(3,3)))
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
#model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(64,(3,3)))
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
#model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(128,(1,1)))
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
#model.add(layers.MaxPool2D((2,2)))



model.add(layers.Flatten())
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.summary()

rmsprop=optimizers.RMSprop(learning_rate=0.0001, rho=0.9)

model.compile(optimizer=rmsprop,loss=losses.categorical_crossentropy,metrics=['accuracy'])

history=model.fit(x_train,y_train,
                  epochs=10,
                  batch_size=20,
                  validation_data=(x_val,y_val))


history_dict=history.history

model.save("model_cifar10.hdf5")
print("Model Saved")

import matplotlib.pyplot as plt
loss_values=history_dict['loss']
val_loss_values=history_dict['val_loss']
epochs=range(1,len(loss_values)+1)

#Loss Plot
plt.plot(epochs,loss_values,'bo',label='Training Loss')
plt.plot(epochs,val_loss_values,'b',label='Validation Loss')
plt.title('Traning and Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

#Accuracy Plot
#plt.clf()
acc_values=history_dict['accuracy']
val_acc_values=history_dict['val_accuracy']
plt.plot(epochs,acc_values,'bo',label='Training ACcuy')
plt.plot(epochs,val_acc_values,'b',label='Validation Accuracy')
plt.title('Traning and Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()



print("Loading Saved Model")
model_new=load_model("model_cifar10.hdf5")

#Evaluating Model on test Data.
result=model_new.evaluate(test_data,test_labels)
print("Test Results")
print("Loss :",result[0])
print("Accuracy :",result[1])


#-----------------------------------------Adding Inception Blocks ---------------------------
from keras.models import Model
from keras import layers
from keras import Input

input_tensor=Input(shape=(32,32,3))

#Layer 1
x=layers.Conv2D(32,(3,3))(input_tensor)
x=layers.BatchNormalization()(x)
x=layers.Activation("relu")(x)
z2=layers.MaxPool2D((2,2))(x)


#Inception Block 
branch_a=layers.Conv2D(128,1,activation='relu',strides=2,padding='same')(z2)
branch_b=layers.Conv2D(128,1,activation='relu',padding='same')(z2)
branch_b=layers.Conv2D(128,3,activation='relu',strides=2,padding='same')(branch_b)
branch_c=layers.AveragePooling2D(3,strides=2,padding='same')(z2)
branch_c=layers.Conv2D(128,3,activation='relu',padding='same')(branch_c)
branch_d=layers.Conv2D(128,1,activation='relu',padding='same')(z2)
branch_d=layers.Conv2D(128,3,activation='relu',padding='same')(branch_d)
branch_d=layers.Conv2D(128,3,activation='relu',strides=2,padding='same')(branch_d)
z2=layers.concatenate([branch_a,branch_b,branch_c,branch_d], axis=-1)


#Inception Block 
branch_a=layers.Conv2D(128,1,activation='relu',strides=2,padding='same')(z2)
branch_b=layers.Conv2D(128,1,activation='relu',padding='same')(z2)
branch_b=layers.Conv2D(128,3,activation='relu',strides=2,padding='same')(branch_b)
branch_c=layers.AveragePooling2D(3,strides=2,padding='same')(z2)
branch_c=layers.Conv2D(128,3,activation='relu',padding='same')(branch_c)
branch_d=layers.Conv2D(128,1,activation='relu',padding='same')(z2)
branch_d=layers.Conv2D(128,3,activation='relu',padding='same')(branch_d)
branch_d=layers.Conv2D(128,3,activation='relu',strides=2,padding='same')(branch_d)
z2=layers.concatenate([branch_a,branch_b,branch_c,branch_d], axis=-1)


branch_a=layers.Conv2D(128,1,activation='relu',strides=2,padding='same')(z2)
branch_b=layers.Conv2D(128,1,activation='relu',padding='same')(z2)
branch_b=layers.Conv2D(128,3,activation='relu',strides=2,padding='same')(branch_b)
branch_c=layers.AveragePooling2D(3,strides=2,padding='same')(z2)
branch_c=layers.Conv2D(128,3,activation='relu',padding='same')(branch_c)
branch_d=layers.Conv2D(128,1,activation='relu',padding='same')(z2)
branch_d=layers.Conv2D(128,3,activation='relu',padding='same')(branch_d)
branch_d=layers.Conv2D(128,3,activation='relu',strides=2,padding='same')(branch_d)
res_output=layers.concatenate([branch_a,branch_b,branch_c,branch_d], axis=-1)

#Layer2
y=layers.Conv2D(64,(3,3),padding='same')(res_output)
y=layers.BatchNormalization()(y)
y=layers.Activation("relu")(y)
y=layers.MaxPool2D((2,2),padding='same')(y)

#Layer2
y=layers.Conv2D(64,(3,3),padding='same')(y)
y=layers.BatchNormalization()(y)
y=layers.Activation("relu")(y)
y=layers.MaxPool2D((2,2),padding='same')(y)

#Layer2
y=layers.Conv2D(64,(3,3),padding='same')(y)
y=layers.BatchNormalization()(y)
y=layers.Activation("relu")(y)
y=layers.MaxPool2D((2,2),padding='same')(y)

"""
#Layer3
z=layers.Conv2D(64,(3,3),padding='same')(y)
z=layers.BatchNormalization()(z)
z=layers.Activation("relu")(z)
#z=layers.MaxPool2D((2,2))(z)

#layer4
z1=layers.Conv2D(64,(3,3),padding='same')(z)
z1=layers.BatchNormalization()(z1)
z1=layers.Activation("relu")(z1)
"""


#layer5
z2=layers.Conv2D(128,(3,3),padding='same')(y)
z2=layers.BatchNormalization()(z2)
z2=layers.Activation("relu")(z2)


"""


branch_a2=layers.Conv2D(128,1,activation='relu',strides=2,padding='same')(res_output1)
branch_b2=layers.Conv2D(128,1,activation='relu',padding='same')(res_output1)
branch_b2=layers.Conv2D(128,3,activation='relu',strides=2,padding='same')(branch_b2)
branch_c2=layers.AveragePooling2D(3,strides=2,padding='same')(res_output1)
branch_c2=layers.Conv2D(128,3,activation='relu',padding='same')(branch_c2)
branch_d2=layers.Conv2D(128,1,activation='relu',padding='same')(res_output1)
branch_d2=layers.Conv2D(128,3,activation='relu',padding='same')(branch_d2)
branch_d2=layers.Conv2D(128,3,activation='relu',strides=2,padding='same')(branch_d2)
res_output2=layers.concatenate([branch_a2,branch_b2,branch_c2,branch_d2], axis=-1)

"""

#Dense Layer
dense_input=layers.Flatten()(res_output)
dense_input1=layers.Dense(512,activation='relu')(dense_input)
output=layers.Dense(10,activation='softmax')(dense_input1)

model=Model(input_tensor,output)
model.summary()

rmsprop=optimizers.RMSprop(learning_rate=0.0001, rho=0.9)

model.compile(optimizer=rmsprop,loss=losses.categorical_crossentropy,metrics=['accuracy'])

history=model.fit(x_train,y_train,
                  epochs=30,
                  batch_size=20,
                  validation_data=(x_val,y_val))


history_dict=history.history

model.save("model_cifar10_inception.hdf5")
print("Model Saved")

import matplotlib.pyplot as plt
loss_values=history_dict['loss']
val_loss_values=history_dict['val_loss']
epochs=range(1,len(loss_values)+1)

#Loss Plot
plt.plot(epochs,loss_values,'bo',label='Training Loss')
plt.plot(epochs,val_loss_values,'b',label='Validation Loss')
plt.title('Traning and Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

#Accuracy Plot
#plt.clf()
acc_values=history_dict['accuracy']
val_acc_values=history_dict['val_accuracy']
plt.plot(epochs,acc_values,'bo',label='Training ACcuy')
plt.plot(epochs,val_acc_values,'b',label='Validation Accuracy')
plt.title('Traning and Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


print("Loading Saved Model")
model_new=load_model("model_cifar10_inception.hdf5")

#Evaluating Model on test Data.
result=model_new.evaluate(test_data,test_labels)
print("Test Results")
print("Loss :",result[0])
print("Accuracy :",result[1])


#------------------------------------------Adding Residual Blocks------------------------
from keras.models import Model
from keras import layers
from keras import Input

input_tensor=Input(shape=(32,32,3))

#Layer 1
x=layers.Conv2D(32,(3,3))(input_tensor)
x=layers.BatchNormalization()(x)
x=layers.Activation("relu")(x)
z=layers.MaxPool2D((2,2))(x)


#Residual Block 
r1=layers.Conv2D(32,3,activation='relu',padding='same')(z)
r1=layers.Conv2D(32,3,activation='relu',padding='same')(r1)
r1=layers.Conv2D(32,3,activation='relu',padding='same')(r1)
z=layers.add([r1,z])

#Residual Block 
r1=layers.Conv2D(32,3,activation='relu',padding='same')(z)
r1=layers.Conv2D(32,3,activation='relu',padding='same')(r1)
r1=layers.Conv2D(32,3,activation='relu',padding='same')(r1)
res_output=layers.add([r1,z])

#Layer2
y=layers.Conv2D(64,(3,3),padding='same')(res_output)
y=layers.BatchNormalization()(y)
y=layers.Activation("relu")(y)
y=layers.MaxPool2D((2,2),padding='same')(y)

#Layer2
y=layers.Conv2D(64,(3,3),padding='same')(y)
y=layers.BatchNormalization()(y)
y=layers.Activation("relu")(y)
y=layers.MaxPool2D((2,2),padding='same')(y)

#Layer2
y=layers.Conv2D(64,(3,3),padding='same')(y)
y=layers.BatchNormalization()(y)
y=layers.Activation("relu")(y)
y=layers.MaxPool2D((2,2),padding='same')(y)


#layer5
z2=layers.Conv2D(128,(3,3),padding='same')(y)
z2=layers.BatchNormalization()(z2)
res_output=layers.Activation("relu")(z2)


#Dense Layer
dense_input=layers.Flatten()(res_output)
dense_input1=layers.Dense(512,activation='relu')(dense_input)
output=layers.Dense(10,activation='softmax')(dense_input1)

model=Model(input_tensor,output)
model.summary()


rmsprop=optimizers.RMSprop(learning_rate=0.0001, rho=0.9)

model.compile(optimizer=rmsprop,loss=losses.categorical_crossentropy,metrics=['accuracy'])

history=model.fit(x_train,y_train,
                  epochs=8,
                  batch_size=20,
                  validation_data=(x_val,y_val))


history_dict=history.history

model.save("model_cifar10_residual.hdf5")
print("Model Saved")

import matplotlib.pyplot as plt
loss_values=history_dict['loss']
val_loss_values=history_dict['val_loss']
epochs=range(1,len(loss_values)+1)

#Loss Plot
plt.plot(epochs,loss_values,'bo',label='Training Loss')
plt.plot(epochs,val_loss_values,'b',label='Validation Loss')
plt.title('Traning and Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

#Accuracy Plot
#plt.clf()
acc_values=history_dict['accuracy']
val_acc_values=history_dict['val_accuracy']
plt.plot(epochs,acc_values,'bo',label='Training ACcuy')
plt.plot(epochs,val_acc_values,'b',label='Validation Accuracy')
plt.title('Traning and Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


print("Loading Saved Model")
model_new=load_model("model_cifar10_residual.hdf5")

#Evaluating Model on test Data.
result=model_new.evaluate(test_data,test_labels)
print("Test Results")
print("Loss :",result[0])
print("Accuracy :",result[1])





