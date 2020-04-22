
"""
Name: Sourav Yadav
Assignment 4
AID: A20450418
Spring 2020
Deep Learning

"""
from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1.0/255)
val_datagen=ImageDataGenerator(rescale=1.0/255)

train_generator=train_datagen.flow_from_directory(
    "C://Users//soura//OneDrive//Desktop//Deep Learning//HW4//PetImages//Train",
    target_size=(150,150),
    batch_size=20,
    class_mode='binary')

validation_generator=val_datagen.flow_from_directory(
    "C://Users//soura//OneDrive//Desktop//Deep Learning//HW4//PetImages//Validation",
    target_size=(150,150),
    batch_size=20,
    class_mode='binary')



#Design of the Basic Multiclassifier Model
#----------------------------------------------------------------------------
from keras import layers
from keras import models

model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),input_shape=(150,150,3)))
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(64,(3,3),input_shape=(150,150,3)))
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(128,(3,3),input_shape=(150,150,3)))
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(128,(3,3),input_shape=(150,150,3)))
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
model.add(layers.MaxPool2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.summary()



from keras import optimizers
model.compile(loss='binary_crossentropy', 
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])


history=model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=20)


model.save("cats_and_dogs_small_1.h5")


import matplotlib.pyplot as plt
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(1,len(acc)+1)


#Plot accuracy
plt.plot(epochs,acc,'bo',label='Traning acc')
plt.plot(epochs,val_acc,'b',label='Validation Acc')
plt.title('Training and Validation Accuracy')
plt.legend()

#Plot Loss
plt.figure()
plt.plot(epochs,loss,'bo',label='Traning loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()

#Evaluating on Test data
test_datagen=ImageDataGenerator(rescale=1.0/255)
test_generator=test_datagen.flow_from_directory(
    "C://Users//soura//OneDrive//Desktop//Deep Learning//HW4//PetImages//Test",
    target_size=(150,150),
    batch_size=20,
    class_mode='binary')

#Loading the Model
from keras.models import load_model
model_new=load_model('cats_and_dogs_small_1.h5')

history=model_new.evaluate_generator(test_generator)
print('Loss: {}'.format(history[0]))
print('Accuracy: {}'.format(history[1]))



#Visualing Convloution layers 
#---------------------------------------------------------------------
img_path='C://Users//soura//OneDrive//Desktop//Deep Learning//HW4//PetImages//Train//Cat//4.jpg'
from keras.preprocessing import image
import numpy as np

img=image.load_img(img_path,target_size=(150,150))
img_tensor=image.img_to_array(img)
img_tensor=np.expand_dims(img_tensor,axis=0)
img_tensor/=255
print(img_tensor.shape)

#Show Image
import matplotlib.pyplot as plt
plt.imshow(img_tensor[0])
plt.show()


#Load existing model
from keras.models import load_model
model_vis_con=load_model('cats_and_dogs_small_1.h5')

from keras import models
layer_outputs=[layer.output for layer in model_vis_con.layers[:8]]

#Create a new model
activation_model=models.Model(inputs=model_vis_con.input,outputs=layer_outputs)

#Run the model on the loaded image
activations=activation_model.predict(img_tensor)



layer_names=[]
for layer in model_vis_con.layers[:8]:
    layer_names.append(layer.name)

image_per_row=16

for layer_name,layer_activation in zip(layer_names,activations):
    #Number of Channels in the Current layer
    n_features=layer_activation.shape[-1]
    
    #Get Image Size
    size=layer_activation.shape[1]
    
    
    #Compute the number of rows in resulting image grid
    
    n_cols=n_features//image_per_row
    
    #Allocate display grid as one big image
    display_grid=np.zeros((size*n_cols,image_per_row*size))
    
    #tile each image into activation grid
    for col in range(n_cols):
        for row in range(image_per_row):
            #retirive i-th channel
            channel_image=layer_activation[0,:,:,col*image_per_row+row]
            #Normalize chnnael to [0,255]
            channel_image-=channel_image.mean()
            channel_image/=channel_image.std()
            channel_image*=64
            channel_image+=128
            channel_image=np.clip(channel_image, 0, 255).astype('uint0')
            display_grid[col*size:(col+1)*size,row*size:(row+1)*size]=channel_image
            
    #Plot 
    scale=1./size
    plt.figure(figsize=(scale*display_grid.shape[1],scale*display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto',cmap='viridis')



first_layer_activation=activations[0]
print(first_layer_activation.shape)


"""
import matplotlib.pyplot as plt
plt.matshow(first_layer_activation[0,:,:,3],cmap='viridis')
plt.matshow(first_layer_activation[0,:,:,4],cmap='viridis')
"""

for i in range(32):
    plt.matshow(first_layer_activation[0,:,:,i],cmap='viridis')



#Visualising Filetrs 
#------------------------------------Part F---------------------------------
from keras import backend as K

model=load_model('cats_and_dogs_small_1.h5')  

def deprocess_image(x):
    x-=x.mean()
    x/=(x.std()+1e-5)
    x*=0.1
    x+=0.5
    x=np.clip(x,0,1)
    x*=255
    #Clip to [0,255] and convert to unsigned byte channels
    x=np.clip(x,0,255).astype('uint8')
    return x

def generate_pattern(layer_name,filter_index,size=150):
    #Define Output and Loss
    layer_output=model.get_layer(layer_name).output
    loss=K.mean(layer_output[:,:,:,filter_index])
    #Compute the gradient of the input with respect to loss
    grads=K.gradients(loss,model.input)[0]
    grads/=((K.sqrt(K.mean(K.square(grads))))+1e-5)
    iterate=K.function([model.input], [loss,grads])
    input_img_data=np.random.random((1,size,size,3))*20+128
    
    step=1
    for i in range(100):
        loss_value,grads_value=iterate([input_img_data])
        input_img_data+=grads_value*step
        
    img=input_img_data[0]
    return deprocess_image(img)


#layer_names=[]
#for layer in model.layers:
    #print(layer.name)
    #layer_names.append(layer.name)

layer_names=[]
conv_ly1=model.layers[0].name
layer_names.append(conv_ly1)
conv_ly2=model.layers[4].name
layer_names.append(conv_ly2)
conv_ly3=model.layers[8].name
layer_names.append(conv_ly3)


#layer_outputs=[layer.output for layer in model.layers[:8]]

k=0
#for layer_name,layer in zip(layer_names,model.layers[i]):
for layer_name in layer_names:
    layer=model.layers[k]
    layer_output=layer.output
    print(layer_output.shape)
    size=int(layer_output.shape[1])
    row=8
    col=int(layer_output.shape[3]//row)
    results=np.zeros((row*size,col*size,3)).astype('uint0')
    
    for i in range(row):
        for j in range(col):
            filter_img=generate_pattern(layer_name,i+(j*8),size=size)        
            horizontal_start=i*size
            horizontal_end=horizontal_start+size
            vertical_start=j*size
            vertical_end=vertical_start+size
            results[horizontal_start:horizontal_end,vertical_start:vertical_end,:]=filter_img[:,:,:]
    plt.figure(figsize=(20,20))
    plt.title(layer_name)
    plt.imshow(results)
    k+=4



#-----------------------------------------------Part F and G------------------------------------------
from keras import optimizers
from keras import models
from keras import layers
#Before Tuning
#conv_base.trainable=False

from keras.applications import VGG16
conv_base=VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(150,150,3))

conv_base.summary()




for layer in conv_base.layers:
#    layer.trainable=False
    print(layer.name,layer.trainable)


model=models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())

model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

conv_base.trainable=True

set_trainable=False
for layer in conv_base.layers:
    if layer.name=='block5_conv1': #or layer.name=='block5_conv2' or layer.name=='block5_conv3' :
        set_trainable=True
    if set_trainable:
        layer.trainable=True
    else :
        layer.trainable=False

rmsprop=optimizers.RMSprop(learning_rate=2e-5, rho=0.9)
model.compile(optimizer=rmsprop,loss='binary_crossentropy',metrics=['accuracy'])

len(model.trainable_weights)

history=model.fit(train_generator,
                  steps_per_epoch=100,
                  epochs=10,
                  #batch_size=20,
                  validation_data=validation_generator,
                  validation_steps=50)



history_dict=history.history

#model.save("model_VGG16_wo_tuning.hdf5")
model.save("model_VGG16_after_tuning.hdf5")
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
plt.plot(epochs,acc_values,'bo',label='Training Accuracy')
plt.plot(epochs,val_acc_values,'b',label='Validation Accuracy')
plt.title('Traning and Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()



print("Loading Saved Model")
#model_new=load_model("model_VGG16_wo_tuning.hdf5")
model_new=load_model("model_VGG16_after_tuning.hdf5")


#Evaluating Model on test Data.
result=model_new.evaluate_generator(test_generator)
print("Test Results")
print("Loss :",result[0])
print("Accuracy :",result[1])


#----------------------------------------Data Augmentation----------------------------------
from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1.0/255,
                                 rotation_range=40,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 fill_mode='nearest'
                                 )
val_datagen=ImageDataGenerator(rescale=1.0/255)
test_datagen=ImageDataGenerator(rescale=1.0/255)

train_generator=train_datagen.flow_from_directory(
    "C://Users//soura//OneDrive//Desktop//Deep Learning//HW4//PetImages//Train",
    target_size=(150,150),
    batch_size=20,
    class_mode='binary')

validation_generator=val_datagen.flow_from_directory(
    "C://Users//soura//OneDrive//Desktop//Deep Learning//HW4//PetImages//Validation",
    target_size=(150,150),
    batch_size=20,
    class_mode='binary')


test_generator=test_datagen.flow_from_directory(
    "C://Users//soura//OneDrive//Desktop//Deep Learning//HW4//PetImages//Test",
    target_size=(150,150),
    batch_size=20,
    class_mode='binary')


from keras import optimizers
from keras import models
from keras import layers
#Before Tuning
#conv_base.trainable=False

from keras.applications import VGG16
conv_base=VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(150,150,3))

conv_base.summary()

#After Tuning


#for layer in conv_base.layers:
#    layer.trainable=False
#    print(layer.name,layer.trainable)

"""
set_trainable=False
for layer in conv_base.layers:
    if layer.name=='block5_conv1': #or layer.name=='block5_conv2' or layer.name=='block5_conv3' :
        set_trainable=True
    if set_trainable:
        layer.trainable=True
    else :
        layer.trainable=False
"""


"""
for layer in conv_base.layers:
    layer.trainable=False
    print(layer.name,layer.trainable)
    
"""

model=models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())

model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

conv_base.trainable=False

rmsprop=optimizers.RMSprop(learning_rate=2e-5, rho=0.9)
model.compile(optimizer=rmsprop,loss='binary_crossentropy',metrics=['accuracy'])

len(model.trainable_weights)

history=model.fit(train_generator,
                  steps_per_epoch=100,
                  epochs=30,
                  #batch_size=20,
                  validation_data=validation_generator,
                  validation_steps=50)



history_dict=history.history

#model.save("model_VGG16_wo_tuning.hdf5")
model.save("model_VGG16_data_aug.hdf5")
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
plt.plot(epochs,acc_values,'bo',label='Training Accuracy')
plt.plot(epochs,val_acc_values,'b',label='Validation Accuracy')
plt.title('Traning and Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()



print("Loading Saved Model")
#model_new=load_model("model_VGG16_wo_tuning.hdf5")
model_new=load_model("model_VGG16_data_aug.hdf5")


#Evaluating Model on test Data.
result=model_new.evaluate_generator(test_generator)
print("Test Results")
print("Loss :",result[0])
print("Accuracy :",result[1])