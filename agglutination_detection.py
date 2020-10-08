from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from skimage import color
from skimage import io


data = list()

for i in range(1,51):   #esto es porque en mi carpeta tenía 50 imagenes que yo había recortado y pasado a escala de grises (todas eran de 60x60 pixeles)
    file_name = str(i)    #a cada imagen en la carpeta le puse el nombre de 1.jpg, 2.jpg, 3.jpg, y asi....
    
    image = Image.open(file_name + '.jpg')
    matrix = np.array(image)    #convierto la imagen a una matriz 
    data.append(matrix)  #inserto todas las matrices de manera consecutiva a la lista data
    
X = np.array(data)    #this contains the full information: (36,60,60)

y = [1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,0,0,0,1,1,1,0,1,1,1,0,1,1] #by hand

X = np.vstack((X,X,X))   #repetí el dataset 3 veces para aumentar el número de datos

y = np.hstack((y,y,y))


#Creating the object: model

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu),   #esta es la parte que no sabemos muy bien el porqué de las cosas
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(2, activation=tf.nn.softmax)])

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

class myCallback(tf.keras.callbacks.Callback):     #tampoco entendemos muy bien esto
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.9):
      print('\nAccuracy is high, stop training!')
      self.model.stop_training = True

callback = myCallback()

#Fitting the model with the data

model.fit(X, y, epochs=30, callbacks = [callback])


#reading new data to test the model
test_list = list()

for i in range (1, 11):    #acá cargaba las imágenes con las que iba a hacer el test. las imágenes se llamaron t1.jpg, t2.jpg, t3.jpg, y así
    im_name = str(i)
    test_im = Image.open('t'+im_name + '.jpg')
    matrix_test = np.array(test_im)
    test_list.append(matrix_test)

test_dataset = np.array(test_list)


#making the predictions

predictions = model.predict_classes(test_dataset)

print(predictions)

model.evaluate(test_dataset, np.array([1,0,0,0,0,0,1,1,1,0]))

print('\n')

for i,patient in enumerate(predictions):
    if patient == 1:
        print('Patient No. '+ str(i+1) + ' is POSITIVE for Covid-19 antibodies'+'\n')
    else:
        print('Patient No. '+ str(i+1) + ' is NEGATIVE for Covid-19 antibodies'+'\n')


        
