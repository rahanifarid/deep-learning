# deep-learning
# voici le code pour un réseau de neurone simple
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
import pandas as pd
# importer le data set 
fashion_mnist = tf.keras.datasets.fashion_mnist
(images,targets),(images_test,targets_test) = fashion_mnist.load_data()
images = images[:10000]
targets = targets[:10000]
# afficher les dimensions des images 
print(images.shape)
print(targets.shape)
# afficher une image en pixels dans un tableau
print(images[0])
# informer sur le nombre de classes qu'ils existent ici cest 9 classes
print(targets[0])
# nommer les classes
targets_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", 
                 "Shirt", "Sneaker", "Bag", "Ankle boot"]
# affucher l'image 10 en gris en utilisant imshow
plt.imshow(images[10], cmap="binary")
plt.title(targets_names[[taregets[10]]])
# transfomer notre image à un vecteur à une seule dimension
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=[28,28]))
print("shape of the image :",images[0:1].shape)
model_output = model.predict(images[0:1])
print("shape of the image :",model_output.shape)
# 256 neuronnes pour la prmiere couche, et 128 neuronnes pour la couche intermediaire
# et 10 neuronnes pour la couche de sortie
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model_output = model.predict(images[0:1])
print(model_output, targets[0:1])
model.summary()
# compiler le modele
model.compile(loss="sparse_categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])
# entrainer notre modele
history = model.fit(images,targets,epochs=10, validation_split=0.2)
# afficher les differentes courbes
loss_curve = history.history["loss"]
acc_curve = history.history["acc"]
plt.plot("loss_curve")
plt.title("loss")
plt.show()
plt.plot("acc_curve")
plt.title("accuracy")
plt.show()

