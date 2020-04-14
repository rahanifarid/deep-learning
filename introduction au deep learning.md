# deep-leraning
# voici mon code si vous voulez avoir une bonne formation en deep learning
import tensorflow as tf
import matplotlib.pyplot as plt
import  tensorflow.keras as keras
import numpy as np
tf.__version__
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
plt.imshow(x_train[0],cmap=plt.cm.binary)
# normalizer nos données 
x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)
# afficher l'image apres la normalisation
plt.imshow(x_train[0],cmap=plt.cm.binary)
# les choses vont dérouler dans l'ordre sequentiel(direct sans retour en arrière)
model = tf.keras.models.Sequential()
# inserer des couches
# une couche d'entrée plate et non pas multidimensionnelle
model.add(tf.keras.layers.Flatten())
# utiliser des couches denses dans notre réseau neuronal
# une couche dense: entierement connecté, chaque noeud est connecté à son précedent
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# la derniere couche a 10 noeuds et utilise la focntion softmax par ce qu'on cherche une distribution
# de probablilité dans laquelle les options de prédictions possibles
# compiler notre modele
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# entrainer notre modele
model.fit(x_train,y_train,epochs=3)
# utiliser des données qu'on a pas utiliser lors de l'entrainement de notre modele
val_loss,val_acc = model.evaluate(x_test,y_test)
print('val_loss:',val_loss)
print('val_acc:',val_acc)
# enregistrer notre modele
model.save('epic_num_reader.model')
# recharger le modele
new_model = tf.keras.models.load_model('epic_num_reader.model')
# faire des prédictions
predictions = new_model.predict(x_test)
print(predictions)
# obtenir un nombre réel
print(np.argmax(predictions[0]))
plt.imshow(x_test[0],cmap=plt.cm.binary)
plt.show()
