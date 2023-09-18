

import numpy as np 
import pandas as pd
import librosa as lb
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
# import pickle

#z score normalization
def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma

#padding
pad2d = lambda a, i: a[:, 0: i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0],i - a.shape[1]))))

#Read csv file
data_path = "development.csv"
dataset = pd.read_csv(data_path) 
# print(dataset["action"])




# Read audios and extract feature from them
x_train_mfcc = []
for k, filename in enumerate(dataset['path']): 
    data, sr = lb.load(filename)      
    clip = librosa.effects.trim(data, top_db= 10)
    wav_data = clip[0]
    data = feature_normalize(wav_data)
    mfcc = lb.feature.mfcc(y=data)
    x_train_mfcc.append(mfcc)
    print(f"{k}/{len(dataset['path'])}")
    




  

#final shape of X
lst = []
indexes = []
for c,i in enumerate(x_train_mfcc):
    if i.shape[1]<=118:
        lst.append(x_train_mfcc[c])
    else:
        indexes.append(c)

        

x_train = []  
for i in lst:
    padded_mfcc = pad2d(i,80)
    x_train.append(padded_mfcc)
    
x_train = np.expand_dims(np.array(x_train), -1)



#label on Y
encoder = LabelBinarizer() 
y_train = np.array(encoder.fit_transform(dataset['action'] + dataset['object']))
y_train = np.delete(y_train, indexes, 0)


#split to validation and train
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.20, random_state=42)


#Save our splitted data
np.save("./x_train.npy", x_train)
np.save("./y_train.npy", y_train)
np.save("./x_valid.npy", x_valid)
np.save("./y_valid.npy", y_valid)



#Create Classifier Model
ip = tf.keras.layers.Input(shape=x_train[0].shape)
m = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu')(ip)
m = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(m)
m =  tf.keras.layers.BatchNormalization()(m)
m = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu')(m)
m = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(m)
m =  tf.keras.layers.BatchNormalization()(m)
m = tf.keras.layers.Dropout(0.2)(m)
m = tf.keras.layers.Flatten()(m)
m = tf.keras.layers.Dense(512, activation='relu')(m)
m = tf.keras.layers.Dense(256, activation='relu')(m)
m = tf.keras.layers.Dense(128, activation='relu')(m)
m = tf.keras.layers.Dense(64, activation='relu')(m)
op = tf.keras.layers.Dense(7, activation='softmax')(m)
model = tf.keras.Model(inputs=ip, outputs=op)
checkpoint_path = "cp_80.ckpt"
# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, mode='max', monitor='accuracy', verbose=1)


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(x_train,
          y_train,
          epochs=100,
          batch_size=32,
          validation_data=(x_valid, y_valid),
          callbacks=[cp_callback])



#showing accuracy of train and validation
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('tessstttyyy.svg', dpi=100)
plt.show()



