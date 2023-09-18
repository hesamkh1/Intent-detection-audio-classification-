import numpy as np 
import pandas as pd
import librosa as lb
import librosa.display
import csv
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer

def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma

pad2d = lambda a, i: a[:, 0: i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0],i - a.shape[1]))))

test_data_path = "evaluation.csv"
dataset_test = pd.read_csv(test_data_path) 
data_path = "development.csv"
dataset = pd.read_csv(data_path) 

x_test = []
x_test_mfcc = []
lst = []
for c,i in enumerate(x_test_mfcc):
    lst.append(i.shape[1])
    if i.shape[1]==111:
        print(c)
# print(sort(lst))




for k, filename in enumerate(dataset_test['path']): 
    data, sr = lb.load(filename)
    clip = librosa.effects.trim(data, top_db= 10)
    wav_data = clip[0]
    data = feature_normalize(wav_data)
    mfcc = lb.feature.mfcc(y=data)
    x_test_mfcc.append(mfcc)
    print(f"{k}/{len(dataset_test['path'])}")



x_test = np.load("./x_test.npy")
x_test= []  
for i in x_test_mfcc:
    padded_mfcc = pad2d(i,80)
    x_test.append(padded_mfcc)

x_test = np.expand_dims(np.array(x_test), -1) 
# np.save("./x_test.npy", x_test)


model = tf.keras.models.load_model('cp_80.ckpt')
encoder = LabelBinarizer()
encoder.fit(dataset['action'] + dataset['object'])


y_pred = model.predict(x_test)
y_pred_encoded = np.argmax(y_pred, axis=1)

decoder = LabelBinarizer()
y_pred_decoded = decoder.fit_transform(y_pred_encoded)
y_pred = encoder.inverse_transform(y_pred_decoded)


header = ['Id', 'Predicted']
data = []
for i in range(y_pred.shape[0]):
    pred = [i, y_pred[i]]
    data.append(pred)
    

with open('test_82.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)




