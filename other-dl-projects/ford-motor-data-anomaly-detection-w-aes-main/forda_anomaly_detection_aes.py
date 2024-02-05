# -*- coding: utf-8 -*-
"""forda anomaly detection aes.ipynb

# Ford Motor Data Anomaly Detection with AutoEncoder

*The main goal of this project is prediction anomalies in the FORD A motor data using an autoencoder. It has a 500 hours univariate time series data obtained from sensors.*<br>


# Results

|                        	| Accuracy 	| Precision 	| Recall  	|
|------------------------	|----------	|-----------	|---------	|
| FORD A Validation Data 	| 68.65 %  	| 0.61634   	| 0.97199 	|
| FORD A Test Data       	| 66.14 %  	| 0.59284   	| 0.95931 	|

<br>

# References

- Wichard, Joerg. (2009). Classification of Ford Motor Data(http://www.j-wichard.de/publications/FordPaper.pdf)

- https://www.timeseriesclassification.com
"""

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.io.arff import loadarff

from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Convolution1DTranspose, Dropout, BatchNormalization, Dense,ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split

# reading .arff file and converting dataframe
def arff_to_pd(path):
  data = loadarff(path)
  raw_data, meta_data = data

  cols = []
  for col_name in meta_data:
    cols.append(col_name)

  data2d = np.zeros([ raw_data.shape[0], len(cols) ])

  for row_number in range(raw_data.shape[0]):
    for col_number in range(len(cols)):
      data2d[row_number][col_number] = raw_data[row_number][col_number]

  df = pd.DataFrame(data2d, columns = cols)

  return df

data = arff_to_pd('/content/drive/MyDrive/Colab Notebooks/GeneralDB/FordA/FordA_TRAIN.arff')
data.head()

y = data.iloc[:,-1]
X = data.iloc[:, :-1]

# Label encoding -1,1 to 0,1
# 0s mean an anomaly in motor, 1s normal situation

LB = LabelEncoder()
y = LB.fit_transform(y)

# Train test splitting

X_train, X_val, y_train, y_val = train_test_split(X,y , test_size = 0.2, random_state =5)

X_train = tf.cast(X_train, tf.float32)
X_val = tf.cast(X_val, tf.float32)

# Seperate data into normal and anormal

train_labels = y_train.astype(bool)
normal_X_train = X_train[train_labels]
anormal_X_train = X_train[~train_labels]

print("Normal train shape : ",normal_X_train.shape)
print("Anormal train shape : ",anormal_X_train.shape)


val_labels = y_val.astype(bool)
normal_X_val = X_val[val_labels]
anormal_X_val = X_val[~val_labels]

print("Normal validation shape : ",normal_X_val.shape)
print("Anormal validation shape : ",anormal_X_val.shape)

inp = Input(shape = normal_X_train.shape[1:])

D1 = Dense(128, activation = 'relu' )(inp)
D1 = BatchNormalization()(D1)

D2 = Dense(64, activation = 'relu' )(D1)
D2 = BatchNormalization()(D2)

D3 = Dense(32, activation = 'relu' )(D2)
D3 = BatchNormalization()(D3)

D4 = Dense(64, activation = 'relu' )(D3)
D4 = BatchNormalization()(D3)

D5 = Dense(128, activation = 'relu' )(D4)
D5 = BatchNormalization()(D5)

out = Dense(500, activation = 'sigmoid')(D5)

model = Model(inputs = inp, outputs=out)
model.compile( optimizer = 'adam', loss = 'mae', metrics=['mae'])
model.summary()

my_callback = [EarlyStopping(patience = 30),
               ReduceLROnPlateau(factor=0.5, patience=15)]


hist = model.fit(normal_X_train, normal_X_train, epochs = 350,
                 validation_data = (normal_X_val,normal_X_val)
                 ,batch_size = 32, callbacks =[my_callback], shuffle=True)

train_reconstruct = model.predict(normal_X_train)
train_reconstruct_loss = tf.keras.losses.mae(train_reconstruct, normal_X_train)


threshold = np.mean(train_reconstruct_loss) + 4*np.std(train_reconstruct_loss)
print(threshold)

plt.hist(train_reconstruct_loss, bins = 50)
plt.xlabel("train normal loss")
plt.ylabel("samples")
plt.axvline(threshold, color ='black', linestyle='dotted')

test_anormal_reconstruct = model.predict(anormal_X_val)
test_anormal_reconstruct_loss = tf.keras.losses.mae(test_anormal_reconstruct, anormal_X_val )


plt.hist(test_anormal_reconstruct_loss, bins = 50)
plt.xlabel("test loss")
plt.ylabel("samples")
plt.axvline(threshold, color = 'black', linestyle = 'dotted')

prediction_val = model.predict(X_val)
val_full_loss = tf.keras.losses.mae(prediction_val, X_val)

result = tf.math.less(val_full_loss, threshold)

acc = metrics.accuracy_score(y_val, result)*100
precision = metrics.precision_score(y_val, result)
recall = metrics.recall_score(y_val, result)

print('Accuracy :{0:.2f} %'.format(acc))
print('Precision :{0:.5f}'.format(precision))
print('Recall :{0:.5f}'.format(recall))

y_test_ = pd.DataFrame(y_val)
y_test_.value_counts()

"""Test data"""

data_test = arff_to_pd('/content/drive/MyDrive/Colab Notebooks/GeneralDB/FordA/FordA_TEST.arff')
data_test.head()

X_test = data_test.iloc[:,0:-1]
y_test = data_test.iloc[:,-1]

y_test = LB.fit_transform(y_test)

prediction_test = model.predict(X_test)
test_full_loss = tf.keras.losses.mae(prediction_test, X_test)

result = tf.math.less(test_full_loss, threshold)

acc = metrics.accuracy_score(y_test, result)*100
precision = metrics.precision_score(y_test, result)
recall = metrics.recall_score(y_test, result)

print('Accuracy :{0:.2f} %'.format(acc))
print('Precision :{0:.5f}'.format(precision))
print('Recall :{0:.5f}'.format(recall))

cm = metrics.confusion_matrix(y_test,result )

disp = metrics.ConfusionMatrixDisplay(cm, display_labels=['anormal','normal'])
disp.plot()