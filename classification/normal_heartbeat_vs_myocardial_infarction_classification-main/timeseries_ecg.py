# -*- coding: utf-8 -*-
# 2023 https://github.com/john-fante
# https://www.kaggle.com/banddaniel

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from scipy.io.arff import loadarff

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Dropout, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

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


# plotting random examples
def plot_samples(df):
  pos = df[df.target == 1]
  neg = df[df.target== -1 ]

  idx_pos = random.randint(0, pos.shape[0])
  idx_neg = random.randint(0, neg.shape[0])

  fig,axs = plt.subplots(1,2, figsize=(12,5), sharey=True)
  axs[0].plot(pos.iloc[idx_pos, :-1])
  axs[0].set_title('positive example no: ' + str(idx_pos))

  axs[1].plot(neg.iloc[idx_neg, : -1])
  axs[1].set_title('negative example no: ' + str(idx_neg))

  plt.tight_layout()

  plt.show()

train_data = arff_to_pd('/content/drive/MyDrive/ECG200/ECG200_TRAIN.arff')
test_data = arff_to_pd('/content/drive/MyDrive/ECG200/ECG200_TEST.arff')

print(train_data.shape)
print(test_data.shape)

train_data

X_train = train_data.iloc[:,0:-1]
y_train = train_data.iloc[:,-1]

X_test = test_data.iloc[:,0:-1]
y_test = test_data.iloc[:,-1]



print(X_train.shape)
print(y_train.shape)

# labels encoding (-1,1) -> 0,1
lb = LabelEncoder()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

X_train = np.array(X_train).reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = np.array(X_test).reshape(X_test.shape[0], X_test.shape[1], 1)

print(X_train.shape)
print(X_test.shape)

# shuffling dataset
idx = np.random.permutation(len(X_train))

X_train = X_train[idx]
y_train = y_train[idx]


print(X_train.shape)
print(y_train.shape)

plot_samples(test_data)

inp = Input(shape = X_train.shape[1:])

conv1 = Conv1D(64, 3, padding='same', activation = 'relu')(inp)
conv1 = Dropout(0.1)(conv1)

conv2 = Conv1D(128, 3, padding='same', activation = 'relu')(conv1)
conv2 = BatchNormalization()(conv2)
conv2 = Dropout(0.1)(conv2)


conv3 = Conv1D(128, 3, padding='same', activation = 'relu')(conv2)
conv3 = BatchNormalization()(conv3)
conv3 = Dropout(0.1)(conv3)

pool = GlobalAveragePooling1D()(conv3)

out = Dense(2, activation='softmax')(pool)

model = Model(inputs = inp, outputs= out)
model.compile(optimizer = 'Adam', loss = 'sparse_categorical_crossentropy', metrics =['sparse_categorical_accuracy'] )
model.summary()

callbacks = [ ModelCheckpoint('best.h5', save_best_only = True, monitor = 'val_loss'),
             EarlyStopping(monitor = 'val_loss', patience = 8, verbose = 1 ),
              ReduceLROnPlateau(monitor = 'val_loss', min_lr = 0.0001, factor = 0.5, patience = 3) ]


hist = model.fit(X_train, y_train, batch_size = 8, epochs = 50 , callbacks = callbacks,
                 validation_split = 0.2 , verbose = 1)

plt.plot(hist.history['sparse_categorical_accuracy'])
plt.plot(hist.history['val_sparse_categorical_accuracy'])
plt.ylabel('acc', fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "val"], loc="best")
plt.show()

model = tf.keras.models.load_model('/content/best.h5')
model.evaluate(X_test,y_test)

pred = model.predict(X_test)

y_pred = []
for i in range(len(pred)):
  y_pred.append( np.argmax(pred[i]))

from sklearn.metrics import classification_report, confusion_matrix

rep = classification_report(y_test, y_pred)

print(rep)

import seaborn as sns

cf = confusion_matrix(y_test, y_pred)
sns.heatmap(cf, annot =True)