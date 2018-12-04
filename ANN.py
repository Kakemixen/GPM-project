import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt

df  = pd.read_csv("data/ANN/ANN_normalized.csv",   sep=",", header=0)
df_train, df_test = train_test_split(df, test_size=0.1) #500 rows for validating

df_train_target = df_train["Winner"]
df_train_features = df_train.drop("Winner", axis=1)
df_test_target = df_test["Winner"]
df_test_features = df_test.drop("Winner", axis=1)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(25, activation=tf.nn.relu),
    tf.keras.layers.Dense(15, activation=tf.nn.relu),
    tf.keras.layers.Dense(5, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'])

t1 = time.clock()
history = model.fit(df_train_features.values, df_train_target.values, validation_split=0.2, epochs=5)
t2 = time.clock()

test_loss, test_acc = model.evaluate(df_test_features.values, df_test_target.values)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)
print('Training time:', t2-t1, "seconds")


# Plot the accuracy
plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left') #


# Plot the loss
plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()