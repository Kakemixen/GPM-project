import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
from keras.utils import plot_model

# Read data
df  = pd.read_csv("data/ANN/ANN_normalized.csv",   sep=",", header=0)
df_train, df_test = train_test_split(df, test_size=0.1) #500 rows for validating

df_train_target = df_train["Winner"]
df_train_features = df_train.drop("Winner", axis=1)
df_test_target = df_test["Winner"]
df_test_features = df_test.drop("Winner", axis=1)

# Create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dense(15, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])
model.compile(optimizer='Adam',
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'])

# Train model
t1 = time.clock()
history = model.fit(df_train_features.values, df_train_target.values, epochs=10, batch_size=50)
t2 = time.clock()

# Evaluate result
test_loss, test_acc = model.evaluate(df_test_features.values, df_test_target.values)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)
print('Training time:', t2-t1, "seconds")

plot_model(model, to_file='model.png')

# Plot accuracy
plt.figure(1)
plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.xlim(0,9)
plt.legend(['training data'], loc='upper left') 


# Plot loss
plt.figure(2)
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.xlim(0,9)
plt.legend(['training data'], loc='upper left')
plt.show()