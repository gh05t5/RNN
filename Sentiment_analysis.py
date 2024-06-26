#imports

from keras.datasets import imdb
from keras.preprocessing import sequence
import keras
import tensorflow as tf 
import os 
import numpy as np 


VOCAB_SIZE = 88584

MAXLEN = 250
BATCH_SIZE = 64

(train_data, train_lables), (test_data, test_labels) = imdb.load_data(num_words=VOCAB_SIZE)

#Let's look at one review 

##print(train_data[1])

train_data = sequence.pad_sequences(train_data, MAXLEN)
test_data = sequence.pad_sequences(test_data, MAXLEN)

##print(train_data[1])

#Creating the model 

model = tf.keras.Sequential(
    [tf.keras.layers.Embedding(VOCAB_SIZE,32, input_length=MAXLEN),
     tf.keras.layers.LSTM(32),
     tf.keras.layers.Dense(1, activation="sigmoid")
     
])

model.build(input_shape=(None, MAXLEN))

print(model.summary())


#Training

model.compile(loss="binary_crossentropy",
              optimizer="rmsprop",
              metrics=['accuracy'])

history = model.fit(train_data, train_lables, epochs=10, validation_split=0.2)

results = model.evaluate(test_data, test_labels)

print(results)