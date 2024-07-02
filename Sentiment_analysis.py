#imports

from keras.datasets import imdb
from nltk.tokenize import word_tokenize
from keras.preprocessing import sequence
import nltk
import keras
import tensorflow as tf 
import os 
import numpy as np 

#Download nltk tokenizer data 

nltk.download('punkt')

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

##print(model.summary())


#Training

model.compile(loss="binary_crossentropy",
              optimizer="rmsprop",
              metrics=['accuracy'])

history = model.fit(train_data, train_lables, epochs=10, validation_split=0.2)

 

results = model.evaluate(test_data, test_labels)

##print(results)

#Saving the model to disk 

model.save('sentiment_model.h5') ##Once the model is trined and saved, comment from line 19 to 62


#Loading the model, uncomment line once the model has been trained.

##model = tf.keras.models.load_model('sentiment_model.h5')

#Encoding function for new reviews 


word_index = imdb.get_word_index()

def encode_text (text, word_index, maxlen=MAXLEN):
    tokens = word_tokenize(text.lower())
    sequences = [word_index.get(word, 0) for word in tokens]
    padded_sequences = sequence.pad_sequences([sequences], maxlen=maxlen)
    return padded_sequences[0]

text = "That movie was just amazing, so amazing"
encoded = encode_text(text, word_index)
print(encoded)


#Reverse word index function 

reverse_word_index = {value: key for (key, value) in word_index.items()}

def decode_integers(integers):
    PAD = 0 
    text = ""
    for num in integers:
        if num != PAD:
            text +=reverse_word_index[num] + " "
    
    return text[:-1]        
            
print(decode_integers(encoded))

#Here we will use two reviews to get the sentiment analysis 

def predict(text, word_index):
    encoded_text = encode_text(text, word_index)
    pred = np.zeros((1,250)) #We use this shape becuse our model expect soemthing, 250
    pred[0] = encoded_text #Here we insert our encoded string into the shape defined before
    result = model.predict(pred)
    print(result[0])
    
positive_review = "That moview was so awesome! I really loved it and would watch it again because it was amazingly great"
predict(positive_review, word_index)

negative_review = "That movie socked. I hated it and wouldn't watch it again. Was one of the worst things I've ever watched"
predict(negative_review, word_index)


    
    