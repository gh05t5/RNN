vocab = {} #maps word to intiger 
word_encoding = 1
def bag_of_words (text):
    global word_encoding
    
    words = text.lower().split(" ") #create a list of all the words in our text
    bag = {} #this will store all the encodings and their frequency
    
    
    for word in words:
        if word in vocab:
            encoding = vocab[word] #gets the encoding from vocab 
        else:
            vocab[word] = word_encoding
            encoding = word_encoding
            word_encoding += 1
            
        if encoding in bag:
            bag[encoding] += 1
        else:
            bag[encoding] = 1
            
    return bag 

text = 'this is a test to see if this test will work is is test a a'

bag = bag_of_words(text)

print(bag)
print(vocab)
        


            