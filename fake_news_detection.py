

#Created by: Pintu (181CO139) and Akshay Dhayal (181CO105)
#23/03/2021
#Obejctive: To build the hybrid cnn-rnn model to detect the fake news.

#NOTE:
# dataset "train.csv" should be in same directory as of program.
# http://nlp.stanford.edu/data/glove.6B.zip download the GloVe zip folder.
# extract the zip folder in same directory as of program.
#"glove.6B.100d.txt" this file must be in same directory as of program.
#In short, dataset, code and glove should be in same folder
#to run this program successfully, as it is.

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

###load dataset.
dataset=pd.read_csv('train.csv')
#remove the missing data.
dataset=dataset.dropna()
n=int(0.8 * len(dataset))
x=dataset['text'].values[:n]
y=dataset['label'].values[:n]
x_valid=dataset['text'].values[n:]
y_valid=dataset['label'].values[n:]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state=1000)
print(type(x), type(y))
print(type(x_valid), type(y_valid))
print(type(x_train), type(x_test))
print(type(y_train), type(y_test))
#those ending with .values are ndarray.
#train_test_split output has same type as of input.

###Creating vocabulary.
tokenizer = Tokenizer(num_words=150000)
tokenizer.fit_on_texts(x_train)
#number of unique words in dict.
print("Number of unique words in dictionary=",len(tokenizer.word_index))
x_train = tokenizer.texts_to_sequences(x_train)
x_valid = tokenizer.texts_to_sequences(x_valid)
x_test = tokenizer.texts_to_sequences(x_test)
# Adding 1 because of  reserved 0 index
vocab_size = len(tokenizer.word_index) + 1
#size of random text in training set.
print("Length of random text=",len(x_train[3]),len(x_train[13]))
maxlen = 400
x_train = pad_sequences(x_train, padding='post', maxlen=maxlen)
x_valid = pad_sequences(x_valid, padding='post', maxlen=maxlen)
x_test = pad_sequences(x_test, padding='post', maxlen=maxlen)

###confirm that texts are converted in vector form.
print(type(x_train))
for row in x_train:
  print(row)
  break

###Create embedding matrix.
def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1
    # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]
    return embedding_matrix
embedding_dim = 100
embedding_matrix = create_embedding_matrix('glove.6B.100d.txt', tokenizer.word_index, embedding_dim)

##Build the hybrid model.
embedding_dim = 100
model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=True))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.MaxPooling1D(2))
model.add(layers.LSTM(32))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
##RUN the model, with epochs=10 & batch_size=64
history = model.fit(x_train, y_train,
                    epochs=10,
                    validation_data=(x_valid, y_valid),
                    batch_size=64)
model.summary()
val_loss, val_acc=model.evaluate(x_test, y_test)

##Prediction and performance
def performance(x_test, y_test):
  y_pred=model.predict(x_test)
  # print(y_pred[0])
  y_pred=[1 if x>=0.5 else 0 for x in y_pred]
  # print(y_pred[0])
  cm=confusion_matrix(y_test, y_pred)
  cr=classification_report(y_valid, y_valid)
  print("Confusion matrix=\n",cm)
  print("Classification report=\n",cr)
  loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
  print("Testing Accuracy:  {:.4f}".format(accuracy))  # actual accuracy
  print("Testing Loss: {:.4f}".format(loss)) #testing loss
#call the performance measure function
performance(x_test, y_test)

##Print the loss and accuracy graph.
import matplotlib.pyplot as plt
plt.style.use('ggplot')
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
#call the function using history
plot_history(history)

##2nd RUN.
from keras.backend import clear_session
clear_session()
embedding_dim = 50
#build embedding matrix.
embedding_matrix = create_embedding_matrix('/content/glove.6B.50d.txt',tokenizer.word_index,embedding_dim)
#buidl the hybrid cnn-rnn model.
model_1 = Sequential()
model_1.add(layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=True))
model_1.add(layers.Conv1D(128, 5, activation='relu'))
model_1.add(layers.MaxPooling1D(2))
model_1.add(layers.LSTM(32))
model_1.add(layers.Dense(1, activation='sigmoid'))
model_1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#train the model.
history_1 = model_1.fit(x_train, y_train,
                    epochs=10,
                    validation_data=(x_valid, y_valid),
                    batch_size=64)
#confusion matrix and classification report.
performance(x_test, y_test)
#call the function using history
plot_history(history_1)


##$$$$$$$$$%%%%%%%%%%%^^^^^^^^^&&&&&&&&&&&*************(((((((((((((("FINISH"))))))))))))))@@@@@@@@@@@####################$$$$$$$$$$$$%%%%%%%%%%#####