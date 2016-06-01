import ntpath
import random
import os
import re
import numpy as np
from datetime import datetime
from os import listdir
from os.path import isfile, join
np.random.seed(1445)  # for reproducibility

from keras.preprocessing import sequence, text
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras import optimizers
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from RNN_LM_Utils import SourceCodeUtils



max_features = 1000
maxlen = 100
batch_size = 32


work_dir = "C:\\Datasets\\Authors\\Java\\"

def load_dataset():
    print 'Loading data...'
    Xdata = []
    ydata = []

    for src_dir, dirs, files in os.walk(work_dir):
        for file_ in files:
            list = []
            src_file = os.path.join(src_dir, file_)
            with open(src_file, 'r') as fp:
                file_str = fp.read()
            fp.close()
            dir_name = os.path.basename(src_dir)
            if dir_name == "1":
                ydata.append(0)
            elif dir_name == "2":
                ydata.append(1)
            elif dir_name == "3":
                ydata.append(2)
            elif dir_name == "4":
                ydata.append(3)
            elif dir_name == "5":
                ydata.append(4)
            elif dir_name == "6":
                ydata.append(5)
            elif dir_name == "7":
                ydata.append(6)
            elif dir_name == "8":
                ydata.append(7)
            elif dir_name == "9":
                ydata.append(8)
            elif dir_name == "10":
                ydata.append(9)
            else:
                print "Error in library name, name of the dir is: {0:s}".format(src_dir)
                continue
            Xdata.append(file_str)
    return (Xdata, ydata)

X_data, y_target = load_dataset()

print(len(X_data),len(y_target), 'data sequences')

nb_classes = np.max(y_target)+1
print(nb_classes, 'classes')

X_train, X_test, y_train, y_test = train_test_split(X_data, y_target, test_size=0.1, random_state=42)

tokenizer = Tokenizer(nb_words=max_features)
tokenizer.fit_on_texts(X_train)
print "Number of words: ", len(tokenizer.word_index)
wcounts = tokenizer.word_counts.items()
wcounts.sort(key=lambda x: x[1], reverse=True)
print "Most frequent words:", wcounts[:10]
print "Most rare words:", wcounts[-10:]
print "Number of words occurring %d times:" % wcounts[-1][1], np.sum(np.array(tokenizer.word_counts.values())==wcounts[-1][1])

print "Converting text to sequences..."
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

#X_train_sequences, y_train = SourceCodeUtils.cut_to_semi_redundant_sequences(X_train_sequences, y_train, maxlen, maxlen/2)
#X_test_sequences, y_test = SourceCodeUtils.cut_to_semi_redundant_sequences(X_test_sequences, y_test, maxlen, maxlen/2)

print "Padding sequences..."
# set the length of the sequences to be max len. cut the sequence from the end
X_train = sequence.pad_sequences(X_train_sequences, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test_sequences, maxlen=maxlen)

print('Convert y to binary vector (for use with categorical_crossentropy)')
Y_train_vector = np.zeros((len(X_train), nb_classes), dtype=np.bool)
Y_train_vector = np_utils.to_categorical(y_train, nb_classes)
y_train = Y_train_vector

Y_test_vector = np.zeros((len(X_test), nb_classes), dtype=np.bool)
Y_test_vector = np_utils.to_categorical(y_test, nb_classes)
y_test = Y_test_vector

print "Sequences train shape:", X_train.shape
print "Sequences test shape:", X_test.shape
print "Target train shape:", y_train.shape
print "Target test shape:", y_test.shape

print('Building model...')
model = Sequential()
model.add(Embedding(input_dim=max_features, output_dim=1024, input_length=maxlen, dropout=0.2))
model.add(LSTM(32, input_dim=1024, input_length=maxlen))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics = ["accuracy"])

print('Train...', str(datetime.now()))
early_stop = EarlyStopping(monitor='val_loss', patience=4, verbose=0, mode='auto')
hist = model.fit(X_train, y_train, nb_epoch=1, batch_size=batch_size, validation_split=0.1, verbose=1, callbacks=[early_stop])

print('Test...', str(datetime.now()))
score = model.evaluate(X_test, y_test, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

SourceCodeUtils.plot_loss(hist)