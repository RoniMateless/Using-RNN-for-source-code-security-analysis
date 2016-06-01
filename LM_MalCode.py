import ntpath
import random
import os
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


max_features = 5000
maxlen = 100
batch_size = 32

mal_work_dir = "C:\\Datasets\\Malicious Source Code\\MalFiles\\"
benign_work_dir = "C:\\Datasets\\Malicious Source Code\\BenignFiles\\"

def load_files(work_dir, target, max_files = 3000):
    Xdata = []
    ydata = []
    count_files=0

    for src_dir, dirs, files in os.walk(work_dir):
        for file_ in files:
            list = []
            if max_files < count_files:
                break
            count_files += 1
            src_file = os.path.join(src_dir, file_)
            with open(src_file, 'r') as fp:
                list = fp.read()
            fp.close()
            dir_name = os.path.basename(src_dir)
            ydata.append(target)
            Xdata.append(list)

    return (Xdata, ydata)


def load_dataset():
    print 'Loading data...'
    X_benign, Y_benign = load_files(benign_work_dir,0, 1000)
    X_mal, Y_mal = load_files(mal_work_dir,1, 1000)

    print "Positive Samples: " + str(len(X_mal))
    print "Negative Samples: " + str(len(X_benign))

    return (X_mal + X_benign, Y_mal + Y_benign)

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

print "Padding sequences..."
# set the length of the sequences to be max len. cut the sequence from the end
X_train = sequence.pad_sequences(X_train_sequences, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test_sequences, maxlen=maxlen)

print "Sequences train shape:", X_train.shape
print "Sequences test shape:", X_test.shape

print('Building model...')
model = Sequential()
model.add(Embedding(input_dim=max_features, output_dim=1024, input_length=maxlen, dropout=0.2))
model.add(LSTM(32, input_dim=1024, input_length=maxlen))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# optimizer
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics = ["accuracy"])

print('Train...', str(datetime.now()))
early_stop = EarlyStopping(monitor='val_loss', patience=4, verbose=0, mode='auto')
hist = model.fit(X_train, y_train, nb_epoch=50, batch_size=batch_size, validation_split=0.1, verbose=1, callbacks=[early_stop])

print('Test...', str(datetime.now()))
score = model.evaluate(X_test, y_test, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])
