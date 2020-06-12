'''
Wrapper for all MachineLearning algorithm
'''

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras import backend as K

class KNN:
    def __init__(self, n_neighbors):
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    def train(self, X_train, y_train):
        self.knn.fit(X_train, y_train)

    def score(self, X_test, y_test):
        return self.knn.score(X_test, y_test)

    def predict(self, X_test):
        return self.knn.predict(X_test)

class SVM:
    def __init__(self):
        self.svm = SVC(C=32)

    def train(self, X_train, y_train):
        self.svm.fit(X_train, y_train)

    def score(self, X_test, y_test):
        return self.svm.score(X_test, y_test)

    def predict(self, X_test):
        return self.svm.predict(X_test)

class NeuralNetFeatures:
    def __init__(self, num_classes):
        self.batch_size = 128
        self.epochs = 20
        self.num_classes = num_classes

        self.model = Sequential()
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='softmax'))

        self.model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

    def train(self, X_train, y_train):
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        self.model.fit(X_train, y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       verbose=1,
                       validation_split=0.1,
                       shuffle=True)

    def score(self, X_test, y_test):
        y_train = keras.utils.to_categorical(y_test, self.num_classes)
        score = self.model.evaluate(X_test, y_test, verbose=0)
        return score

    def predict(self, X_test):
        y_predict = self.model.predict(X_test)
        y_out = np.argmax(y_predict, axis=1)
        return y_out

class NeuralNetRaw:
    def __init__(self, num_classes):
        self.batch_size = 128
        self.epochs = 20
        self.num_classes = num_classes

        self.model = Sequential()
        self.model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(self.num_classes, activation='softmax'))

        self.model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

    def train(self, X_train, y_train):
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        self.model.fit(X_train, y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       verbose=1,
                       validation_split=0.1,
                       shuffle=True)

    def score(self, X_test, y_test):
        y_train = keras.utils.to_categorical(y_test, self.num_classes)
        score = self.model.evaluate(X_test, y_test, verbose=0)
        return score

    def predict(self, X_test):
        y_predict = self.model.predict(X_test)
        y_out = np.argmax(y_predict, axis=1)
        return y_out
