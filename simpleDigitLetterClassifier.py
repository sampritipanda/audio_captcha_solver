import os
import json
import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from python_speech_features import mfcc

from collectTrainData import collectTrainData

def featureExtraction(X, rate):
    X_data = []
    for signal in X:
        features = mfcc(signal,rate).flatten()
        X_data.append(features)
    return np.array(X_data)

#collect data
DIR_TRAIN = os.path.join("data", "securimage_digits_distorted", "train")
DIR_TEST = os.path.join("data", "securimage_digits_distorted", "test")
LEFT = 100
RIGHT = 100
print("Train_data location = " + DIR_TRAIN)
print("Test_data location = " + DIR_TEST)
X_train,y_train,rate = collectTrainData(DIR_TRAIN, LEFT, RIGHT, False)
X_test, y_test, rate = collectTrainData(DIR_TEST, LEFT, RIGHT, False)
#====================================================================================
print("KNN with features = raw array of amplitudes:")
# split the data into training and test sets\n",
#X_train, X_test, y_train, y_test = train_test_split(
#    X, y, stratify=y, random_state=0)
knn = KNeighborsClassifier(n_neighbors=1) #KNN algorithm
knn.fit(X_train, y_train)
print("Test set score of 1-nn: {:.4f}".format(knn.score(X_test, y_test)))

"""
#plot samples
print("=> Plot samples array of amplitudes for each single digit......")
nSample = 7
count = [0 for i in range(10)]
image = [None for i in range(10*nSample)]
plt.close()
fig, axes = plt.subplots(10 , nSample , figsize =(7 , 7), subplot_kw={'xticks': (), 'yticks': ()})
for signal,label in zip(X_train,y_train):
    if count[label] < nSample:
        image[nSample*label + count[label]] = signal
        count[label] += 1

seconds = len(image[0])/rate
times = np.array([seconds*i/len(image[0]) for i in range(len(image[0]))])
for img,ax in zip(image, axes.ravel()):
    ax.plot(times, img)
    ax.set_xlim(0, seconds)
plt.show()
"""
