import json
import random
import os

import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from python_speech_features import mfcc

from collectTrainData import collectTrainData
from getPotentialSpeakLocation import getPotentialSpeakLocation

def featureExtraction(X, rate):
    X_data = []
    for signal in X:
        features = mfcc(signal,rate).flatten()
        X_data.append(features)
    return np.array(X_data)

#Paramters
DIR_TRAIN = os.path.join("data", "securimage_digits", "train")
DIR_TEST = os.path.join("data", "securimage_digits", "test")
LEFT = 2500
RIGHT = 2500
OFF = 2500
print("Train_data location = " + DIR_TRAIN)
print("Test_data location = " + DIR_TEST)

#1.collect the raw train data
X_train,y_train,rate = collectTrainData(DIR_TRAIN, LEFT, RIGHT, False)

#2. feature extraction
X_train = featureExtraction(X_train, rate) 

#3. build the model
knn = KNeighborsClassifier(n_neighbors=7) #KNN algorithm
knn.fit(X_train, y_train)

#4. try to solve each output
#Iterate through all outputs
prefixes = list(set([x.split('.')[0] for x in os.listdir(DIR_TEST)])) #list all files names
for i in range(10):
    prefix = prefixes[i]
    #4.1. Read the file
    wavFile = os.path.join(DIR_TEST, prefix + ".wav")    
    outFile = os.path.join(DIR_TEST, prefix + ".txt")
    #read/parse .wav file and .txt file
    rate, data = scipy.io.wavfile.read(wavFile)
    output = json.load(open(outFile))
    #4.2. Get potential spoken locs
    locs = getPotentialSpeakLocation(data, rate, LEFT, RIGHT, 4, False, OFF)
    #4.3. Build the answer
    captchas = ""
    #Iterate through each loc
    print("===================================")
    print("Actual locs = " + str(locs))
    print("Expected locs = " + output["offsets"])
    for loc in locs:
        sta = loc - LEFT
        fin = loc + RIGHT
        signal = data[sta:fin + 1]
        #feature exaction
        signal = featureExtraction([signal], rate)[0]
        predicted = knn.predict(np.array([signal]))
        captchas += str(predicted[0])
    print("Actual output = %s | Expected output = %s"%(captchas, output["code"]))

    
    