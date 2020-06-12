import json
import random
import os

import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from collectTrainData import collectTrainData
from getPotentialSpeakLocation import getPotentialSpeakLocation
from FeatureExtraction import Rasta,Mfcc,Raw
from MLAlgo import KNN, SVM, NeuralNetFeatures, NeuralNetRaw


def runPipeline(dir_train, dir_test, left, right, mlModel, featureExtraction, applyPca):
    #1.collect the raw train data
    print("Collect raw train data from %s..."%(dir_train))
    X_train,y_train,rate = collectTrainData(dir_train, left, right)

    #2. feature extraction
    print("Apply feature extraction to the raw data...")
    X_train = featureExtraction(X_train, rate)

    pca = PCA(n_components = 0.95)
    scaler = MinMaxScaler()

    #******PCA (optional)
    if applyPca:
        print("Apply PCA to reduce the dimensionality of the data to 95%...")
        # X_train = scaler.fit_transform(X_train)
        pca.fit(X_train)
        X_train = pca.transform(X_train)

    #3. train the model
    print("Train the model...")
    mlModel.train(X_train, y_train)

    #4. solve each test data
    print("Solve each test input from %s:"%dir_test)
    prefixes = list(set([x.split('.')[0] for x in os.listdir(DIR_TEST)])) #list all files names
    prefixes = sorted(prefixes)
    count = 0
    count_34th = 0
    indiv_count = 0
    for i in range(len(prefixes)):
        prefix = prefixes[i]
        #4.1. Read the file
        wavFile = os.path.join(DIR_TEST, prefix + ".wav")
        outFile = os.path.join(DIR_TEST, prefix + ".txt")
        #read/parse .wav file and .txt file
        rate, data = scipy.io.wavfile.read(wavFile)
        data = np.asarray([0] * LEFT + list(data) + [0] * RIGHT)
        output = json.load(open(outFile))
        #4.2. Get potential spoken locs
        locs = getPotentialSpeakLocation(data, rate, LEFT, RIGHT, 4)
        #Get the expectedLocs from the test file
        expectedLocs = map(int, output["offsets"][1:-1].split(','))
        expectedLocs = [(x + LEFT) for x in expectedLocs]
        print("======Test %d:"%(i))
        print("Actual spoken locs = " + str(locs))
        print("Expected spoken locs = " + str(expectedLocs))
        #4.3. Build the answer
        captchas = ""
        signals = []
        #Iterate through each loc
        for loc in locs:
            sta = loc - LEFT
            fin = loc + RIGHT
            signals.append(data[sta:fin])
        signals = np.array(signals)
        #feature extraction
        signals = featureExtraction(signals, rate)
        #******PCA (optional)
        if applyPca:
            # signals = scaler.fit_transform(signals)
            signals = pca.transform(signals)
        #predict the output for each individual token
        predictedVals = mlModel.predict(signals)
        for c in predictedVals:
            captchas += str(c) if c < 10 else chr(ord('a') + c - 10)
        if captchas == output["code"]:
            count += 1
        cur_cnt = 0
        for i in range(4):
            if captchas[i] == output["code"][i]:
                cur_cnt += 1
                indiv_count += 1
        if cur_cnt == 3:
            count_34th += 1
        print("Actual output = %s | Expected output = %s"%(captchas, output["code"]))

    print("Accuracy = %.4f"%(count/len(prefixes)))
    print("3/4th Accuracy = %.4f"%((count + count_34th)/len(prefixes)))
    print("Accuracy of Individual Digits = %.4f"%((indiv_count)/(len(prefixes) * 4)))

#########CONFIGURATION FOR THE PIPELINE
if __name__ == '__main__':
    DIR_TRAIN = os.path.join("data", "securimage_all", "train")
    DIR_TEST = os.path.join("data", "securimage_all", "test")
    LEFT = 2500
    RIGHT = 2500
    MLMODEL = SVM()
    FEATURE_EXTRATION = Mfcc(flatten=True)
    runPipeline(DIR_TRAIN, DIR_TEST, LEFT, RIGHT, MLMODEL, FEATURE_EXTRATION, applyPca=True)
