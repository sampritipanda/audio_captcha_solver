import os
import json

import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import random

from getPotentialSpeakLocation import getPotentialSpeakLocation

DISTORT_COUNT = 25
DISTORT_OFFSET = 300

#Parameters:
#dir: directory to gather data
#left, right: how far the segment extend to left/right from center point
#visualization=True: => special mode: visualization
#maxFiles=int or None => maxFiles is the maximum number of files we need to collect the data
#Return:
#X -> data: np.2D array
#y -> label: np.1D array
#rate -> rate of the wav file: int
def collectTrainData(dir, left, right, maxFiles=None, visualization=False):
    prefixes = list(set([x.split('.')[0] for x in os.listdir(dir)])) #list all files names
    plt.close() #init plit
    #setup
    X, y, sampleRate = [], [], 0    #return value
    numFiles = len(prefixes)
    if maxFiles is not None:
        numFiles = min(numFiles, maxFiles)
    fig, axes = plt.subplots(max(numFiles, 2) , 1 , figsize =(6 , 10), subplot_kw={'xticks': (), 'yticks': ()})
    #iterate through all audio/description files
    for i,ax in zip(range(numFiles), axes.ravel()):
        prefix = prefixes[i]
        wavFile = os.path.join(dir, prefix + ".wav")    
        outFile = os.path.join(dir, prefix + ".txt")
        #read/parse .wav file and .txt file
        rate, data = scipy.io.wavfile.read(wavFile)
        output = json.load(open(outFile))
        #pad zero to the begin and end of the array to prevent out of bound
        data = np.asarray([0] * left + list(data) + [0] * right)
        #collect the spoken location data
        spokenLocations = map(int, output["offsets"][1:-1].split(','))
        spokenLocations = [(x + left) for x in spokenLocations]
        expectedLocs = getPotentialSpeakLocation(data, rate, left, right, 4)
        for i in range(len(spokenLocations)):
            if abs(spokenLocations[i] - expectedLocs[i]) < min(left, right):
                spokenLocations[i] = expectedLocs[i]

        #colect the label of each position
        labels = []
        for x in list(output["code"]):
            labels.append(int(x) if x.isdigit() else 10 + ord(x) - ord('a'))
        #calculate the length of the audio
        seconds = len(data)/rate
        #calculate rate
        sampleRate = rate
        #===============visualization
        if visualization:
            ax.plot(np.array([seconds*i/len(data) for i in range(len(data))]), data) 
            ax.set_xlim(0, 10)  
        for location,label in zip(spokenLocations, labels):
            #distort the train data 
            #Why?
            #1. More train data
            #2. Prevent overfiting
            for r in range(DISTORT_COUNT):
                random_offset = random.randrange(-DISTORT_OFFSET, DISTORT_OFFSET)
                if r == 0:
                    random_offset = 0
                #get start/end point of each segment
                sta = location - left + random_offset
                fin = location + right + random_offset
                #===============visualization
                if r == 0 and visualization:
                    ax.axvline(seconds/len(data)*sta, color='red')
                    ax.axvline(seconds/len(data)*fin, color='red')
                #append to the data collection
                X.append(data[sta:fin])
                y.append(label)
    if visualization:
        plt.show()
    
    return np.array(X), np.array(y), sampleRate
    
if __name__ == "__main__":
    DIR = os.path.join("data", "securimage_all", "train")
    LEFT = 2500
    RIGHT = 2500
    X,y,rate = collectTrainData(DIR, LEFT, RIGHT, 10, False)
    print("===================X:")
    print(X)
    print("====================y:")
    print(y)
    collectTrainData(DIR, LEFT, RIGHT, 10, True)
