import os
import json

import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt

#Parameters:
#dir: directory to gather data
#left, right: how far the segment extend to left/right from center point
#visualization=True: => special mode: visualization
#Return:
#X -> data: np.2D array
#y -> label: np.1D array
def collectTrainData(dir, left, right, visualization=False):
    prefixes = list(set([x.split('.')[0] for x in os.listdir(dir)])) #list all files names
    plt.close()
    fig, axes = plt.subplots(len(prefixes) , 1 , figsize =(6 , 10), subplot_kw={'xticks': (), 'yticks': ()})
    X, y = [], []
    for i,prefix,ax in zip(range(len(prefixes)), prefixes, axes.ravel()):
        wavFile = os.path.join(dir, prefix + ".wav")    
        outFile = os.path.join(dir, prefix + ".txt")
        #read/parse .wav file and .txt file
        rate, data = scipy.io.wavfile.read(wavFile)
        output = json.load(open(outFile))
        #collect the necessary information from the .txt file
        startPoints = map(int, output["offsets"][1:-1].split(','))
        labels = map(int, list(output["code"]))
        #calculate the length of the audio
        seconds = len(data)/rate
        ax.plot(np.array([seconds*i/len(data) for i in range(len(data))]), data) #visualization
        for i,j in zip(startPoints, labels):
            #get start/end point of each segment
            sta = i - left + 1
            fin = i + right - 1
            #plot
            ax.axvline(seconds/len(data)*sta, color='red')
            ax.axvline(seconds/len(data)*fin, color='red')
            #append to the data collection
            X.append(data[sta:fin])
            y.append(j)
    if visualization:
        plt.show()
    
    return np.array(X), np.array(y)
    
if __name__ == "__main__":
    DIR = os.path.join("training_data", "output_securimage")
    LEFT = 2500
    RIGHT = 2500
    X,y = collectTrainData(DIR, LEFT, RIGHT, False)
    print("===================X:")
    print(X)
    print("====================y:")
    print(y)
    collectTrainData(DIR, LEFT, RIGHT, True)
