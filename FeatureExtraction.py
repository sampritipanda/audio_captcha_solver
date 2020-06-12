'''
Wrapper for all FeatureExtraction algorithm
'''
import numpy as np

from python_speech_features import mfcc
import rasta

'''
Leave the data as raw
'''
class Raw:
    def __call__(self, X, rate):
        return X

'''
FeatureExtraction using Mfcc
https://medium.com/@jonathan_hui/speech-recognition-feature-extraction-mfcc-plp-5455f5a69dd9
'''
class Mfcc:
    def __init__(self, flatten=True):
        self.flatten = flatten

    def __call__(self, X, rate):
        ret = []
        for signal in X:
            features = mfcc(signal,rate)
            if self.flatten:
                features = features.flatten()
            ret.append(features)
        return np.array(ret)


class Rasta:
    def __call__(self, X, rate):
        ret = []
        for signal in X:
            features = rasta.rastaplp(signal.astype(np.float32), rate, dorasta=True).flatten()
            ret.append(features)
        return np.array(ret)
