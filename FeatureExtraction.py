'''
Wrapper for all FeatureExtraction algorithm
'''
import numpy as np

from python_speech_features import mfcc

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
    def __call__(self, X, rate):
        ret = []
        for signal in X:
            features = mfcc(signal,rate).flatten()
            ret.append(features)
        return np.array(ret)
