import numpy as np
import matplotlib.pyplot as plt

def getPotentialSpeakLocation(data, rate, left, right, num_letters):
    size = left + right
    sample_width = 10
    x = []
    for i in range(left, len(data) - size, sample_width):
        x.append((np.mean(np.abs(data[i:i+size])), i))
    # plt.close()
    # seconds = len(data)/rate
    # fig, axes = plt.subplots(2 , 1 , figsize =(6 , 10), subplot_kw={'xticks': (), 'yticks': ()})
    # ax = axes.ravel()[0]
    # ax.plot(np.array([seconds*i/len(data) for i in range(len(data))]), data) #visualization
    # ax.set_xlim(0, 10)  #!!!!!!!!!!
    # plt.show()
    x = sorted(x)[::-1]
    locs = []
    i = 0
    while len(locs) < num_letters:
        cur_loc = x[i][1] + size//2
        valid = True
        for loc in locs:
            if abs(loc - cur_loc) <= size:
                valid = False
                break
        if valid:
            locs.append(cur_loc)
        i += 1
    locs = sorted(locs)
    return locs
