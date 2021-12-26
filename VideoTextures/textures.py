#CITE This is original work completed for spring 2019 offering of computational photography
#CITE END


import numpy as np
import scipy as sp
import cv2
import scipy.signal


def videoVolume(images):
    length = len(images)
    video = np.zeros(shape = (length, images[0].shape[0],images[0].shape[1],images[0].shape[2]),dtype = "uint8")
    for i in range(length):
        video[i] = images[i]
    return video

def computeSimilarityMetric(video_volume):
    
    num_images = video_volume.shape[0]
    distance = np.zeros(shape = (num_images,num_images), dtype="float64")
    video_volume = video_volume.astype(dtype = 'float64')
    for i in range(num_images-1):
        for j in range(i+1,num_images):
            difference = video_volume[i] - video_volume[j]
            difference = np.square(difference)
            sum = np.sum(difference)
            rssd = np.sqrt(sum)
            distance[i,j] = rssd
            distance[j,i] = rssd
    average = np.sum(distance)/num_images/num_images
    distance = distance/average
    return distance


def transitionDifference(similarity):
    sim_shape = similarity.shape
    w = np.identity(5)
    binomial = binomialFilter5()
    w = w*binomial
    diff = scipy.signal.convolve2d(similarity, w, mode = 'valid')
    return diff


def findBiggestLoop(transition_diff, alpha):
    score_shape = transition_diff.shape
    score = 0
    start = 0
    end = 0
    for i in range(score_shape[0]):
        for j in range(i+1,score_shape[1]):
            print(score)
            score_temp = alpha*(j-i) - transition_diff[j][i]
            if score_temp*255>score*255:
                score = score_temp
                start = i
                end = j
    
    indices = (start+2,end+2)
    return indices


def synthesizeLoop(video_volume, start, end):
    loop = []
    frames = video_volume[start:end+1]
    for i in range(frames.shape[0]):
        loop.append(frames[i])
        
    return loop


def binomialFilter5():
    """Return a binomial filter of length 5.

    NOTE: DO NOT MODIFY THIS FUNCTION.

    Returns
    -------
    numpy.ndarray(dtype: np.float)
        A 5x1 numpy array representing a binomial filter.
    """
    return np.array([1 / 16., 1 / 4., 3 / 8., 1 / 4., 1 / 16.], dtype=float)
