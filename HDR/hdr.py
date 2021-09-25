import numpy as np
import scipy as sp
import cv2
import random

def linearWeight(pixel_value):
    z_min, z_max = 0., 255.
    if pixel_value <= (z_max + z_min)/2:
        value = np.float64(pixel_value) - z_min
    else: 
        value = z_max - np.float64(pixel_value)
    return value

def sampleIntensities(images):
    num_intensities = 256
    num_images = len(images)
    intensity_values = np.zeros((num_intensities, num_images), dtype=np.uint8)

    mid_img = images[num_images // 2] 
    mid_img_index = num_images // 2
    pixel_index_dict = {}
    for i in range(mid_img.shape[0]):
        for j in range(mid_img.shape[1]):
            key = str(mid_img[i][j])
            if key not in pixel_index_dict:
                pixel_index_dict.update({key:[(i,j)]})
            else:
                pixel_index_dict[key].append((i,j))
    
    pixels = {}
    pixels_list = []
    for i in range(len(pixel_index_dict)):
        pixels.update({str(i):[]})

    for i in range(len(pixels)):
        if len(pixel_index_dict[str(i)])>0:
            pixels[str(i)] = random.choice(pixel_index_dict[str(i)])
            pixels_list.append(pixels[str(i)])
            index = pixel_index_dict[str(i)].index(pixels[str(i)])
    for i in range(len(images)):
        for j in range(len(pixels_list)):
            intensity_values[j,i] = images[i][pixels_list[j]]
    return  intensity_values
    
    


def computeResponseCurve(intensity_samples, log_exposures, smoothing_lambda, weighting_function):

    intensity_range = 255
    num_samples = intensity_samples.shape[0]
    num_images = len(log_exposures)

    mat_A = np.zeros((num_images * num_samples + intensity_range,
                      num_samples + intensity_range + 1), dtype=np.float64)
    mat_b = np.zeros((mat_A.shape[0], 1), dtype=np.float64)

    n = 256
    k = 0
    for i in range(intensity_samples.shape[0]):
        for j in range(intensity_samples.shape[1]):
            w_ij = weighting_function(intensity_samples[i][j])
            mat_A[k,intensity_samples[i][j]] = w_ij
            mat_A[k,n+i] =  -1*w_ij
            mat_b[k,0] = w_ij*log_exposures[j]
            k = k + 1

    for i in range(n-2):
        mat_A[k][i] = smoothing_lambda*weighting_function(i+1)
        mat_A[k][i+1] = -2*smoothing_lambda*weighting_function(i+1)
        mat_A[k][i+2] = smoothing_lambda*weighting_function(i+1)
        k = k + 1

    mat_A[k,intensity_range//2] = 1

    x = np.dot(np.linalg.pinv(mat_A),mat_b)

    g = x[0:intensity_range + 1]

    return g[:, 0]


def computeRadianceMap(images, log_exposure_times, response_curve, weighting_function):
    img_shape = images[0].shape
    img_rad_map = np.zeros(img_shape, dtype=np.float64)

    mid_img = len(images)//2
    for i in range(img_shape[0]):
        w = np.zeros(len(images))
        g = np.zeros(len(images))
        for j in range(img_shape[1]):
            for k in range(len(images)):
                z = images[k][i][j]
                w[k] = weighting_function(z)
                g[k] = response_curve[z]
            w_sum = np.sum(w)
            if w_sum>0:
                w_scaled = w/w_sum
                img_rad_map[i][j] = np.sum(w_scaled*(g-log_exposure_times))
            else:
                img_rad_map[i][j] = response_curve[images[mid_img][i][j]] - log_exposure_times[mid_img]
    return img_rad_map
