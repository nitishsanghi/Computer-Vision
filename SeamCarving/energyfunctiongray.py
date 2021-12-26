import numpy as np
from math import*
import matplotlib.pyplot as plt
import cv2 
import glob
import os
from numba import jit
from skimage.filters.rank import entropy
from skimage.morphology import square
#from scipy.signal import convolve
from scipy.ndimage.filters import convolve
import time



@jit
def l1energynorm(img_gray):
    img_sobelx_3 = cv2.Sobel(img_gray,cv2.CV_64F,1,0,3)
    img_sobely_3 = cv2.Sobel(img_gray,cv2.CV_64F,0,1,3)
    return np.absolute(img_sobelx_3) + np.absolute(img_sobely_3)

@jit
def x_gradient_gray(image):
    x_gradient = np.array([[-1.0, -2.0, -1.0],[0.0, 0.0, 0.0],[1.0, 2.0, 1.0]])
    return convolve(image, x_gradient)
@jit
def y_gradient_gray(image):
    y_gradient = np.array([[-1.0, 0.0, 1.0],[-2.0, 0.0, 2.0],[-1.0, 0.0, 1.0]])
    return convolve(image, y_gradient)
@jit
def l1_energy_function(y_gradient_gray,x_gradient_gray):
     absolute_gradient_sum = np.absolute(x_gradient_gray) + np.absolute(y_gradient_gray)
     return absolute_gradient_sum

@jit
def cumulativeenergy(img_energy, matrix_energy):
    for i in range(matrix_energy.shape[0]):
        for j in range(matrix_energy.shape[1]):
            if i==0:
                matrix_energy[i][j][0] = img_energy[i][j]
                matrix_energy[i][j][1] = j
            else:
                if j == 0:
                    matrix_energy[i][j][0] = img_energy[i][j] + np.amin(matrix_energy[i-1,:j+2,0])
                    matrix_energy[i][j][1] = j + np.argmin(matrix_energy[i-1,:j+2,0])
                elif j==matrix_energy.shape[1]-1:
                    matrix_energy[i][j][0] = img_energy[i][j] + np.amin(matrix_energy[i-1,j-1:j+1,0])
                    matrix_energy[i][j][1] = j + np.argmin(matrix_energy[i-1,j-1:j+1,0]) - 1
                else:
                    matrix_energy[i][j][0] = img_energy[i][j] + np.amin(matrix_energy[i-1,j-1:j+2,0])
                    matrix_energy[i][j][1] = j + np.argmin(matrix_energy[i-1,j-1:j+2,0]) - 1
    return matrix_energy

@jit
def lowestcumulativeenergyseam(matrix_energy):
    last_row = matrix_energy[-1,:,0]
    x = matrix_energy.shape[0]-1
    y = np.argmin(last_row)
    seamlist = []
    seamlisty = []
    seamenergy = []
    seamlist.append([x,y])
    seamlisty.append(y)
    seamenergy.append(matrix_energy[x,y,0])
    for i in range(1,matrix_energy.shape[0]):
        y = int(matrix_energy[x][y][1])
        x = x -1
        seamlist.append([x,y])
        seamlisty.append(y)
        seamenergy.append(matrix_energy[x,y,0])
    return seamlisty

@jit
def removeverticalseam(img_gray,img_rgb,seampixellist):
        img_gray_seam_removed = np.zeros(shape = (img_gray.shape[0],img_gray.shape[1]-1),dtype = 'uint8')
        img_red_seam_removed = np.zeros(shape = (img_gray.shape[0],img_gray.shape[1]-1),dtype = 'uint8')
        img_green_seam_removed = np.zeros(shape = (img_gray.shape[0],img_gray.shape[1]-1),dtype = 'uint8')
        img_blue_seam_removed = np.zeros(shape = (img_gray.shape[0],img_gray.shape[1]-1),dtype = 'uint8')
        for i in range(len(seampixellist)):
            x = seampixellist[i]
            img_gray_seam_removed[img_rgb.shape[0]-1-i,:x] = img_gray[img_gray.shape[0]-1-i,:x]
            img_gray_seam_removed[img_rgb.shape[0]-1-i,x:] = img_gray[img_gray.shape[0]-1-i,x+1:]
            img_red_seam_removed[img_rgb.shape[0]-1-i,:x] = img_rgb[img_rgb.shape[0]-1-i,:x,0]
            img_red_seam_removed[img_rgb.shape[0]-1-i,x:] = img_rgb[img_rgb.shape[0]-1-i,x+1:,0]
            img_green_seam_removed[img_rgb.shape[0]-1-i,:x] = img_rgb[img_rgb.shape[0]-1-i,:x,1]
            img_green_seam_removed[img_rgb.shape[0]-1-i,x:] = img_rgb[img_rgb.shape[0]-1-i,x+1:,1]
            img_blue_seam_removed[img_rgb.shape[0]-1-i,:x] = img_rgb[img_rgb.shape[0]-1-i,:x,2]
            img_blue_seam_removed[img_rgb.shape[0]-1-i,x:] = img_rgb[img_rgb.shape[0]-1-i,x+1:,2]
        img_rgb_seam_removed = np.zeros(shape = (img_gray.shape[0],img_gray.shape[1]-1,3),dtype = 'uint8')
        img_rgb_seam_removed[:,:,0] = img_red_seam_removed
        img_rgb_seam_removed[:,:,1] = img_green_seam_removed
        img_rgb_seam_removed[:,:,2] = img_blue_seam_removed
        img_rgb = img_rgb_seam_removed
        return img_rgb,img_gray_seam_removed

@jit
def highlightseam(img_rgb,seampixellist):
    for i in range(len(seampixellist)):
        img_rgb[img_rgb.shape[0]-i-1,seampixellist[i]] = [0,0,255]
    return img_rgb

filenames = os.listdir('images')
img_rgb = cv2.imread('images/fig5.png')

delta_x = 350
for i in range(delta_x):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_gray = img_gray.astype("float64")
    #img_energy = l1energynorm(img_gray)
    #img_energy = l1_energy_function(x_gradient_rgb(img_gray),y_gradient_rgb(img_gray))
    img_energy = l1_energy_function(x_gradient_gray(img_gray),y_gradient_gray(img_gray))
    matrix_energy_vertical = np.zeros(shape = (img_gray.shape[0],img_gray.shape[1],2), dtype = "float")
    matrix_energy_vertical = cumulativeenergy(img_energy,matrix_energy_vertical)
    seampixellist = lowestcumulativeenergyseam(matrix_energy_vertical)
    cv2.imshow('seamline',highlightseam(img_rgb,seampixellist))
    img_rgb,img_gray = removeverticalseam(img_gray,img_rgb,seampixellist)
    cv2.imshow('image color',img_rgb)
    cv2.waitKey(2)
    

img_rgb.astype('uint8')
cv2.imwrite('images/seamed_image.jpg',img_rgb)
cv2.imshow('image color',img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
