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


@jit
def l1energynorm(img_sobelx_3,img_sobely_3):
    return np.absolute(img_sobelx_3) + np.absolute(img_sobely_3)

@jit
def x_gradient_rgb(image):
    x_gradient = np.array([[-1.0, -2.0, -1.0],[0.0, 0.0, 0.0],[1.0, 2.0, 1.0]])
    x_gradient = np.stack([x_gradient]*3, axis=2)
    return convolve(image, x_gradient)
@jit
def y_gradient_rgb(image):
    y_gradient = np.array([[-1.0, 0.0, 1.0],[-2.0, 0.0, 2.0],[-1.0, 0.0, 1.0]])
    y_gradient = np.stack([y_gradient]*3, axis=2)
    return convolve(image, y_gradient)
@jit
def l1_energy_function(y_gradient_rgb,x_gradient_rgb):
     absolute_gradient_sum = np.absolute(x_gradient_rgb) + np.absolute(y_gradient_rgb)
     absolute_rgb_sum = absolute_gradient_sum.sum(axis=2)
     return absolute_rgb_sum

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
def removeverticalseam(img_rgb,seampixellist):
        img_red_seam_removed = np.zeros(shape = (img_rgb.shape[0],img_rgb.shape[1]-1),dtype = 'uint8')
        img_green_seam_removed = np.zeros(shape = (img_rgb.shape[0],img_rgb.shape[1]-1),dtype = 'uint8')
        img_blue_seam_removed = np.zeros(shape = (img_rgb.shape[0],img_rgb.shape[1]-1),dtype = 'uint8')
        for i in range(len(seampixellist)):
            x = seampixellist[i]
            img_red_seam_removed[img_rgb.shape[0]-1-i,:x] = img_rgb[img_rgb.shape[0]-1-i,:x,0]
            img_red_seam_removed[img_rgb.shape[0]-1-i,x:] = img_rgb[img_rgb.shape[0]-1-i,x+1:,0]
            img_green_seam_removed[img_rgb.shape[0]-1-i,:x] = img_rgb[img_rgb.shape[0]-1-i,:x,1]
            img_green_seam_removed[img_rgb.shape[0]-1-i,x:] = img_rgb[img_rgb.shape[0]-1-i,x+1:,1]
            img_blue_seam_removed[img_rgb.shape[0]-1-i,:x] = img_rgb[img_rgb.shape[0]-1-i,:x,2]
            img_blue_seam_removed[img_rgb.shape[0]-1-i,x:] = img_rgb[img_rgb.shape[0]-1-i,x+1:,2]
        img_rgb_seam_removed = np.zeros(shape = (img_rgb.shape[0],img_rgb.shape[1]-1,3),dtype = 'uint8')
        img_rgb_seam_removed[:,:,0] = img_red_seam_removed
        img_rgb_seam_removed[:,:,1] = img_green_seam_removed
        img_rgb_seam_removed[:,:,2] = img_blue_seam_removed
        img_rgb = img_rgb_seam_removed
        return img_rgb


    
@jit
def imageenlarge(pixelcoordinates,img_rgb_primary):
    img_rgb_insertion  = img_rgb_primary
    number_of_pixel = len(pixelcoordinates)
    for nop in range(number_of_pixel):
        seamlist1 = pixelcoordinates.pop(0)
        img_rgb_temp = np.zeros_like(img_rgb_insertion)
        img_rgb_temp = img_rgb_insertion
        #cv2.imshow("seamline",highlightseam(img_rgb_temp,seamlist1))
        img_red_seam_insertion = np.zeros(shape = (img_rgb_insertion.shape[0],img_rgb_insertion.shape[1]+1), dtype="uint8")
        img_green_seam_insertion = np.zeros(shape = (img_rgb_insertion.shape[0],img_rgb_insertion.shape[1]+1), dtype="uint8")
        img_blue_seam_insertion = np.zeros(shape = (img_rgb_insertion.shape[0],img_rgb_insertion.shape[1]+1), dtype="uint8")
        for i in range(len(seamlist1)):
            x = seamlist1[i]
            img_red_seam_insertion[img_rgb_insertion.shape[0]-1-i,:x] = img_rgb_insertion[img_rgb_insertion.shape[0]-1-i,:x,0]
            img_red_seam_insertion[img_rgb_insertion.shape[0]-1-i,x] = meanpixel(img_rgb_insertion,img_rgb_insertion.shape[0]-1-i,x,0)
            #img_red_seam_insertion[img_rgb_insertion.shape[0]-1-i,x] = 0
            img_red_seam_insertion[img_rgb_insertion.shape[0]-1-i,x+1:] = img_rgb_insertion[img_rgb_insertion.shape[0]-1-i,x:,0]
            
            img_green_seam_insertion[img_rgb_insertion.shape[0]-1-i,:x] = img_rgb_insertion[img_rgb_insertion.shape[0]-1-i,:x,1]
            img_green_seam_insertion[img_rgb_insertion.shape[0]-1-i,x] = meanpixel(img_rgb_insertion,img_rgb_insertion.shape[0]-1-i,x,1)
            #img_green_seam_insertion[img_rgb_insertion.shape[0]-1-i,x] = 0
            img_green_seam_insertion[img_rgb_insertion.shape[0]-1-i,x+1:] = img_rgb_insertion[img_rgb_insertion.shape[0]-1-i,x:,1]
            
            img_blue_seam_insertion[img_rgb_insertion.shape[0]-1-i,:x] = img_rgb_insertion[img_rgb_insertion.shape[0]-1-i,:x,2]
            img_blue_seam_insertion[img_rgb_insertion.shape[0]-1-i,x] = meanpixel(img_rgb_insertion,img_rgb_insertion.shape[0]-1-i,x,2)
            #img_blue_seam_insertion[img_rgb_insertion.shape[0]-1-i,x] = 255
            img_blue_seam_insertion[img_rgb_insertion.shape[0]-1-i,x+1:] = img_rgb_insertion[img_rgb_insertion.shape[0]-1-i,x:,2]
        img_rgb_seam_insertion = np.zeros(shape = (img_rgb_insertion.shape[0],img_rgb_insertion.shape[1]+1,3),dtype = 'uint8')
        img_rgb_seam_insertion[:,:,0] = img_red_seam_insertion
        img_rgb_seam_insertion[:,:,1] = img_green_seam_insertion
        img_rgb_seam_insertion[:,:,2] = img_blue_seam_insertion
        img_rgb_insertion = img_rgb_seam_insertion
        remaining_seams = []
        for seamlist2 in pixelcoordinates:
            for i in range(len(seamlist2)):
                if seamlist2[i] >= seamlist1[i]:
                    seamlist2[i] = seamlist2[i] + 2
            remaining_seams.append(seamlist2)
        pixelcoordinates = remaining_seams
        cv2.imshow("seaminsertion",img_rgb_insertion)
        cv2.waitKey(2)
        
    return img_rgb_insertion
    
@jit
def meanpixel(img_rgb,x,y,color):
    if y==0:
        return img_rgb[x,y,color]/2 + img_rgb[x,y+1,color]/2
    elif y== img_rgb.shape[1]-1:
        return img_rgb[x,y,color]/2 + img_rgb[x,y-1,color]/2
    else:
        return img_rgb[x,y,color]/3 + img_rgb[x,y+1,color]/3 + img_rgb[x,y-1,color]/3
        
@jit
def highlightseam(img_rgb,seampixellist):
    img_rgb_temp = img_rgb
    for i in range(len(seampixellist)):
        img_rgb_temp[img_rgb.shape[0]-i-1,seampixellist[i]] = [0,0,255]
    return img_rgb_temp

filenames = os.listdir('images')
img_rgb = cv2.imread('images/fig8.png')
#img_rgb = cv2.imread('images/fig8enlargedstep1.png')
img_rgb_primary = img_rgb
delta_x = 120

seams = []
pixelcoordinates = []
img_gray_primary = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
for i in range(delta_x):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_rgb = img_rgb.astype("float64")
    img_energy = l1_energy_function(x_gradient_rgb(img_rgb),y_gradient_rgb(img_rgb))
    matrix_energy_vertical = np.zeros(shape = (img_rgb.shape[0],img_rgb.shape[1],2), dtype = "float")
    matrix_energy_vertical = cumulativeenergy(img_energy,matrix_energy_vertical)
    seampixellist = lowestcumulativeenergyseam(matrix_energy_vertical)
    pixelcoordinates.append(seampixellist)
    img_rgb = removeverticalseam(img_rgb,seampixellist)

img_rgb_insertion = imageenlarge(pixelcoordinates,img_rgb_primary)
    

cv2.imwrite('images/fig8enlargedstep1blurred.png',img_rgb_insertion)

img_rgb = cv2.imread('images/fig8enlargedstep1blurred.png')
#img_rgb = cv2.imread('images/fig8enlargedstep1.png')
img_rgb_primary = img_rgb
delta_x = 120

seams = []
pixelcoordinates = []
img_gray_primary = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
for i in range(delta_x):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_rgb = img_rgb.astype("float64")
    img_energy = l1_energy_function(x_gradient_rgb(img_rgb),y_gradient_rgb(img_rgb))
    matrix_energy_vertical = np.zeros(shape = (img_rgb.shape[0],img_rgb.shape[1],2), dtype = "float")
    matrix_energy_vertical = cumulativeenergy(img_energy,matrix_energy_vertical)
    seampixellist = lowestcumulativeenergyseam(matrix_energy_vertical)
    pixelcoordinates.append(seampixellist)
    img_rgb = removeverticalseam(img_rgb,seampixellist)

img_rgb_insertion = imageenlarge(pixelcoordinates,img_rgb_primary)
    

cv2.imwrite('images/fig8enlargedstep2blurred.png',img_rgb_insertion)





# image_entropy = entropyenergy(img_rgb_primary) 
# max_value = np.amax(image_entropy)
# min_value = np.amin(image_entropy)

#image_entropy = np.multiply(image_entropy,1/(max_value-min_value))
#cv2.imshow('entropy',image_entropy)
# cv2.imshow('image color primary',img_rgb_primary)
# cv2.imshow('image color',img_rgb)
# cv2.imshow('image color insertion',img_rgb_insertion)
# cv2.imshow('image gray',img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()