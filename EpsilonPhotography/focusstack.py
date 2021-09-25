
# coding: utf-8

# In[133]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

import cv2
import os
import glob
import numpy as np
import scipy as sp


import matplotlib as mpl
import matplotlib.pyplot as plt


# In[134]:


## Depth of Field Laplacian operator function ##

def depthoffield(imagelist,kernel):
    gradient_kernel = kernel
    gradientimagelist = []
    for x in imagelist:
        gradientimagelist.append(cv2.Laplacian(x, cv2.CV_64F, ksize = gradient_kernel))
    return gradientimagelist


# In[135]:


## Depth of Field Sobel operator function ##

def depthoffieldsobel(imagelist,kernel):
    sobel_kernel = kernel
    sobelxgradientimagelist = []
    sobelygradientimagelist = []
    gradientimagelist = []
    for x in imagelist:
        xgrad = cv2.Sobel(x, cv2.CV_64F, 1,0,ksize = sobel_kernel)
        ygrad = cv2.Sobel(x, cv2.CV_64F, 0,1,ksize = sobel_kernel)
        sobelxgradientimagelist.append(xgrad)
        sobelygradientimagelist.append(ygrad)
        gradientimagelist.append((np.sqrt(xgrad*xgrad+ygrad*ygrad)))
    return gradientimagelist


# In[136]:


## Mask to identify highest gradient magnitude pixels for a region amongst all images ##

def makingmask(imagelist):
    imagelist = np.absolute(imagelist)
    maxpixels = imagelist.max(axis=0)
    mask = imagelist == maxpixels
    mask = mask.astype(np.uint8)
    return mask


# In[137]:


## Gray Scale Focal stacking ##
## Not used in for final result. To test algorithm on single channel##
def grayfocusstack(imagelist,masklist):
    focuspiclist = []
    for x,y in zip(imagelist,masklist):
        focuspiclist.append(np.multiply(x,y))


# In[138]:


## Function using the masks for isolating highest gradient areas in the color pictures ##

def colorstacks(imagelist,masklist):
    for x,y in zip(imagelist,masklist):
        x[:,:,0] = np.multiply(x[:,:,0],y)
        x[:,:,1] = np.multiply(x[:,:,1],y)
        x[:,:,2] = np.multiply(x[:,:,2],y)
    return imagelist


# In[139]:


## Function merges the images to form the higher depth of field composite ##

def colorfocusstack(imagelist):
    colorfocusstack = imagelist[0]*0
    for x in imagelist:
        colorfocusstack = colorfocusstack + x
    return colorfocusstack


# In[140]:


def transformingimages(a,b,c,d):
    refimage =  a
    alignimage =  c
    refimagegray = b
    alignimagegray = d
 
    motionmodel = cv2.MOTION_TRANSLATION
    motionmatrix = np.eye(2, 3, dtype=np.float32)
    (constant, motionmatrix) = cv2.findTransformECC(refimagegray,alignimagegray,motionmatrix, motionmodel, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 4000,  1e-10))
    alignedimage = cv2.warpAffine(alignimage, motionmatrix, (refimage.shape[1],refimage.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
   
    return alignedimage


# In[ ]:


## Calling image alignment function ##

imagelist = []
for x in glob.glob("*.JPG"):
#for x in glob.glob("/Users/nitishsanghi/assignments/A2-Epsilon_Photography/dataimages/*.JPG"):        
    imagelist.append(cv2.imread(x,cv2.IMREAD_COLOR))

grayimagelist = []
for x in imagelist:
    grayimagelist.append(cv2.cvtColor(x,cv2.COLOR_BGR2GRAY))

    i = 0
for x,y in zip(imagelist,grayimagelist):
    filename = 'image_transformed_'+str(i)+'.jpg'
    i +=1
    cv2.imwrite(filename,transformingimages(imagelist[0],grayimagelist[0],x,y))


# In[ ]:


## Calling main function ##

z = 31 ## Kernel for Gradient ##
y = 31 ## Kernel for Blurring ##

## Reading transformed/aligned images ##
imagelist = []
for x in glob.glob("*.jpg"):        
#for x in glob.glob("/Users/nitishsanghi/assignments/A2-Epsilon_Photography/transformedimagesraspi/*.jpg"):        
    imagelist.append(cv2.imread(x,cv2.IMREAD_COLOR))

## Converting to gray scale for gradient ##
grayimagelist = []
for x in imagelist:
    grayimagelist.append(cv2.cvtColor(x,cv2.COLOR_BGR2GRAY))

## Blurring to smoothen and reduce noise ##   
blur_kernel = y
blurimagelist = []
for x in grayimagelist:
    blurimagelist.append(cv2.GaussianBlur(x, (blur_kernel, blur_kernel),0))

## Calling nested function to create composite ##    
filename = 'colorfocussed'+str(y)+'_grad_'+str(z)+'.jpg'
cv2.imwrite(filename,colorfocusstack(colorstacks(imagelist,makingmask(depthoffield(blurimagelist,z)))))

