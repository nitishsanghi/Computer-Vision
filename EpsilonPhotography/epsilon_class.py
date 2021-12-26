### CITE: 
###The code used for this assignment is my original code which I coded for 2019 Spring offering of Computational Photography
### END CITE

import cv2
import os
import glob
import numpy as np
import scipy as sp


import matplotlib as mpl
import matplotlib.pyplot as plt


class preprocess:
    def __init__(self):
        print("Initialize preproces")
        self.imagelist = self.readimages()
        self.grayimagelist = self.grayconvimages(self.imagelist)
        self.transformedlist = self.imagealignment()
        self.blurimagelist = self.blurimages(self.grayconvimages(self.transformedlist))

    def readimages(self):
        print("Raw images read")
        templist = []
        for x in glob.glob("*.JPG"):       
            templist.append(cv2.imread(x,cv2.IMREAD_COLOR))
        return templist

    def grayconvimages(self, imglist):
        print("Gray scale conversion")
        templist = []
        for x in imglist:
            templist.append(cv2.cvtColor(x,cv2.COLOR_BGR2GRAY))
        return templist

    def transformimage(self, refimage, refimagegray, alignimage, alignimagegray):
        print("Image transformation")
        motionmodel = cv2.MOTION_TRANSLATION
        motionmatrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 4000,  1e-10)
        (constant, motionmatrix) = cv2.findTransformECC(refimagegray, alignimagegray, motionmatrix, motionmodel, criteria)
        alignedimage = cv2.warpAffine(alignimage, motionmatrix, (refimage.shape[1],refimage.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return alignedimage

    def imagealignment(self):
        print("Image alignment")
        i = 0
        templist = []
        for x,y in zip(self.imagelist, self.grayimagelist):
            filename = 'image_transformed_'+str(i)+'.jpg'
            i +=1
            image = self.transformimage(self.imagelist[0], self.grayimagelist[0],x,y)
            cv2.imwrite(filename, image)
            templist.append(image)
        return templist

    def blurimages(self, imglist):
        print("Blurring images")
        templist = []
        for x in imglist:
            templist.append(cv2.GaussianBlur(x, (31, 31),0))
        return templist

class epsilon:
    def __init__(self, imglist, blurlist):
        print("Initializing epsilon")
        cv2.imwrite('final_artifact.jpg',self.colorfocusstack(self.colorstacks(imglist,self.makingmask(self.depthoffield(blurlist)))))

    def depthoffield(self, imagelist):
        print("Depth of field laplacian")
        templist = []
        for x in imagelist:
           templist.append(cv2.Laplacian(x, cv2.CV_64F, ksize = 31))
        return templist

    def makingmask(self, imagelist):
        print("Make mask")
        imagelist = np.absolute(imagelist)
        maxpixels = imagelist.max(axis=0)
        mask = imagelist == maxpixels
        mask = mask.astype(np.uint8)
        return mask

    def colorstacks(self, imagelist, masklist):
        print("Stack of color images")
        for x,y in zip(imagelist,masklist):
            x[:,:,0] = np.multiply(x[:,:,0],y)
            x[:,:,1] = np.multiply(x[:,:,1],y)
            x[:,:,2] = np.multiply(x[:,:,2],y)
        return imagelist

    def colorfocusstack(self, imagelist):
        print("Focussed color stack")
        colorfocusstack = imagelist[0]*0
        for x in imagelist:
            colorfocusstack = colorfocusstack + x
        return colorfocusstack

if __name__ == "__main__":
    preprocess_images = preprocess()
    focus_result = epsilon(preprocess_images.imagelist, preprocess_images.blurimagelist)
