import numpy as np
import matplotlib.pyplot as plt
import scipy 
import os
import cv2
import glob


class exposurefusion:
    def __init__(self):
        self.imagelist = self.readimages()
        self.grayimagelist = self.grayconvimages(self.imagelist)
        self.transformlist = self.imagealignment(self.imagelist, self.grayimagelist)
        self.mertensfunction(self.transformlist)

    def readimages(self):
        templist = []
        for x in glob.glob("*.JPG"):
            templist.append(cv2.imread(x,cv2.IMREAD_COLOR))
        return templist


    def grayconvimages(self, imglist):
        templist = []
        for x in imglist:
            templist.append(cv2.cvtColor(x,cv2.COLOR_BGR2GRAY))
        return templist


    def transformingimages(self, refimage, refimagegray, alignimage, alignimagegray):
        motionmodel = cv2.MOTION_TRANSLATION
        motionmatrix = np.eye(2, 3, dtype=np.float32)
        (constant, motionmatrix) = cv2.findTransformECC(refimagegray,alignimagegray,motionmatrix, motionmodel, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 4000,  1e-10))
        alignedimage = cv2.warpAffine(alignimage, motionmatrix, (refimage.shape[1],refimage.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return alignedimage

    def imagealignment(self, imglist, graylist):
        i = 0
        templist = []
        for x,y in zip(imglist,graylist):
            filename = 'image_transformed_'+str(i)+'.jpg'
            i +=1
            image = self.transformingimages(imglist[0],graylist[0],x,y)
            cv2.imwrite(filename,image)
            templist.append(image)
        return templist


    def mertensfunction(self, imglist):
        merge_mertens = cv2.createMergeMertens()
        res_mertens = merge_mertens.process(imglist)
        res_mertens_8bit = np.clip(res_mertens*255, 0, 255).astype('uint8')
        cv2.imwrite("exposure_fusion.jpg", res_mertens_8bit)

if __name__ == "__main__":
    exposure_object = exposurefusion()



