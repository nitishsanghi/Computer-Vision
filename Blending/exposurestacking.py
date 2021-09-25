import numpy as np
import matplotlib.pyplot as plt
import scipy 
import os
import cv2


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




filelist = os.listdir("/Users/nitishsanghi/Assignments/A4-Blending/aboveandbeyond")
imagelist = []
grayimagelist = []
for x in filelist[1:]:
	filepath = os.path.join("/Users/nitishsanghi/Assignments/A4-Blending/aboveandbeyond",x)
	imagelist.append(cv2.imread(filepath,1))


for x in imagelist:
    grayimagelist.append(cv2.cvtColor(x,cv2.COLOR_BGR2GRAY))

    i = 0
for x,y in zip(imagelist,grayimagelist):
    filename = 'image_transformed_'+str(i)+'.jpg'
    i +=1
    cv2.imwrite(filename,transformingimages(imagelist[0],grayimagelist[0],x,y))


merge_mertens = cv2.createMergeMertens()
res_mertens = merge_mertens.process(imagelist)
res_mertens_8bit = np.clip(res_mertens*255, 0, 255).astype('uint8')
cv2.imwrite("exposure_fusion.jpg", res_mertens_8bit)




