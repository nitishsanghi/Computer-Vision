import numpy as np
import scipy as sp
import cv2

def getImageCorners(image):
    corners = np.zeros((4, 1, 2), dtype=np.float32)
    corners[0][0][0] = 0
    corners[0][0][1] = 0
    corners[1][0][0] = 0
    corners[1][0][1] = image.shape[0]
    corners[2][0][0] = image.shape[1]
    corners[2][0][1] = image.shape[0]
    corners[3][0][0] = image.shape[1]
    corners[3][0][1] = 0

    return corners

def findMatchesBetweenImages(image_1, image_2, num_matches):
    feat_detector = cv2.ORB_create(nfeatures=500)
    image_1_kp, image_1_desc = feat_detector.detectAndCompute(image_1, None)
    image_2_kp, image_2_desc = feat_detector.detectAndCompute(image_2, None)
    bfm = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bfm.match(image_1_desc, image_2_desc),key=lambda x: x.distance)[:num_matches]
    return image_1_kp, image_2_kp, matches

def findHomography(image_1_kp, image_2_kp, matches):
    image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    for i in range(len(matches)):
            image_1_points[i] = image_1_kp[matches[i].queryIdx].pt
            image_2_points[i] = image_2_kp[matches[i].trainIdx].pt  
    M, mask = cv2.findHomography(image_1_points, image_2_points, cv2.RANSAC,5.0)

    return M

def getBoundingCorners(corners_1, corners_2, homography):
    corners_1_homographed = np.zeros((4, 1, 2), dtype=np.float32)
    x_min = 0
    y_min = 0
    for i in range(len(corners_1)):
        x = homography[0][0]*corners_1[i][0][0] + homography[0][1]*corners_1[i][0][1] + homography[0][2]
        y = homography[1][0]*corners_1[i][0][0] + homography[1][1]*corners_1[i][0][1] + homography[1][2]
        w = homography[2][0]*corners_1[i][0][0] + homography[2][1]*corners_1[i][0][1] + homography[2][2]
        corners_1_homographed[i][0][0] = x/w
        corners_1_homographed[i][0][1] = y/w
        if corners_1_homographed[i][0][0]<x_min:
            x_min = corners_1_homographed[i][0][0]
        if corners_2[i][0][0]<x_min:
            x_min = corners_2[i][0][0]
        if corners_1_homographed[i][0][1]<y_min:
            y_min = corners_1_homographed[i][0][1]
        if corners_2[i][0][1]<y_min:
            y_min = corners_2[i][0][1]

    x_max = 0
    y_max = 0
    for i in range(len(corners_1_homographed)):
        if corners_1_homographed[i][0][0]>x_max:
                x_max = corners_1_homographed[i][0][0]
        if corners_2[i][0][0]>x_max:
                x_max = corners_1[i][0][0]
    for i in range(len(corners_1_homographed)):
            if corners_1_homographed[i][0][1]>y_max:
                y_max = corners_1_homographed[i][0][1]
            if corners_2[i][0][1]>y_max:
                y_max = corners_2[i][0][1]
    min_coord = np.zeros((2), dtype=np.float64)
    max_coord = np.zeros((2), dtype=np.float64)
    min_coord[0] = x_min 
    min_coord[1] = y_min
    max_coord[0] = x_max 
    max_coord[1] = y_max
    return min_coord, max_coord

def warpCanvas(image, homography, min_xy, max_xy):
    canvas_size = tuple(np.round(max_xy - min_xy).astype(np.int))
    translation = np.matrix([[1.0, 0.0, -1*min_xy[0]],[0.0, 1.0, -1*min_xy[1]],[0, 0, 1]])
    M_ht = np.matmul(translation,homography)
    image = cv2.warpPerspective(image,M_ht,(canvas_size[0],canvas_size[1]))
    return image

def blendImagePair(image_1, image_2, num_matches):
    kp1, kp2, matches = findMatchesBetweenImages(image_1, image_2, num_matches)
    homography = findHomography(kp1, kp2, matches)
    corners_1 = getImageCorners(image_1)
    corners_2 = getImageCorners(image_2)
    min_xy, max_xy = getBoundingCorners(corners_1, corners_2, homography)
    oimg1 = warpCanvas(image_1, homography, min_xy, max_xy)
    y_0 = int(-min_xy[1])
    y_1 = int(-min_xy[1]) + image_2.shape[0]
    x_0 = int(-min_xy[0])
    x_1 = int(-min_xy[0]) + image_2.shape[1]
    row = y_1 - y_0
    col = x_1 - x_0
    oimg2 = oimg1*0
    output_image = oimg1*0
    oimg2[y_0:y_1, x_0:x_1] = image_2
    oimg1_gray = cv2.cvtColor(oimg1, cv2.COLOR_BGR2GRAY)
    oimg2_gray = cv2.cvtColor(oimg2, cv2.COLOR_BGR2GRAY)
    output_image_gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)

    alpha = 1
    for i in range(oimg1_gray.shape[0]):
        for j in range(oimg1_gray.shape[1]):
            if oimg1_gray[i][j] and oimg2_gray[i][j]:
                output_image_gray[i][j] = 255
            else: 
                output_image_gray[i][j] = 0
    alpha_mask = output_image_gray.astype(float)/255*alpha
    gamma_mask = output_image_gray.astype(float)/255*alpha

    for i in range(alpha_mask.shape[0]):
        indexes = np.nonzero(alpha_mask[i])
        if len(indexes[0])>1:
            scaler = indexes[0][-1]-indexes[0][0]+1
            for j in range(indexes[0][-1]-indexes[0][0]+1):
                alpha_mask[i][indexes[0][0]+j] = 1-1/scaler*j 
    alpha_mask = np.stack((alpha_mask,alpha_mask,alpha_mask),axis=2)   
    gamma_mask = np.stack((gamma_mask,gamma_mask,gamma_mask),axis=2)  
    output_image =  np.multiply(1 - gamma_mask,oimg1) + np.multiply(1 - gamma_mask,oimg2) + np.multiply(alpha_mask,oimg1)  + np.multiply(gamma_mask-alpha_mask,oimg2)
    return output_image.astype(np.uint8)
