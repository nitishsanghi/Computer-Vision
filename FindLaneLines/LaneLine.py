import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os

from moviepy.editor import VideoFileClip
from IPython.display import HTML


#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')

def grayscale(img):
	return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
def BGRtoHSV(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)   
    
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
 
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    hough_lines.lines = lines
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, a=0.8, b=1., c=0.):
    return cv2.addWeighted(initial_img, a, img, b, c)

def average_lines(lines,y_shape):
    #containers for slopes and coordinates of detected hough lines
    m_right = []
    x_right = []
    y_right = []
    m_left = []
    x_left = []
    y_left = []
    #Sorting out lines as right and left
    for line in lines:
        x1 = line[0,0]
        x2 = line[0,2]
        y1 = line[0,1]
        y2 = line[0,3]
        #avoiding NaN scenario
        if (x2-x1)==0:
            pass
        else:
            m1 = float(y2-y1)
            m2 = float(x2-x1)
            if (.4<(m1/m2)<.7):
                m_right.append((m1/m2))
                x_right.append(x1)
                x_right.append(x2)
                y_right.append(y1)
                y_right.append(y2)
            if (-1<(m1/m2)<-.6):
                m_left.append((m1/m2))
                x_left.append(x1)
                y_left.append(y1)
                x_left.append(x2)
                y_left.append(y2)
            else:
                pass
    #Finding mean slopes for right and left lane lines            
    m_r = np.mean(m_right)
    m_l = np.mean(m_left)
    
    #Finding coordinates to draw solid lines in ROI
    x2_right = np.amin(x_right)
    y2_right = np.amin(y_right)
    y1_right = y_shape
    x1_right = int(x2_right-(y2_right - y1_right)/m_r)
    y2_right = 325
    x2_right = abs(int(x1_right-(y1_right - y2_right)/m_r))
    #Finding coordinates to draw solid lines in ROI
    x2_left = np.amax(x_left)
    y2_left = np.amin(y_left)
    y1_left = y_shape
    x1_left = int(x2_left-(y2_left - y1_left)/m_l)
    y2_left = 325
    x2_left = abs(int(x1_left-(y1_left - y2_left)/m_l))
        
    a = [[x1_right, y1_right, x2_right, y2_right],[x1_left, y1_left, x2_left, y2_left]]
    return a
    
#Custom draw line function for drawing solidlines
def draw_lines_cus(img, lines, color=[255, 255, 255], thickness=5):
    for line in lines:
            cv2.line(img, (line[0], line[1]), (line[2], line[3]), color, thickness)


os.listdir("test_images/")

#Reading images
image = mpimg.imread('test_images/whiteCarLaneSwitch.jpg')

#Gaussian Blurring
image_blur =  gaussian_blur(image,3)
#plt.imshow(image_blur)

#Conversion to Grayscale
image_gray = grayscale(image_blur)
plt.imshow(image_gray)

#Vertices determined for internal and external polygonal masking
vertices = np.array([[(125, 539),(440,325),(535,325),(890,539)]])
#vertices_neg = np.array([[(200, 539),(470,325),(490,325),(820,539)]])

#Determining Region of Interest of gray image
image_roi = region_of_interest(image_gray,vertices)
plt.imshow(image_roi)

#Using Canny Edge Detector
image_canny = canny(image_gray, 100,200)
plt.imshow(image_canny)

#Determining Region of Interest
image_roi = region_of_interest(image_canny,vertices)
plt.imshow(image_roi)

#Finding Hough Lines
image_hough = hough_lines(image_roi,1,np.pi/180,10,10,10)
plt.imshow(image_hough)

#Merging images
pre_final_image = weighted_img(image_hough,image,.8,1,0)
plt.imshow(image_hough)


#Null image to draw lines and drawing lines
line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
draw_lines_cus(line_img, average_lines(hough_lines.lines,image.shape[0]), color=[255, 0, 0], thickness=10)
plt.imshow(line_img)

#Final image generated by adding weighted null and original image
final_image = weighted_img(line_img,image,1,.5,0)
plt.imshow(final_image)
cv2.imwrite("test.jpeg", final_image)

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes

        #Gaussian Blurring
    image_blur =  gaussian_blur(image,3)
    #plt.imshow(image_blur)

    #Conversion to Grayscale
    image_gray = grayscale(image_blur)
    plt.imshow(image_gray)

    #Vertices determined for internal and external polygonal masking
    vertices = np.array([[(125, 539),(440,325),(535,325),(890,539)]])
    #vertices_neg = np.array([[(200, 539),(470,325),(490,325),(820,539)]])

    #Determining Region of Interest of gray image
    image_roi = region_of_interest(image_gray,vertices)
    plt.imshow(image_roi)

    #Using Canny Edge Detector
    image_canny = canny(image_gray, 100,200)
    plt.imshow(image_canny)

    #Determining Region of Interest
    image_roi = region_of_interest(image_canny,vertices)
    plt.imshow(image_roi)

    #Finding Hough Lines
    image_hough = hough_lines(image_roi,1,np.pi/180,10,10,10)
    plt.imshow(image_hough)

    #Merging images
    pre_final_image = weighted_img(image_hough,image,.8,1,0)
    plt.imshow(image_hough)


    #Null image to draw lines and drawing lines
    line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    draw_lines_cus(line_img, average_lines(hough_lines.lines,image.shape[0]), color=[255, 0, 0], thickness=10)
    plt.imshow(line_img)

    #Final image generated by adding weighted null and original image
    final_image = weighted_img(line_img,image,1,.5,0)
    plt.imshow(final_image)
    result = final_image
    return result

# white_output = 'white.mp4'
# clip1 = VideoFileClip("solidWhiteRight.mp4")
# white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
# white_clip.write_videofile(white_output, audio=False)

# yellow_output = 'yellow.mp4'
# clip2 = VideoFileClip('solidYellowLeft.mp4')
# yellow_clip = clip2.fl_image(process_image)
# yellow_clip.write_videofile(yellow_output, audio=False)

# challenge_output = 'extra.mp4'
# clip2 = VideoFileClip('challenge.mp4')
# challenge_clip = clip2.fl_image(process_image)
# %time challenge_clip.write_videofile(challenge_output, audio=False)