import cv2
print("is this working")
vidcap = cv2.VideoCapture('billiards.mp4')
success,image = vidcap.read()
count = 0
while success:
  success,image = vidcap.read()
  height , width , layers =  image.shape
  new_h=height/2
  new_w=width/2
  resize = cv2.resize(image, (new_w, new_h)) 
  print('Read a new frame: ', success)
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
  count += 1