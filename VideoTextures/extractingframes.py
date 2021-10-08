import cv2
print("is this working")
vidcap = cv2.VideoCapture('billiards.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("\billiards\frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1