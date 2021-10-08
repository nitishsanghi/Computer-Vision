
# coding: utf-8

# In[1]:


# Module Import List
import cv2
import os
import csv
import matplotlib.pyplot as plt
from numpy import*
from sklearn.model_selection import train_test_split
import sklearn
from PIL import Image
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers import Conv2D, MaxPooling2D, convolutional
from keras.layers import Convolution2D, Cropping2D
from keras.callbacks import EarlyStopping # To interrupt if loss not increasing

get_ipython().magic('matplotlib inline')


# In[2]:


# Samples list is accumulated from the csv file by opening the file and reading each row
samples = []
with open('/home/carnd/CarND-Behavioral-Cloning-P3/data/driving_log_clean.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


# In[3]:


# Samples list is split into training and test/validation sample data
train_samples, validation_samples = train_test_split(samples, test_size = 0.2)


# In[4]:


# Generator used to batch images and train the model faster by avoiding memory overlaod
def generator(samples, batch_size = 32):
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

# Lists for collecting image and angles
            images = []
            angles = []
            for batch_sample in batch_samples:

#Center image and angle information extracted
                name_center = '/home/carnd/CarND-Behavioral-Cloning-P3/data/CIMG/' + batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name_center)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                center_image_f = cv2.flip(center_image,1)
                center_angle_f = center_angle*-1.0
                images.append(center_image_f)
                angles.append(center_angle_f)
                
#Left image and angle information extracted                
                name_left = '/home/carnd/CarND-Behavioral-Cloning-P3/data/CIMG/' + batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name_left)
                left_angle = float(batch_sample[3])+0.1
                images.append(left_image)
                angles.append(left_angle)
                left_image_f = cv2.flip(left_image,1)
                left_angle_f = left_angle*-1.0
                images.append(left_image_f)
                angles.append(left_angle_f)
                
#Right image and angle information extracted                
                name_right = '/home/carnd/CarND-Behavioral-Cloning-P3/data/CIMG/' + batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name_right)
                right_angle = float(batch_sample[3])-0.1
                images.append(right_image)
                angles.append(right_angle)
                right_image_f = cv2.flip(right_image,1)
                right_angle_f = right_angle*-1.0
                images.append(right_image_f)
                angles.append(right_angle_f)

# Training data in each batch has 96 samples, 3 times the batch size due 
# to addition of left and right images
            X_train = array(images)
            y_train = array(angles)
            yield sklearn.utils.shuffle(X_train,y_train)


# In[5]:


# Batch size defined
batch_size = 32

# training and validation generators defined for the fit_generator()
train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)


# In[6]:


#NVIDIA CNN used with cropping, mean centralization and normalization
model = Sequential()
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x/255.0) -0.5,input_shape=(55,320,3)))
model.add(Conv2D(24,(5,5),strides = 2,activation='relu'))
model.add(Conv2D(36,(5,5),strides = 2,activation='relu'))
model.add(Conv2D(48,(5,5),strides = 2,activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
#model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
history_object = model.fit_generator(train_generator, steps_per_epoch = len(train_samples)/batch_size, validation_data = validation_generator, validation_steps = len(validation_samples)/batch_size,epochs=3,verbose = 1)

print(history_object.history.keys())
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')


# In[ ]:




