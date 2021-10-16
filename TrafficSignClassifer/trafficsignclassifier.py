# Load pickled data
import pickle
import matplotlib.pyplot as plt
import matplotlib.axes as axs
from random import randint
from numpy import*
import skimage.io
from skimage.transform import*
from sklearn.utils import shuffle
import cv2
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from time import*




#Reading datasets from file
training_file = "../train.p"
validation_file="../valid.p"
testing_file = "../test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

n_train = X_train.shape[0]
n_validation = X_valid.shape[0]
n_test = X_test.shape[0]

image_shape = X_train[0].shape

n_classes = max(y_train)+1

### Data exploration visualization code 
plt.figure(1)
plt.xlabel('Features')
plt.ylabel('Frequency')
plt.hist(y_train,range(max(y_train+2)), label = 'Train')
plt.hist(y_test,range(max(y_test+2)), label = 'Test')
plt.hist(y_valid,range(max(y_valid+2)), label = 'Valid')
plt.legend()

        
average_intensity = []
for x in X_train:
    average_intensity.append(sum(x)/3072)
plt.figure(2)
plt.xlabel('Features')
plt.ylabel('Frequency')
plt.hist(average_intensity, label = 'Average Pixel Intensity')

#Binning all indexes by classes
y_list = []
for i in range(43):
    temp_list = []
    for x in range(len(y_train)):
        if i == y_train[x]:
            temp_list.append(x)      
    y_list.append((i,temp_list))

# Finding Image brightest or darkest
def pixint(X_train,y_list,n):
    X_list = []
    for x in y_list:
        if n ==1:
            bright = 0
            temp = 0
            for y in x[1]:
                if bright < sum(X_train[y])/1024:
                    temp = y
                    bright = sum(X_train[y])/1024
            X_list.append((x[0],temp))
        else:
            bright = sum(X_train[x[1][0]])/1024
            temp = 0
            for y in x[1]:
                if bright > sum(X_train[y])/1024:
                    temp = y
                    bright = sum(X_train[y])/1024
            X_list.append((x[0],temp))
    return X_list

#Plot all classes of traffic signals Brightest/Darkest images
def plotter(n,X_train,X_list):
    counter = 0
    plt.figure(n,figsize = [20,20])
    for j in range(1,43):
        if counter < 43:
            plt.subplot(8,6,j)
            plt.imshow(X_train[X_list[counter][1]])
            plt.title("Traffic Sign Class:" +' '+ str(X_list[counter][0]))
            counter +=1
        else:
            pass
    plt.tight_layout()

plotter(1,X_train,pixint(X_train,y_list,1))
plotter(2,X_train,pixint(X_train,y_list,0))

# Finding unBalanced training dataset
y_list_ub = []
for x in y_list:
    if len(x[1])<1000:
        y_list_ub.append(x)
        print(str(x[0])+' : '+ str(len(x[1])))
        
print(y_list_ub)

# Balancing Data using Augmentation function

suml = 0
for x in y_list_ub:
    suml += len(x[1])


### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.
Xg_train = zeros(shape=(len(X_train),32,32))
Xge_train = zeros(shape=(len(X_train),32,32))
Xg_test = zeros(shape=(len(X_test),32,32))
Xg_valid = zeros(shape=(len(X_valid),32,32))

for i in range(len(X_train)):
    Xg_train[i] = cv2.cvtColor(X_train[i], cv2.COLOR_RGB2GRAY)
    
for i in range(len(X_test)):
    Xg_test[i] = cv2.cvtColor(X_test[i], cv2.COLOR_RGB2GRAY)
    
for i in range(len(X_valid)):
    Xg_valid[i] = cv2.cvtColor(X_valid[i], cv2.COLOR_RGB2GRAY)

plotter(1,Xg_train,pixint(Xg_train,y_list,1))
plotter(2,Xg_train,pixint(Xg_train,y_list,0))

# Balancing Data using Augmentation function
temp_list = []
temp_list_y = [] 
    

# Noise, Swirl, Rotation and translate
for x in y_list_ub:
    counter = len(x[1])
    while (counter < 1500):
        for y in x[1]:
            if counter ==1500:
                break   
            if random.randint(0,1e10)%2 ==0:
                if random.randint(0,1e10)%2 !=0:
                    temp_list.append(Xg_train[y] + random.rand(32,32)*randint(0,10))
                    temp_list_y.append(y_train[y])
                else:  
                    temp_list.append(swirl(Xg_train[y],rotation=0, strength=1.5*random.rand(), radius=10, mode = 'reflect'))
                    temp_list_y.append(y_train[y])
            else:
                    temp_list.append(roll(roll(Xg_train[y],int(5*random.rand()),axis = 0),int(5*random.rand()),axis =1))
                    temp_list_y.append(y_train[y])
            counter +=1


temp_list = asarray(temp_list)
temp_list_y = asarray(temp_list_y)
print(len(temp_list))
print(Xg_train.shape)
Xg_train = concatenate((Xg_train,temp_list), axis = 0)
y_train = concatenate((y_train,temp_list_y),axis = 0)
print(Xg_train.shape)
plt.hist(y_train,range(max(y_train+2)), label = 'Train')

Xg_train, y_train = shuffle(Xg_train,y_train)


# Augmentation of training dataset



# Finding unBalanced training dataset
y_list_ub = []
for x in y_list:
    if len(x[1])<1000:
        y_list_ub.append(x)
        print(str(x[0])+' : '+ str(len(x[1])))
        
print(y_list_ub)


temp_list = []
temp_list_y = [] 

# Noise and Swirl
for i in range(len(Xg_train)):
    if random.randint(0,1e10)%2 ==0:
        if random.randint(0,1e10)%2 !=0:
            temp_list.append(Xg_train[i] + random.rand(32,32)*randint(0,20))
            temp_list_y.append(y_train[i])
        else:  
            temp_list.append(swirl(Xg_train[i],rotation=0, strength=1.5, radius=10, mode = 'reflect'))
            temp_list_y.append(y_train[i])
    else:
            pass

temp_list = asarray(temp_list)
temp_list_y = asarray(temp_list_y)
        
Xg_train = concatenate((Xg_train,temp_list), axis = 0)
y_train = concatenate((y_train,temp_list_y),axis = 0)
 
temp_list = []
temp_list_y = [] 

Xg_train, y_train = shuffle(Xg_train,y_train)

# Swirl
for i in range(len(Xg_train)):
    if random.randint(0,1e10)%2 ==0:
        if random.randint(0,1e10)%2 !=0:
            pass
        else:  
            temp_list.append(swirl(Xg_train[i],rotation=random.randn()*.1, strength=0, radius=1, mode = 'reflect'))
            temp_list_y.append(y_train[i])
    else:
            pass
        

temp_list = asarray(temp_list)
temp_list_y = asarray(temp_list_y)
Xg_train = concatenate((Xg_train,temp_list), axis = 0)
y_train = concatenate((y_train,temp_list_y),axis = 0)
print(Xg_train.shape)

Xg_train, y_train = shuffle(Xg_train,y_train)

plt.figure(1)
plt.imshow(Xg_train[10000])
plt.figure(2)
plt.imshow(roll(roll(Xg_train[10000],5,axis = 0),5,axis =1))

# Normalizing Grayscale Image

for i in range(len(X_train)):
    Xg_train[i] -=mean(Xg_train[i])
    Xg_train[i] /=std(Xg_train[i])
    
for i in range(len(X_test)):
    Xg_test[i] -=mean(Xg_test[i])
    Xg_test[i] /=std(Xg_test[i])
    
for i in range(len(X_valid)):
    Xg_valid[i] -=mean(Xg_valid[i])
    Xg_valid[i] /=std(Xg_valid[i])

# Reshaping 32x32 to 32x32x1

Xg_train = Xg_train.reshape((len(Xg_train),32,32,1))
Xg_test = Xg_test.reshape((len(X_test),32,32,1))
Xg_valid = Xg_valid.reshape((len(X_valid),32,32,1))
print(Xg_train.shape)
print(Xg_valid.shape)
print(Xg_test.shape)

### Define your architecture here.
### Feel free to use as many code cells as needed.



EPOCHS = 20
BATCH_SIZE = 128



def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = .001
    sigma = .1
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x10.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 10), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(10))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)
    
    # SOLUTION: Layer 2: Convolutional. Input = 28x28x10 Output = 24x24x20.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 10, 20), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(20))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Layer 3: Convolutional. Input = 24x24x20. Output = 20x20x30.
    conv3_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 20, 30), mean = mu, stddev = sigma))
    conv3_b = tf.Variable(tf.zeros(30))
    conv3   = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='VALID') + conv3_b

    # SOLUTION: Activation.
    conv3 = tf.nn.relu(conv3)
    
    # SOLUTION: Pooling. Input = 20x20x30. Output = 10x10x30.
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # SOLUTION: Layer 4: Convolutional. Input = 10x10x30. Output = 6x6x40.
    conv4_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 30, 40), mean = mu, stddev = sigma))
    conv4_b = tf.Variable(tf.zeros(40))
    conv4   = tf.nn.conv2d(conv3, conv4_W, strides=[1, 1, 1, 1], padding='VALID') + conv4_b

    # SOLUTION: Activation.
    conv4 = tf.nn.relu(conv4)
    
    # SOLUTION: Pooling. Input = 6x6x40. Output = 3x3x40.
    conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # SOLUTION: Flatten. Input = 3x3x40. Output = 360.
    fc0   = flatten(conv4)
    
    # SOLUTION: Layer 5: Fully Connected. Input = 384. Output = 192.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(360, 180), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(180))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)

    # SOLUTION: Layer 6: Fully Connected. Input = 192. Output = 96.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(180, 90), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(90))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)
    
    # SOLUTION: Layer 7: Fully Connected. Input = 96. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(90, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)

one_hot_y = tf.one_hot(y, 43)

rate = 0.002

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.


loss_epoch = []
loss_batch = []
rate_l = []
valid_acc = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(Xg_train)
    print("Training...")
    last_time = time()
    print(int(last_time))
    print()
    for i in range(EPOCHS):
        rate_l.append(rate)
        Xg_train, y_train = shuffle(Xg_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = Xg_train[offset:end], y_train[offset:end]
            _, loss_value = sess.run([training_operation,loss_operation], feed_dict={x: batch_x, y: batch_y})
            loss_batch.append(loss_value)
        loss_epoch.append(sum(asarray(loss_batch))/len(loss_batch))
        
        validation_accuracy = evaluate(Xg_valid, y_valid)
        training_accuracy = evaluate(Xg_train, y_valid)
        print("EPOCH {} ...".format(i+1))
        rate = .002 - .000003*i*i
        print(rate)
        valid_acc.append(validation_accuracy)
        now_time = time()
        diff_time = int(now_time - last_time)
        print(diff_time)
        last_time = time()
        plt.plot(EPOCHS,loss_array)
        print("Training Accuracy = {:.3f}".format(training_accuracy))
        print(loss_epoch[len(loss_epoch)-1])
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
    plt.figure(1)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(asarray(loss_epoch))
    plt.figure(2)
    plt.plot(asarray(rate_l))
    plt.figure(3)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(asarray(valid_acc))

    saver.save(sess, './lenet')
    print("Model saved")


print(loss_epoch)
print(rate_l)

plt.figure(2)
plt.plot(asarray(rate_l))
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

#     test_accuracy = evaluate(Xg_test, y_test)
#     print("Test Accuracy = {:.3f}".format(test_accuracy))

# ### Load the images and plot them here.
# ### Feel free to use as many code cells as needed.
import glob

filenames = [img for img in glob.glob("/Users/nitishsanghi/CarND-Traffic-Sign-Classifier-Project/sign*.png")]

images = []
for img in filenames:
    n= cv2.imread(img)
    images.append(n)
    print (img)

X_test_new = []
y_test_new = [3, 11, 25 ,18 , 38]
i = 1
for img in images:
    plt.figure(i)
    img = cv2.resize(img,(32,32))
    X_test_new.append(img)
    plt.imshow(img)
    i +=1
    print(img.shape)
    
print(asarray(X_test_new).shape)




### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.

# Preprocessing

Xg_test_new = zeros(shape = [len(X_test_new),32,32])

for i in range(len(X_test_new)):
    Xg_test_new[i] = cv2.cvtColor(X_test_new[i], cv2.COLOR_RGB2GRAY)
    plt.figure(i)
    plt.imshow(Xg_test_new[i],'gray')
    
print(Xg_test_new.shape)

for i in range(len(X_test_new)):
    Xg_test_new[i] -=mean(Xg_test_new[i])
    Xg_test_new[i] /=std(Xg_test_new[i])

Xg_test_new = Xg_test_new.reshape((len(Xg_test_new),32,32,1))

Xg_test_new.astype(float64)
print(Xg_test_new.shape)



result = tf.placeholder(tf.float32, (5, 43))


with tf.Session() as sess:       
    saver = tf.train.import_meta_graph('./lenet.meta')
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    so = tf.nn.softmax(logits,-1)
    t_e = tf.nn.top_k(so,k = 1)
    result = sess.run(t_e, feed_dict={x: Xg_test_new})
    print(result)



### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.
counter = 0
for i in range(len(y_test_new)):
    if result[1][i] == y_test_new[i]:
        counter +=1
    else:
        pass

accuracy = counter/5.0*100    
print(accuracy)


### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.

with tf.Session() as sess:       
    saver = tf.train.import_meta_graph('./lenet.meta')
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    so = tf.nn.softmax(logits,-1)
    t_e = tf.nn.top_k(so,k = 5)
    result = sess.run(t_e, feed_dict={x: Xg_test_new})
    print(result)