
# coding: utf-8

# In[1]:


# Importing all required references
import os
import re
import sys
import tarfile
import pickle
import numpy as np
from six.moves import urllib
from time import time
from time import sleep
import tensorflow as tf
import math


# In[2]:


data_dir = "./data/cifar-10-batches-py/"


# In[4]:


# Function to convert classes into One Hot Labels
def dense_to_one_hot(labels_dense, num_classes=10):
    
    num_labels = labels_dense.shape[0] #Get total Number of Records
    index_offset = np.arange(num_labels) * num_classes # Get an numpy array  into Index Offset from 0 to total number of records
    labels_one_hot = np.zeros((num_labels, num_classes)) # Create an Numpy array of Zeros with shape noOfRecords and noOfClass  
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1 #Use Flat function to form One Hot Label Encoder

    return labels_one_hot


# In[26]:


# Function  to get the Raw Data for Training and Validation
def get_data_set(name="train"):
    
#     Create Empty List which has to be returned
    X = []
    Y = []
    
    #Get all filenames from the Data folder
    filenames = [os.path.join(data_dir, 'data_batch_%d' % i) for i in range(1, 6)]
    #Get Data for training
    if name== "train":
        #Loop through All filenames
        for filename in filenames:
            file = open(filename,'rb') #Open the File in read only and binary format
            dicts =pickle.load(file, encoding='latin1') #Load data into dicts by encosing using 'latin1'
            file.close() #Close the file
            x = dicts["data"] #Read the Data
            y = dicts["labels"] #Read the labels
            #Convert Data into Numpy Array by dividing all values by 255. 255 because the RGB format is 255 X 255 X 255
            x = np.array(x, dtype=float) / 255.0  
            x = x.reshape([-1, 3, 32, 32]) # Reshape to 3,32,32 array. Input data is 32x32x3
            x = x.transpose([0, 2, 3, 1]) # Transpose the data
            x = x.reshape(-1, 32*32*3) # Reshape back to size of input
    
            #Save the Data to the X,Y
            if len(X) == 0:
                X = x
                Y = y
            else:
                X = np.concatenate((X, x), axis=0)
                Y = np.concatenate((Y, y), axis=0)
                
    #Get Data for Validation/Test. Similar process to get data from train data.
    elif name is "test":
        
        f = open(data_dir+'/test_batch', 'rb')
        dicts = pickle.load(f, encoding='latin1')
        f.close()
        X = dicts["data"]
        Y = np.array(dicts["labels"])
        X = np.array(X, dtype=float) / 255.0
        X = X.reshape([-1, 3, 32, 32])
        X = X.transpose([0, 2, 3, 1])
        X = X.reshape(-1, 32*32*3)
                      
    return X, dense_to_one_hot(Y) #Return X value and One Hot Encoded Y value
     
      


# In[3]:


# Read Data in to Train and Test Variables
train_x, train_y = get_data_set("train")
test_x, test_y = get_data_set("test")
print("Reading Data Completed")


# In[45]:


tf.reset_default_graph() # Reset the Graphs


# In[4]:


def model():
    """
    Create the Model for the CNN Image recongnization. AlexNet kind of architecture is followed.
    
    """
    # Get the Image Size, Channels and Num of Classes
    _IMAGE_SIZE = 32
    _IMAGE_CHANNELS = 3
    _NUM_CLASSES = 10

    
    # Create Placeholder for the variables
    with tf.name_scope('main_params'):
        x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS], name='Input')
        y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
        x_image = tf.reshape(x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images') #Reshape the inputs

#         global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

    # Create the First Convolutional Layer
    with tf.variable_scope('conv1') as scope:
        conv = tf.layers.conv2d(
            inputs=x_image,
            filters=32,
            kernel_size=[3, 3],
            padding='SAME',
            activation=tf.nn.relu # RelU activation used
        )
        conv = tf.layers.conv2d(
            inputs=conv,
            filters=64,
            kernel_size=[3, 3],
            padding='SAME',
            activation=tf.nn.relu # RelU activation used
        )
        #Create an pooling Layer
        pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='SAME')
        #Create a dropout layer - to reduce Overfitting
        drop = tf.layers.dropout(pool, rate=0.25, name=scope.name)

    # Create the second Convolutiona Layer
    with tf.variable_scope('conv2') as scope:
        conv = tf.layers.conv2d(
            inputs=drop,
            filters=128,
            kernel_size=[3, 3],
            padding='SAME',
            activation=tf.nn.relu # RelU activation used
        )
        # Pooling layer with SAME Padding
        pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='SAME')
        conv = tf.layers.conv2d(
            inputs=pool,
            filters=128,
            kernel_size=[2, 2],
            padding='SAME',
            activation=tf.nn.relu # RelU activation used
        )
        # Pooling Layer with 2x2 stride with SAME padding
        pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='SAME')
        drop = tf.layers.dropout(pool, rate=0.25, name=scope.name)

    # Create a Fully Connecgted Dense Layer
    with tf.variable_scope('fully_connected') as scope:
        flat = tf.reshape(drop, [-1, 4 * 4 * 128])

        fc = tf.layers.dense(inputs=flat, units=1500, activation=tf.nn.relu)
        drop = tf.layers.dropout(fc, rate=0.5)
        
        #Create Softmax layer for the output Layer
        softmax = tf.layers.dense(inputs=drop, units=_NUM_CLASSES, activation=tf.nn.softmax, name=scope.name)

    # Predict the maximum of the Softmax as one of the 10 classes
    y_pred_cls = tf.argmax(softmax, axis=1)

    return x, y, softmax, y_pred_cls, learning_rate


# In[47]:


# Get the Learning Rate for the Training based on the Epoch counts
def lr(epoch):
    learning_rate = 1e-3
    if epoch > 80:
        learning_rate *= 0.5e-3
    elif epoch > 60:
        learning_rate *= 1e-3
    elif epoch > 40:
        learning_rate *= 1e-2
    elif epoch > 20:
        learning_rate *= 1e-1
    return learning_rate


# In[48]:


#Hyper Parameters
EPOCHS = 60
BATCH_SIZE = 64
SAVER_PATH = "./saved_model/"


# In[49]:


# This Cell to create  Model, Create Loss, Optimize to update weights, and Identify the Prediction Accuracy
X,Y,logits, y_pred_class,learning_rate = model()

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)) # Get the Loss
# minimize loss using Adam Optimizer
optimizer  = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                   beta1=0.9,
                                   beta2=0.999,
                                   epsilon=1e-08).minimize(cost)

#Initialize the tensor flow variables
init=tf.global_variables_initializer()

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[52]:


#Create the Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[57]:


# TRAIN THE MODEL WITH A PREDEFINED EPOCHS

#Identify the batch count using the Batch Size
batches = int(math.ceil(len(train_x)/BATCH_SIZE))

for epoch in range(EPOCHS): # Loop through for the EPOCHS
    
    start_time = time()
    
    for batch in range(batches): # Loop through the Batches
        batch_x = train_x[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE] # Get batch X Data
        batch_y = train_y[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE] # Get batch Y Data
        
        batch_loss, _, batch_acc = sess.run([cost, optimizer,  accuracy],
            feed_dict={X: batch_x, Y: batch_y, learning_rate: lr(epoch)}) # Get the batch Loss and accuracy (Basically Training)
        
    duration = time()-start_time

    #on Each EPOCH Completion display the message
    print("\nEpoch {} Completed in  {} minutes".format((epoch+1),duration/60))

    # Do the validation accuracy with Test Data
    
    i=0
    predicted_class=[]
    
    # Loop and validate all test Data
    while (i < len(test_x)):
        
        j = min(i + BATCH_SIZE, len(test_x))
        batch_test_x = test_x[i:j, :]
        batch_test_y = test_y[i:j, :]
        
        # Get the Model Prediction
        predicted_class[i:j] = sess.run(y_pred_class,feed_dict={X: batch_test_x, Y: batch_test_y, learning_rate: lr(epoch)})
        i =j
    
    # Identify the accuracy
    correct= (np.argmax(test_y, axis=1) == predicted_class)
    valid_acc = correct.mean()*100
    correct_validated = correct.sum()
    
    #Print Validation accuracy after each EPOCHS
    mes = "\nValidation Result for Epoch {} - accuracy: {:.2f}% ({}/{})"
    print(mes.format((epoch+1), valid_acc, correct_validated, len(test_x)))

    
    
    
        
    


# # With System Limitation, the Model took more than 25 hrs to complete
# NO OF EPOCHS RUN: 60
# VALIDATION ACCURACY = 76.36%
