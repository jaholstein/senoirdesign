#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 17:04:13 2018

@author: John
"""

import time

import random

from sklearn.metrics import confusion_matrix
from datetime import timedelta

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import datasetmake







# Conv 1 filters
filter_size1 = 3 
num_filters1 = 32

# Conv2 filters
filter_size2 = 3
num_filters2 = 32

# Convol3 filters
filter_size3 = 3
num_filters3 = 64

# Fully-connected
fc_size = 128

# Number of color channels
num_channels = 3

# image dimensions
img_size = 128

# Size of image when flattened
img_size_flat = img_size * img_size * num_channels

# Square images with 128x128 image size
img_shape = (img_size, img_size)

# classes
classes = ['face', 'background']
num_classes = len(classes)

# batch size of images
batch_size = 32

# validation split
validation_size = .15

early_stopping = None  

#Needed Paths
train_path = r'/Users/John/Documents/FaceData/train'
test_path = r'/Users/John/Documents/FaceData/test'
checkpoint_dir = r'/Users/John/Documents/FaceData/models'

data = datasetmake.read_train_sets(train_path, img_size, classes, validation_size=validation_size)
test_images, test_ids = datasetmake.read_test_set(test_path, img_size)


#Print the size of the training set
print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(test_images)))
print("- Validation-set:\t{}".format(len(data.valid.labels)))



def plot_images(images, cls_true, cls_pred=None):
    
    if len(images) == 0:
        print("no images to show")
        return 
    else:
        random_indices = random.sample(range(len(images)), min(len(images), 9))
        
        
    images, cls_true  = zip(*[(images[i], cls_true[i]) for i in random_indices])
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_size, img_size, num_channels))

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
    
    
    
# Get some random images and their labels from the train set.

images, cls_true  = data.train.images, data.train.cls

# Plot the images and labels using our helper-function above.
plot_images(images=images, cls_true=cls_true)



#Function to initialize weights
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

#Function to initialize biases
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


#Build a convolutional layer
def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True):  

    #Reshape for tensorflow api
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    
    conv_weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

      #Convolutional operation with 1x1x1x1 stride and padding
    layer = tf.nn.conv2d(input=input, filter=conv_weights, strides=[1, 1, 1, 1], padding='SAME')

    
    layer += biases

    
    if use_pooling:
        #2x2 maxpooling
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    layer = tf.nn.relu(layer)

   
    #Return layer and filer weights
    return layer, conv_weights



def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    #Get number of features
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
   
    layer_flat = tf.reshape(layer, [-1, num_features])


    # Return both the flattened layer and the number of features.
    return layer_flat, num_features


def new_fc_layer(input, num_inputs, num_outputs, use_relu=True): 

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer



#Variable to store input image in tensor
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')

x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])


#Variables to store output tensor and classifier
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

y_true_cls = tf.argmax(y_true, 1)




#Convolutional layer 1
layer_conv1, weights_conv1 = new_conv_layer(input=x_image, num_input_channels=num_channels, filter_size=filter_size1, num_filters=num_filters1, use_pooling=True)

#Convolutional layer 2
layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1, num_input_channels=num_filters1, filter_size=filter_size2, num_filters=num_filters2, use_pooling=True)

#Convolutional layer 3
layer_conv3, weights_conv3 = new_conv_layer(input=layer_conv2, num_input_channels=num_filters2, filter_size=filter_size3, num_filters=num_filters3, use_pooling=True)

#Flattened layer
layer_flat, num_features = flatten_layer(layer_conv3)


#Fully connected layer
layer_fc1 = new_fc_layer(input=layer_flat, num_inputs=num_features, num_outputs=fc_size, use_relu=True)

#Fully connected layer
layer_fc2 = new_fc_layer(input=layer_fc1, num_inputs=fc_size, num_outputs=num_classes, use_relu=False)


#Predicted percentage
y_pred = tf.nn.softmax(layer_fc2)

#predicted class
y_predicted_class = tf.argmax(y_pred, dimension=1)


cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)



cost = tf.reduce_mean(cross_entropy)




optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)




correct_prediction = tf.equal(y_predicted_class, y_true_cls)


accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#Start a tf session
session = tf.Session()



session.run(tf.global_variables_initializer())


train_batch_size = batch_size


def print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    # Calculate the accuracy on the training-set.
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    message = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"
    print(message.format(epoch + 1, acc, val_acc, val_loss))


# Counter for iterations performed so far.
total_iterations = 0

def optimize_graph(num_iterations):
    global total_iterations

    start_time = time.time()
    
    best_val_loss = float("inf")
    patience = 0

    for i in range(total_iterations,
                   total_iterations + num_iterations):

      
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(train_batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(train_batch_size)


        x_batch = x_batch.reshape(train_batch_size, img_size_flat)
        x_valid_batch = x_valid_batch.reshape(train_batch_size, img_size_flat)

       
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
        
        feed_dict_validate = {x: x_valid_batch,
                              y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_train)
        

        if i % int(data.train.num_examples/batch_size) == 0: 
            val_loss = session.run(cost, feed_dict=feed_dict_validate)
            epoch = int(i / int(data.train.num_examples/batch_size))
            
            print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss)
            
            if early_stopping:    
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                else:
                    patience += 1

                if patience == early_stopping:
                    break

    total_iterations += num_iterations

    end_time = time.time()

    time_dif = end_time - start_time

    print("Time elapsed: " + str(timedelta(seconds=int(round(time_dif)))))

def plot_example_errors(cls_pred, correct):
  

    # Negate the boolean array.
    incorrect = (correct == False)
    

    images = data.valid.images[incorrect]
    
    cls_pred = cls_pred[incorrect]

    cls_true = data.valid.cls[incorrect]
    
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])
    
    
    
    
def plot_confusion_matrix(cls_pred):
   
    cls_true = data.valid.cls
    
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    print(cm)

    plt.matshow(cm)

    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

   
    plt.show()
    
    
    
def print_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    
    num_test = len(data.valid.images)

    
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

   
    i = 0

    while i < num_test - 64:
       
        j = min(i + batch_size, num_test)

        images = data.valid.images[i:j, :].reshape(batch_size, img_size_flat)
        

        labels = data.valid.labels[i:j, :]

        feed_dict = {x: images,
                     y_true: labels}

        cls_pred[i:j] = session.run(y_predicted_class, feed_dict=feed_dict)

        i = j

    cls_true = np.array(data.valid.cls)
    cls_pred = np.array([classes[x] for x in cls_pred]) 

    correct = (cls_true == cls_pred)

    correct_sum = correct.sum()

    acc = float(correct_sum) / num_test

    message = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(message.format(acc, correct_sum, num_test))

    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)    
    
    
    
    
    
    
    
    
optimize_graph(num_iterations=1000)
print_accuracy()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    