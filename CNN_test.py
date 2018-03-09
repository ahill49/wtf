
# coding: utf-8

# In[1]:


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from scipy.misc import imread
from random import shuffle

import urllib


# Read in bad and good images

# In[2]:


def lat_long_zoom_to_quadkey(x, y, zoom):
    quadKey = ''
    for i in range(zoom, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (x & mask) != 0:
            digit += 1
        if (y & mask) != 0:
            digit += 2
        quadKey += str(digit)
    #print "\nThe quadkey is {}".format(quadKey)
    return quadKey


def read_api_key(filename_api_key):
    try:
        f = open(filename_api_key)
        api_key = f.read()
    except:
        print("Can't read API key file:\n" + filename_api_key)
        return api_key


def quadKey_to_url(quadKey,api_key):
    # Read textfile with Bing Maps API key
    # See: https://msdn.microsoft.com/en-us/library/ff428642.aspx

    # TODO get this into a config file, and set up others (Google, OSM, etc)
    tile_url = ("http://a0.ortho.tiles.virtualearth.net/tiles/a{}.jpeg?"
                "g=854&mkt=en-US&token={}".format(quadKey, api_key))
    #print "\nThe tile URL is: {}".format(tile_url)
    return tile_url


def retrieve_bing_image(tile_url, image_name):
    # retrieve Bing maps image
    urllib.request.urlretrieve(tile_url, image_name)
    if os.path.getsize(image_name) <= 1033:
        os.remove(image_name)
    

def get_train_images(csv_file, filename_api_key, EMPTY_IMAGE=False, IMAGE_LIMIT=100): 
    
    #retrieve bing images
    #default (EMPTY_IMAGE=False) is to get the images with buildings in them
    #to get the 'empty images' set EMPTY_IMAGE=True
    
    #read csv file into pandas df, and api key
    df = pd.read_csv(csv_file)
    api_key = read_api_key(filename_api_key)
    
    #determine whether we are getting the empty images or not
    if EMPTY_IMAGE: 
        bic = 1
        out_dir = 'images/no/'
    else:
        bic = 0
        out_dir = 'images/yes/'
        
    #get task_x, task_y, task_z    
    tasks = df[['task_x', 'task_y','task_z']].loc[df['bad_imagery_count'] == bic].as_matrix()[:IMAGE_LIMIT]
    
    #get the quadkeys
    quadKeys = [lat_long_zoom_to_quadkey(task[0], task[1], task[2]) for task in tasks]
    
    #get tile urls
    tile_urls = [quadKey_to_url(quadKey,api_key) for quadKey in quadKeys]
    
    #get image names
    image_names = [out_dir + str(task[0]) + '-' + str(task[1]) + '.jpg' for task in tasks]
    
    #retrieve images
    [retrieve_bing_image(tile_urls[i], image_names[i]) for i in range(0,len(tile_urls))]


# define other helpful functions for data cleaning 
# 

# In[ ]:





# define filepaths 

# In[3]:


malawi_results = 'results_2020.csv'
filename_api_key = 'api_key.txt'
dir_no = 'images/no/'
dir_yes = 'images/yes/'


# get training images

# In[4]:


#get_train_images(malawi_results, filename_api_key, IMAGE_LIMIT=400)
#get_train_images(malawi_results, filename_api_key, EMPTY_IMAGE=True, IMAGE_LIMIT=400)


# load images and get their data labels

# In[5]:


#get full images list 
yes_names = os.listdir(dir_yes)
no_names = os.listdir(dir_no)

print(len(yes_names) + len(no_names))

#read in images
yes_images = [imread(dir_yes + name) for name in yes_names]
no_images = [imread(dir_no + name) for name in no_names]

#these are important for the NN
IMG_SIZEX, IMG_SIZEY, NLAYERS = yes_images[0].shape

print(yes_images[0].shape)

#make labels for the CNN
yes_labels = [[1,0] for x in range(0,len(yes_images))]
no_labels = [[0,1] for x in range(0,len(no_images))]

#make lists that contain both the image, and its associated label
train_data_yes = [[yes_images[x], yes_labels[x]] for x in range(0,len(yes_labels))]
train_data_no = [[no_images[x], no_labels[x]] for x in range(0,len(no_labels))]

#concatenate the arrays, and shuffle the data
train_data = train_data_yes + train_data_no
shuffle(train_data)


# Simple neural network architecture taken from this blog (https://goo.gl/UbmU1i)

# In[6]:


convnet = input_data(shape=[None, 256, 256, 3], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=0.1, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')


# In[7]:
#Train the network

train = train_data[:700]
test = train_data[700:]

NX, NY = 256, 256

X = np.array([i[0] for i in train]).reshape(-1,NX,NY,3)
Y = [i[1] for i in train]

print(X[0].shape)

test_x = np.array([i[0] for i in test]).reshape(-11,NX,NY,3)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=3, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=99, show_metric=True)#, run_id=MODEL_NAME)

