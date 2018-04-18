#! /usr/bin/python

# This is version of exp1/DL_GPX.py from DeepVGI-0.3 
# Code has been re-written and stripped down in attempt to understand DeepVGI-0.3
#
# WARNING: WORK IN PROGRESS, CODE CRASHES AT NN_model.train(nn)

import os
import sys
import csv
import gc
import random
import numpy as np
from scipy import misc

# Need to be in /exp1 for loading local lib functions
if not os.getcwd().endswith('exp1'):
    os.chdir('exp1')

# Import local library functions
sys.path.append("../lib")
import NN_Model
#import sample_client
#import FileIO
#import Parameters

# Reading inout data could be done with lib functions, but...
# Put everything in main for now
if __name__ == '__main__':
    
    # Set the parameters here instead of using lib/Parameters 
    # Using defauls values from lib/Parameters
    
    # evaluate_only = False # Don't need this flag
    tr_n1 = 200 # Number of positive training sample images
    tr_n0 = 200 # number of negative training sample images
    
    nn = 'lenet'    # network model
    tr_b = 30      # Batch size for NN
    tr_e = 500    # epoch number for NN (default 1000)
    tr_t = 8       # number of threads for NN
   
    te_z = 1000    # test size (also called te_n)
    
    # print 'command -v -y <p_sample_size> -n <n_sample_size> -b <batch_size> -e <epoch_num> -t <thread_num>, ' \         '-z <test_size>, -m <network_model>'
    print 'default settings: n1=%d, n0=%d, b=%d, e=%d, t=%d, z=%d, m=%s' % (tr_n1, tr_n0, tr_b, tr_e, tr_t, te_z, nn)
         
    #################################################
    # Here we replace following original function call
    # -> img_X, Y = read_train_sample(tr_n1, tr_n0)
    #################################################
    
    # Need lists of filenames with positive and negative images
    # derived from file lists rather than directories of pre-filtered images 
    # i.e. avoiding '../samples0/train/MS_record' and '../samples0/train/MS_negative'

    # Use lists of MapSwipe results (which do not list negative tiles i.e. ('yes/maybe/bad', but not 'no')
    # and lists of all tiles in MapSwipe task to derive positive/negative lists
    filename_MS_results = '../data/2020_Malawi_2/results_2020.csv'
    filename_all_tiles = '../data/2020_Malawi_2/tiles_p2020_lvl18.csv'

    # Read MS results
    lines = csv.DictReader(open(filename_MS_results))
    p_imgs_raw = []
    for line in lines:
        # Only collect high-confidence positive images (not uncertain/bad)
        maybe_plus_bad_count = (int(line['maybe_count']) + int(line['bad_imagery_count']))
        if int(line['yes_count']) < 2 or int(line['yes_count']) < maybe_plus_bad_count:   
            continue
        
        task_x = line['task_x']
        task_y = line['task_y']
        img = '%s-%s-all.jpeg' % (task_x, task_y)
        p_imgs_raw.append(img)
        
    p_imgs = list(set(p_imgs_raw))

    # Read all tile names
    lines = csv.DictReader(open(filename_all_tiles), delimiter='\t')  # tab separated file
    all_imgs_raw = []
    for line in lines:
        task_x = line['x']
        task_y = line['y']
        img = '%s-%s-all.jpeg' % (task_x, task_y)
        all_imgs_raw.append(img)
    all_imgs = list(set(all_imgs_raw))

    # Negative image set contains elements in all_imgs not in p_imags
    n_imgs = list(set(all_imgs) - set(p_imgs))    
    
    print '# tiles=%i' % (len(all_imgs))
    print '# positives=%i' % (len(p_imgs))
    print '# negatives=%i' % (len(n_imgs))
    
    # Prepare feature data
    img_X1, img_X0 = np.zeros((tr_n1, 256, 256, 3)), np.zeros((tr_n0, 256, 256, 3))
    label = np.zeros((tr_n1 + tr_n0, 2))
    img_dir = '../data/2020_Malawi_2/images/'

    # Select random sample positive images and read pixel values
    p_imgs = random.sample(p_imgs, tr_n1)
    for i, img in enumerate(p_imgs):
        img_X1[i] = misc.imread(os.path.join(img_dir, img))
    label[0:tr_n1, 1] = 1

    # Do same for negative images
    n_imgs = random.sample(n_imgs, tr_n0)
    for i, img in enumerate(n_imgs):
        img_X0[i] = misc.imread(os.path.join(img_dir, img))
    label[tr_n1:(tr_n1 + tr_n0), 0] = 1

    # Randomise order of features and labels
    rndm_idx = range(tr_n1 + tr_n0)
    random.shuffle(rndm_idx)
    img_X = np.concatenate((img_X1, img_X0))
    img_X, label = img_X[rndm_idx], label[rndm_idx]

    # For debug, can show images
    # from matplotlib import pyplot as plt
    # plt.imshow(img_X[1,:,:,:])
    # plt.show()
    
    MS_model = NN_Model.Model(img_X, label, nn + '_ZYX')
    
    if True:
        MS_model.set_batch_size(tr_b)
        MS_model.set_epoch_num(tr_e)
        MS_model.set_thread_num(tr_t)
        
        print '--------------- Training NN ---------------'

        MS_model.train(nn)
    
    print '--------------- Evaluation on Training Samples ---------------'
    
    MS_model.evaluate()
    del img_X, label
    #gc.collect()

    ### CODE BELOW IS UNCHANGED ORIGINAL CODE SNIPPET

    print '--------------- Evaluation on Validation Samples ---------------'
    #img_X2, Y2 = FileIO.read_gRoad_valid_sample(te_n)
    #MS_model.set_evaluation_input(img_X2, Y2)
    #MS_model.evaluate()
    #del img_X2, Y2
    #gc.collect()
