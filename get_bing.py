#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 21:36:50 2018

@author: A. Thean, Jiaojan Chen

Comment:
    Reusing functions from DeepVGI-0.3, get_image.py (Jiaojan Chen)
    
"""
import os
import csv
import urllib

# Transform tile coords to quadkeys
# Each quadkey uniquely identifies a single tile at a particular level of detaill
# See https://msdn.microsoft.com/en-us/library/bb259689.aspx
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
        print "Can't read API key file:\n" + filename_api_key
        return api_key

def quadKey_to_url(quadKey,api_key):
    # Read textfile with Bing Maps API key
    # See: https://msdn.microsoft.com/en-us/library/ff428642.aspx

    # TODO get this into a config file, and set up others (Google, OSM, etc)
    tile_url = ("http://a0.ortho.tiles.virtualearth.net/tiles/a{}.jpeg?"
                "g=854&mkt=en-US&token={}".format(quadKey, api_key))
    #print "\nThe tile URL is: {}".format(tile_url)
    return tile_url


# Read Results file in format used for MapSwipe analytics
# http://mapswipe.geog.uni-heidelberg.de
def read_mapswipe_results(csv_filename):
    with open(csv_filename) as csvfile:
        reader = csv.DictReader(csvfile)
        list_x, list_y, list_dcsn = [], [], []
        for row in reader:
            list_x.append(row['task_x'])
            list_y.append(row['task_y'])
            # decison = 1 (yes), 2 (maybe), 3 (bad image)
            list_dcsn.append(row['decision'])
            
        # Get zoom (same for all)
        task_z = row['task_z']
        tilelist = zip(list_x, list_y, list_dcsn)
    return tilelist, task_z


def retrieve_bing_image(tile_url, image_name):
    # retrieve Bing maps image
    urllib.urlretrieve(tile_url, image_name)
    if os.path.getsize(image_name) <= 1033L:
        os.remove(image_name)

if __name__ == "__main__":
    
    # Initialise parms
    #data_dir = '../data/124_MS_Madagascar/'
    #csv_filename = data_dir + 'results_124.csv'
    # index_range = range(680, 780, 1)
    data_dir = '../data/2020_Malawi_2/'
    csv_filename = data_dir + 'results_2020.csv'
    index_range = range(0, 99, 1)
    outdir_img = data_dir + 'images/'
    filename_api_key = '../api_key.txt'

    # Read api_key and Mapswipe results
    api_key = read_api_key(filename_api_key)
    tilelist, task_z = read_mapswipe_results(csv_filename)
    
    # Retrieve image tiles
    for i in index_range:
        
        
        task_x = tilelist[i][0]
        task_y = tilelist[i][1]
        task_dcsn = tilelist[i][2]
         
        # Get quadkey
        quadKey = lat_long_zoom_to_quadkey(int(task_x),int(task_y),int(task_z))
        # Get Tile URL
        tile_url = quadKey_to_url(quadKey,api_key)
        # Retrieve image
        image_name = outdir_img + str(task_x) + '-' + str(task_y) + '-' + str(task_dcsn) + '.jpeg'
        retrieve_bing_image(tile_url,image_name)

