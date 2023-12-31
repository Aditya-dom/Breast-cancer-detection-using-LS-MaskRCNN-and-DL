#!/usr/bin/env python
# coding: utf-8

# In[23]:


import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

from config import Config
import utils
import mode as modellib
import visualize
from mode import log

#from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

#get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
from collections import defaultdict

# Root directory of the project
ROOT_DIR = os.getcwd()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to COCO trained weights
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# For searching in the string 
CC = 'CC'
MLO = 'MLO'


# In[3]:


import requests
from io import BytesIO
from PIL import Image
import pandas as pd


# # Configurations

# In[4]:


class BreastTumorsConfig(Config):
    """Configuration for training on the toy tumors dataset.
    Derives from the base Config class and overrides values specific
    to the toy tumors dataset.
    """
    # Give the configuration a recognizable name
    NAME = "Breast"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    #NUM_CLASSES = 1 + 3  # background + 3 shapes
    # How many tumor shapes do we have?
    NUM_CLASSES = 1 + 3  # background + tumors

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 200

    # Use a small epoch since the data is simple
    #STEPS_PER_EPOCH = 2831
    #STEPS_PER_EPOCH = 2205
    STEPS_PER_EPOCH = 500

    # use small validation steps since the epoch is small
    #VALIDATION_STPES = 626
    VALIDATION_STPES = 50
    
    USE_MINI_MASK=False
    
config = BreastTumorsConfig()
#config.print()


# # Notebook Preferences 

# In[5]:


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


# # Dataset

# In[40]:


import skimage.color

class BreastTumorsDataset(utils.Dataset):
    
    def load_dataset(self, class_file1, image_dir1, mask_dir1, class_file2=None, image_dir2=None, mask_dir2=None):
        """
        Initialize the dataset with the images from dataset_dir folder.
        """
        
        # Add classes
        self.df_class1 = pd.read_csv(class_file1)
        
        
        self.add_class("bt_shapes", 1, 'NORMAL')
        self.add_class("bt_shapes", 2, 'BENIGN')
        self.add_class("bt_shapes", 3, 'MALIGNANT')
        
        for root, dirs, files in os.walk(image_dir1):
            for filename in files:
                self.add_image(
                    source='bt_shapes',
                    image_id=filename,
                    path=os.path.join(image_dir1, filename)
                )        
        
        self.mask_dir1 = mask_dir1

        if class_file2 != None:
            self.df_class2 = pd.read_csv(class_file2)
        
        if image_dir2 != None:
            for root, dirs, files in os.walk(image_dir2):
                for filename in files:
                    self.add_image(
                        source='bt_shapes',
                        image_id=filename,
                        path=os.path.join(image_dir2, filename)
                    )        
                
        self.mask_dir2 = mask_dir2
        
    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        #print(self.image_info[image_id]['path'])
        image = cv2.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        return image

    def load_mask(self, image_id):
        """Load masks for the given image ID.
        """
        imagepath = self.image_info[image_id]['path']
        
        image_dir, imagefile = os.path.split(imagepath)
        imagefilename, ext = os.path.splitext(imagefile)
        another = imagefilename
        var = imagefilename
        #print("Imagefile:{}".format(var))
        if(CC in var):
            var, _ = var.split('CC')
            var+='CC'
        elif(MLO in var):
            var, _ = var.split('MLO')
            var+='MLO'
        else:
            pass
            
        temp = var
        var_4_pathology = var
        #print("Imagefile But Better:{}".format(temp))
        prefix, patient_id1, patient_id2, side, viewsuffix = temp.split('_')
        #view, abn, angle, patch_width, patch_height, hstride, vstride, row, column, pathology = viewsuffix.split('-')
        #patient_id = patient_id1 + '_' + patient_id2        
        #abn_num = abn[1:]
        #maskfile = prefix + '_' + \
        #            patient_id + '_' + \
        #            side + '_' + \
        #            view + '_' + \
        #            abn_num + '_' + \
        #            'mask' + '-' + \
        #            abn + '-' + \
        #            angle + '-' + \
        #            patch_width + '-' + \
        #            patch_height + '-' + \
        #            hstride + '-' + \
        #            vstride + '-' + \
        #            row + '-' + \
        #            column + '-' + \
        #            pathology + ext
        
        #maskfile = os.path.join(self.mask_dir1,prefix+'_'+patient_id1+'_'+patient_id2+'_'+side+'_'+viewsuffix+'_mask.png')
        if(prefix=='Calc-Test' or prefix=='Mass-Test'):
            csv = pd.read_csv('../csv/calc_case_description_test_set.csv')
            maskfile = csv[csv['image file path'].str.contains(temp)]
            maskfile = maskfile['ROI mask file path'].to_frame().iloc[0][0].split('/')
            maskfile[0]+='_mask.png'
            #print("Maskfile:{}".format(maskfile[0]))
        else:
            csv = pd.read_csv('../csv/calc_case_description_train_set.csv')
            maskfile = csv[csv['image file path'].str.contains(temp)]
            maskfile = maskfile['ROI mask file path'].to_frame().iloc[0][0].split('/')
            maskfile[0]+='_mask.png'
            #print("Maskfile:{}".format(maskfile[0]))
        # Load mask
        if(prefix == 'Calc-Test' or prefix=='Mass-Test'):
            path_for_mask = '../../dataset/test_roi/test_roi/'
            #mask = cv2.imread(path_for_mask+maskfile[0])
            full_path = path_for_mask+another+'.png'
            #print("This is full Path "+full_path)
            mask = cv2.imread(full_path)
        else:
            path_for_mask = '../../dataset/train_roi/train_roi/'
            #mask = cv2.imread(path_for_mask+maskfile[0])
            full_path = path_for_mask+another+'.png'
            #print("This is full Path "+full_path)
            mask = cv2.imread(full_path)
        # If grayscale. Convert to RGB for consistency.
        #if mask.ndim != 3:
        #    mask = skimage.color.gray2rgb(mask)
        pathology = csv[csv['image file path'].str.contains(var_4_pathology)]
        pathology = pathology['pathology'].to_frame().iloc[0][0]
        # Map class names to class IDs.
        if pathology == 'MALIGNANT':
            class_ids = np.array([3, 3, 3])
        else:
            class_ids = np.array([2, 2, 2])

        return mask, class_ids.astype(np.int32)
    
    
    def image_reference(self, image_id):
        """Return a link to the images in the folder"""
        info = self.image_info[image_id]
        if info["source"] == "bt_shapes":
            return self.image_info[image_id]['path']
        else:
            super(self.__class__).image_reference(self, image_id)


# In[41]:

'''
#Validation dataset
dataset_val = BreastTumorsDataset()

dataset_val.load_dataset(class_file1=os.path.join('../csv', 'mass_case_description_test_set.csv'),                           image_dir1=os.path.join('data', 'Mass-Test Full Mammogram Patches1024 NR'),                           mask_dir1=os.path.join('data', 'Mass-Test Full ROI Patches1024 NR'))


dataset_val.prepare()
'''

# In[42]:


#Train dataset
dataset_train = BreastTumorsDataset()

dataset_train.load_dataset(class_file1=os.path.join('../csv', 'calc_case_description_train_set.csv'),                           image_dir1=os.path.join('../../dataset/train_mam/', 'train_mam'),                           mask_dir1=os.path.join('../../dataset/train_roi/', 'train_roi'))


dataset_train.prepare()


# In[43]:


#Train dataset
dataset_val = BreastTumorsDataset()

dataset_val.load_dataset(class_file1=os.path.join('../csv', 'calc_case_description_test_set.csv'),                           image_dir1=os.path.join('../../dataset/test_mam/', 'test_mam'),                           mask_dir1=os.path.join('../../dataset/test_roi/', 'test_roi'))


dataset_val.prepare()


# In[38]:


dataset_train.image_ids


# # Create Model

# In[14]:


# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)


# In[12]:


# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    LAST_MODEL = os.path.join("./logs", "Model1.h5")
    # Load the last model you trained and continue training
    model.load_weights(LAST_MODEL, by_name=True)


# In[44]:

model.keras_model.summary()
model.train(dataset_train, dataset_val, 
            #learning_rate=config.LEARNING_RATE,
            learning_rate=0.001, 
            epochs=240, 
            layers='heads')
#model.train(dataset_train, dataset_val, 
#            #learning_rate=config.LEARNING_RATE,
#            learning_rate=0.002, 
#            epochs=4, 
#            layers='heads')

#model.train(dataset_train, dataset_val, 
#            #learning_rate=config.LEARNING_RATE,
#            learning_rate=0.0002,
#            epochs=100, 
#            layers="all")
rand_num_for_model = str(np.random.randint(0, 1000))
rand_model_name  = rand_num_for_model + '_dataset.h5'
print("Model Name is: ", rand_model_name)
model_path = os.path.join(MODEL_DIR, rand_model_name)
model.keras_model.save_weights(model_path)

# ## Detection

# In[ ]:
#
#
#class InferenceConfig(BreastTumorsConfig):
#    GPU_COUNT = 1
#    IMAGES_PER_GPU = 1
#
#inference_config = InferenceConfig()
#
## Recreate the model in inference mode
#model = modellib.MaskRCNN(mode="inference", 
#                          config=inference_config,
#                          model_dir=MODEL_DIR)
#
## Get path to saved weights
## Either set a specific path or find last trained weights
#
##model_path = os.path.join(ROOT_DIR, 'logs', 'roi_breast_tumors_patches1024_nr20171125T0505', 'mask_rcnn_roi_breast_tumors_patches1024_nr_0259.h5')
#model_path = os.path.join(ROOT_DIR,'patches.h5')
## Load trained weights (fill in path to trained weights here)
#assert model_path != "", "Provide path to trained weights"
#print("Loading weights from ", model_path)
#model.load_weights(model_path, by_name=True)
#
#
## # Malignant Case
#
## In[ ]:
#
#
## Test on a random image
##image_id = random.choice(dataset_val.image_ids)
#image_id = 0
#print('Image id: {0}'.format(image_id))
#original_image, image_meta, gt_bbox, gt_mask =    modellib.load_image_gt(dataset_val, inference_config, 
#                           image_id, use_mini_mask=False)
#
#log("original_image", original_image)
#log("image_meta", image_meta)
#log("gt_bbox", gt_bbox)
#log("gt_mask", gt_mask)
#
#fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,20))
#
#visualize.display_instances(original_image, gt_bbox[:,:4], gt_mask, gt_bbox[:,4], 
#                            dataset_val.class_names, figsize=(8, 8),ax=ax1)
#
#
#print()
#start =  time.time()
#results = model.detect([original_image], verbose=1)
#end = time.time()
#
#r = results[0]
#visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
#                            dataset_val.class_names, r['scores'], ax=ax2)
#
#ax1.set_title('Ground Truth', fontsize=20)
#ax2.set_title('Prediction',fontsize=20)
#
#fig.subplots_adjust(hspace=50)
#
#print('Detection time = {0} seconds'.format(end-start))
#
#
## # Benign Case
#
## In[ ]:
#
#
## Test on a random image
##image_id = random.choice(dataset_val.image_ids)
#image_id = 12
#print('Image id: {0}'.format(image_id))
#original_image, image_meta, gt_bbox, gt_mask =    modellib.load_image_gt(dataset_val, inference_config, 
#                           image_id, use_mini_mask=False)
#
#log("original_image", original_image)
#log("image_meta", image_meta)
#log("gt_bbox", gt_bbox)
#log("gt_mask", gt_mask)
#
#fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,20))
#
#visualize.display_instances(original_image, gt_bbox[:,:4], gt_mask, gt_bbox[:,4], 
#                            dataset_val.class_names, figsize=(8, 8),ax=ax1)
#
#
#print()
#start =  time.time()
#results = model.detect([original_image], verbose=1)
#end = time.time()
#
#r = results[0]
#visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
#                            dataset_val.class_names, r['scores'], ax=ax2)
#
#ax1.set_title('Ground Truth', fontsize=20)
#ax2.set_title('Prediction', fontsize=20)
#
#fig.subplots_adjust(hspace=50)
#
#print('Detection time = {0} seconds'.format(end-start))
#
#
## In[ ]:
#
#
#
#
#
## In[ ]:
#
#
#
#
#
## In[ ]:
#
#
#
#
#
## In[ ]:
#
#
#
#
#
## In[ ]:
#
#
#
#
#
## In[ ]:
#
#
#
#
#
## In[ ]:
#
#
#
#
#
## In[ ]:
#
#
#
#
#
## In[ ]:
#
#
#
#
#
## # Ground Truth
#
## In[ ]:
#
#
## Test on a random image
#image_id = random.choice(dataset_val.image_ids)
##image_id = 110
#print('Image id: {0}'.format(image_id))
#original_image, image_meta, gt_bbox, gt_mask =    modellib.load_image_gt(dataset_val, inference_config, 
#                           image_id, use_mini_mask=False)
#
#log("original_image", original_image)
#log("image_meta", image_meta)
#log("gt_bbox", gt_bbox)
#log("gt_mask", gt_mask)
#
#visualize.display_instances(original_image, gt_bbox[:,:4], gt_mask, gt_bbox[:,4], 
#                            dataset_val.class_names, figsize=(8, 8))
#
#
## # Prediction
#
## In[ ]:
#
#
#start =  time.time()
#results = model.detect([original_image], verbose=1)
#end = time.time()
#
#r = results[0]
#visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
#                            dataset_val.class_names, r['scores'], ax=get_ax())
#
#print('Detection time = {0} seconds'.format(end-start))
#
#
## In[ ]:
#
#
#from IPython.display import HTML
#HTML('''<script>
#code_show_err=false; 
#function code_toggle_err() {
# if (code_show_err){
# $('div.output_stderr').hide();
# } else {
# $('div.output_stderr').show();
# }
# code_show_err = !code_show_err
#} 
#$( document ).ready(code_toggle_err);
#</script>
#To toggle on/off output_stderr, click <a href="javascript:code_toggle_err()">here</a>.''')
#
#
## # Class Evaluation
#
## In[ ]:
#
#
#
###image_ids = np.random.choice(dataset_val.image_ids, 2)
#image_ids = dataset_val.image_ids
#
###APs = []
#image_paths = []
#source_classes = []
#target_classes = []
#match_classes = []
#match_cases = defaultdict(int)
#
#
#for image_id in image_ids:
#    print('Image id: {0}'.format(image_id))
#    # Load image and ground truth data
#    image, image_meta, gt_bbox, gt_mask =        modellib.load_image_gt(dataset_val, inference_config,
#                               image_id, use_mini_mask=False)
#    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
#    # Run object detection
#    results = model.detect([image], verbose=0)
#    r = results[0]
#    # Compute AP
#    ##AP, precisions, recalls, overlaps =\
#    ##    utils.compute_ap(gt_bbox[:,:4], gt_bbox[:,4],
#    ##                     r["rois"], r["class_ids"], r["scores"], iou_threshold=0.5)
#    ##APs.append(AP)
#    
#    mask, source_class = dataset_val.load_mask(image_id)
#    target_class = r["class_ids"]
#    imagepath = dataset_val.image_reference(image_id)
#    prefix, p, patientid, side, suffix = imagepath.split('_')
#    view, tumor, angle, patch_width, patch_height, hstride, vstride, row, column, pathext = suffix.split('-')
#    pathology, ext = pathext.split('.')
#    
#    key = str(patientid) + side + view + tumor + pathology
#    print(key)
#    match_cases[key] = 0
#    
#    print('Source class')
#    print(source_class)
#    print('Target class')
#    print(target_class)
#    
#    source_classes.append(source_class[0])
#    
#
#    if len(target_class) == 0:
#        target_classes.append(target_class)
#        match_classes.append(0)
#    elif len(target_class) == 1:
#        target_classes.append(target_class[0])
#        match = 1 if source_class[0] == target_class[0] else 0
#        match_classes.append(match)
#        if match == 1: # if matched then set to 1
#            match_cases[key] = 1
#    else:
#        target_classes.append(target_class[0])
#        match = 0
#        for z in target_class:
#            if source_class[0] == z:
#                match = 1
#
#        match_classes.append(match)
#        if match == 1: # if matched then set to 1
#            match_cases[key] = 1
#        
#    image_paths.append(dataset_val.image_reference(image_id))
#    
#    
#print(match_cases)    
###print("mAP: ", np.mean(APs))
#
#count = 0
#total = 0
#for k, v in match_cases.items():
#    count += 1
#    total += v
#    
#print('case accuracy: ', total/count)
#
#print('accuracy: ', np.mean(match_classes))
#
#
## In[ ]:
