import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import shap
import sys, os
import time
from tqdm.notebook import tqdm

from tensorflow.keras.layers import (Input, Layer, Dense)
from tensorflow.keras.models import Model

import argparse
import pickle
import math


# IMPORTANT: SET RANDOM SEEDS FOR REPRODUCIBILITY
os.environ['PYTHONHASHSEED'] = str(420)
import random
random.seed(420)
np.random.seed(420)
tf.random.set_seed(420)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Command Line Arguements
parser = argparse.ArgumentParser(description='Imagenette Kernal SHAP Explainer')
parser.add_argument('--index', type=int, default=9999, metavar='i',
                    help='Index for Job Array')
args = parser.parse_args()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Get Index (Either from argument or from SLURM JOB ARRAY)
if 'SLURM_ARRAY_TASK_ID' in os.environ:
    args.index = int(os.environ['SLURM_ARRAY_TASK_ID'])
    print('SLURM_ARRAY_TASK_ID found..., using index %s' % args.index)
else:
    print('no SLURM_ARRAY_TASK_ID... using index %s' % args.index)
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Load and Select Image
images_dir = os.path.join(os.getcwd(), 'images')
img = np.load(os.path.join(images_dir, 'processed_images.npy'), allow_pickle=True)

background = None
img = img[args.index]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Load Model

from tensorflow.keras.applications.resnet50 import ResNet50

INPUT_SHAPE = (224,224,3)

base_model = ResNet50(
    include_top=True, weights='imagenet', 
    input_shape=INPUT_SHAPE
)
base_model.trainable = False

model_input = Input(shape=INPUT_SHAPE, dtype='float32', name='input')

net = base_model(model_input)
out = Dense(10, activation='softmax')(net)

model = Model(model_input, out)

model_weights_path = 'model/20210511_21_28_36/model_weights.h5'

model.load_weights(model_weights_path)
model.trainable = False

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Explain Image

### Generate Masked Image Prediction Function

# Mask Function, Takes image, mask, background dataset 
# --> Resizes Mask from flat 14*14 --> 224 x 224
# background=None
def mask_image(masks, image, background=None):
    # Reshape/size Mask 
    mask_shape = int(masks.shape[1]**.5)
    masks = np.reshape(masks, (masks.shape[0], mask_shape, mask_shape, 1))
    resize_aspect = image.shape[0]/mask_shape
    masks = np.repeat(masks, resize_aspect, axis =1)
    masks = np.repeat(masks, resize_aspect, axis =2)
    
    # Mask Image 
    if background is not None:
        if len(background.shape) == 3:
            masked_images = np.vstack([np.expand_dims(
                (mask * image) + ((1-mask)*background[0]), 0
            ) for mask in masks])
        else:
            # Fill with Background
            masked_images = []
            for mask in masks:
                bg = [im * (1-mask) for im in background]
                masked_images.append(np.vstack([np.expand_dims((mask*image) + fill, 0) for fill in bg]))     
    else:     
        masked_images = np.vstack([np.expand_dims(mask * image, 0) for mask in masks])
        
    return masked_images #masks, image
    
# Function to Make Predictions from Masked Images
def f_mask(z):
    if background is None or len(background.shape)==3:
        y_p = []
        if z.shape[0] == 1:
            masked_images = mask_image(z, img, background)
            return(model(masked_images).numpy())
        else:
            for i in tqdm(range(int(math.ceil(z.shape[0]/100)))):
                m = z[i*100:(i+1)*100]
                masked_images = mask_image(m, img, background)
                y_p.append(model(masked_images).numpy())
            print (np.vstack(y_p).shape)
            return np.vstack(y_p)
    else:
        y_p = []
        if z.shape[0] == 1:
            masked_images = mask_image(z, img, background)
            for masked_image in masked_images:
                y_p.append(np.mean(model(masked_image), 0))
        else:
            for i in tqdm(range(int(math.ceil(z.shape[0]/100)))):
                m = z[i*100:(i+1)*100]
                masked_images = mask_image(m, img, background)
                for masked_image in masked_images:
                    y_p.append(np.mean(model(masked_image), 0))
        return np.vstack(y_p)
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

### Explain with Kernel SHAP

explainer = shap.KernelExplainer(f_mask, np.zeros((1,14*14)), link='identity')
t = time.time()
shap_values = explainer.shap_values(np.ones((1,14*14)), nsamples='auto', l1_reg=False)
explaining_time = time.time() - t

def resize_mask(masks, image):
    mask_shape = int(masks.shape[1]**.5)
    masks = np.reshape(masks, (masks.shape[0], mask_shape, mask_shape, 1))
    resize_aspect = image.shape[0]/mask_shape
    masks = np.repeat(masks, resize_aspect, axis =1)
    masks = np.repeat(masks, resize_aspect, axis =2)
    
    return masks

shap_values = [resize_mask(sv, img)  for sv in shap_values]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

### Save

save_dir = 'kernelshap'
model_dir = os.path.join(os.getcwd(), save_dir, str(args.index))
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

with open(os.path.join(model_dir, 'explaining_time.pkl'), 'wb') as f:
    pickle.dump(explaining_time, f)
    
with open(os.path.join(model_dir, 'shap_values.pkl'), 'wb') as f:
    pickle.dump(shap_values, f)