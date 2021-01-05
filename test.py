import torch
import keras
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets, models
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import tensorflow as tf
import random

# this script is used for testing that the preprocessing code works

PATH = '/data/jedrzej/medical/covid_dataset/'

# get all of the training CXRs and labels
train = tf.keras.preprocessing.image_dataset_from_directory(
    PATH,
    color_mode = 'rgb',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=20)

# extracts training CXRs
x_train = np.concatenate([x for x, y in train], axis=0)

class createAugment(keras.utils.Sequence):
  # Generates masked_image, masks, and target images for training
  def __init__(self, X, y, batch_size=20, dim=(224, 224), n_channels=3, shuffle=True):
      # Initialize the constructor
      self.batch_size = batch_size
      self.X = X
      self.y = y
      self.dim = dim
      self.n_channels = n_channels
      self.shuffle = shuffle
      self.on_epoch_end()

  def __len__(self):
    # Denotes the number of batches per epoch
    return int(np.floor(len(self.X) / self.batch_size))

  def __getitem__(self, index):
    # Generate one batch of data
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
    # Generate data
    X_inputs, y_output = self.__data_generation(indexes)
    return X_inputs, y_output

  def on_epoch_end(self):
    # Updates indexes after each epoch
    self.indexes = np.arange(len(self.X))
    if self.shuffle:
      np.random.shuffle(self.indexes)

  def __data_generation(self, idxs):
    # Masked_images is a matrix of masked images used as input
    Masked_images = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels)) # Masked image
    # Mask_batch is a matrix of binary masks used as input
    Mask_batch = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels)) # Binary Masks
    # y_batch is a matrix of original images used for computing error from reconstructed image
    y_batch = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels)) # Original image
    

    ## Iterate through random indexes
    for i, idx in enumerate(idxs):
      image_copy = self.X[idx].copy()
  
      ## Get mask associated to that image
      masked_image, mask = self.__createMask(image_copy)
      
      # Masked_images[i,] = masked_image / 255
      # Mask_batch[i,] = mask / 255
      # y_batch[i] = self.y[idx] / 255

      Masked_images[i,] = masked_image
      Mask_batch[i,] = mask
      y_batch[i] = self.y[idx]

    ## Return mask as well because partial convolution require the same.
    return [Masked_images, Mask_batch], y_batch

  def __createMask(self, img):
    ## Prepare masking matrix
    mask = np.full((224,224,3), 255, np.float32) ## White background
    for _ in range(np.random.randint(1, 10)):
      # Get random x locations to start line
      x1, x2 = np.random.randint(1, 224), np.random.randint(1, 224)
      # Get random y locations to start line
      y1, y2 = np.random.randint(1, 224), np.random.randint(1, 224)
      # Get random thickness of the line drawn
      thickness = np.random.randint(1, 3)
      # Draw black line on the white mask
      cv2.line(mask,(x1,y1),(x2,y2),(0,0,0),thickness)

    ## Mask the image
    masked_image = img.copy()

    masked_image[mask==0] = 255

    print(masked_image)

    return masked_image, mask

traingen = createAugment(x_train, x_train)

# print the first image in the training data
print(traingen.X[0])