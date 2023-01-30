import cv2
import numpy as np
from numpy.linalg import norm
import os
import tensorflow as tf

model = tf.keras.models.load_model('facenet_weight/facenet_model.h5', compile=False)
def embedding(img_samples):
    if len(img_samples.shape) == 3:
        img_samples = np.expand_dims(img_samples, axis=0)
    return model.predict(img_samples)

def normalize(img):
    img = cv2.resize(img, (160, 160))
    img = img.astype('float32')
    mean, std = img.mean(), img.std()
    img = (img - mean) / std
    return img

def l2_normalize(x):
    return x / np.linalg.norm(x)

def findEuclideanDistance(A, B):
    return np.linalg.norm(A - B)
