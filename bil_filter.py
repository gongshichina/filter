# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 10:17:06 2018

@author: hp
"""

import cv2
import numpy as np


def kernel_color(src, x, y, diameter, sigmaColor):
    # dst - x, y    (i, j) x+i, y+j 
    
    h = int((diameter - 1) / 2)
    color_w = src[x:x+diameter, y:y+diameter].copy()
    color_w = color_w - color_w[h, h]
    color_w = gaussian_map(color_w, sigmaColor)
    
    return color_w
    
def kernel_space(sigmaSpace, diameter):
    h = (diameter - 1) / 2 
    space_w = None
    space_w = np.zeros((diameter, diameter), dtype=np.float)
    
    for i in range(diameter):
        for j in range(diameter):
            space_w[i, j] = ((i - h)**2 + (j - h)**2)**0.5
    space_w = gaussian_map(space_w, sigmaSpace)
    return space_w
    
    
    
def gaussian_map(x, sigma):
    return (1.0 / (2 * np.pi * (sigma ** 2))) * np.exp(- np.square(x) / (2*sigma**2))


def bilateral_filter_own(src, diameter, sigmaColor, sigmaSpace, padding_mode):
    
    src = src.astype(np.int)   # data type is a big trap!!!! take care!!!
    dst = np.zeros(src.shape)
    h = int((diameter - 1) / 2) 
    if src.ndim == 2:
        src = np.pad(src, ((h,),(h,)), padding_mode)
    else:
        src = np.pad(src, ((h,),(h,),(0,)), padding_mode)
        
    space_w = kernel_space(sigmaSpace, diameter)
 
    # dst - x, y    (i, j) x+i, y+j 
    for x in range(dst.shape[0]):
        for y in range(dst.shape[1]):
            color_w = kernel_color(src, x, y, diameter, sigmaColor)  # d * d   d * d * c
            if src.ndim == 2:    
                weight = color_w * space_w 
            else:
                weight = color_w.reshape(-1, diameter, diameter) * space_w
                weight = weight.reshape(diameter, diameter, -1)
            k = np.sum(weight, axis=(0, 1))
  
            weight = weight / k
                   
            for i in range(weight.shape[0]):
                for j in range(weight.shape[1]):
                    dst[x, y] += weight[i, j] * src[x+i, y+j]
            
    return dst.astype(np.uint8)
    
src = cv2.imread('bil2.png')

image_procesed = bilateral_filter_own(src, 5, 12, 16, 'reflect')   #可以选择pad模式

image_procesed_opencv = cv2.bilateralFilter(src, 5, 12, 16)


cv2.imwrite('opencv_bil2.png', image_procesed_opencv)
cv2.imwrite('processed_bil2.png', image_procesed)