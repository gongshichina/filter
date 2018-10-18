# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 22:44:23 2018

@author: hp
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 18:36:42 2018

@author: hp
"""
import numpy as np
import matplotlib.pyplot as plt 
from scipy.ndimage import zoom
import time


# (x, y)  h = r / 2

# (i, j) 

def box_map(img, r):
    h = int(r / 2)
     # 2 D　image
    box_mean = np.zeros(img.shape)
    if img.ndim == 2:
        img = np.pad(img, ((h,),(h,)), 'reflect')
    else:
        img = np.pad(img, ((h,),(h,),(0,)), 'reflect')
    for i in range(box_mean.shape[0]):
        for j in range(box_mean.shape[1]):
            box_mean[i, j] = img[i:i+r, j:j+r].mean()
    return box_mean

    
    
def guided_filter(p, I, r, eps):
    # (x, y)
  
    I_mean = box_map(I, r)
    p_mean = box_map(p, r)
    
    corr_I = box_map(np.square(I), r)
    corr_Ip = box_map(I * p, r)
    
    var_I = corr_I - I_mean**2
    cov_Ip = corr_Ip - I_mean * p_mean
    
    a = cov_Ip / (var_I + eps)
    b = p_mean - a * I_mean
    
    a_mean = box_map(a, r)
    b_mean = box_map(b, r)
    
    for x in range(p.shape[0]):
        for y in range(p.shape[1]):
            a_mean[x, y] = a_mean[x:x+r, y:y+r].mean()
            b_mean[x, y] = b_mean[x:x+r, y:y+r].mean()
            
    q = a_mean * I + b_mean
   
    return q, (a_mean, b_mean)
    
def Fast_guided_filter(p, I, r, s, eps, mode='linear'):
    # sample a dot from images every  **s** dots
    
    I_s = None
    p_s = None
    
    scale = np.array([s, s, 1]).astype('float')
    # downsample
    p_s = zoom(p, 1 / scale)
    I_s = zoom(I, 1 / scale)
    
    r_s = int(r / s)
    
    
    _, ab_mean = guided_filter(p_s, I_s, r_s, eps)
    
    
    a_mean_s = ab_mean[0]
    b_mean_s = ab_mean[1]
    
    #  upsample
    a_mean = zoom(a_mean_s, scale)
    b_mean = zoom(b_mean_s, scale)
    
    
    q = a_mean * I + b_mean
    
    return q
    
  

if __name__ is '__main__':
    I = plt.imread('jenni16.jpg')
    p = plt.imread('jenni16.jpg')
    
    if I.dtype == 'uint8':
        I = I.astype('float') / 255
        p = p.astype('float') / 255
        
    
    r = 8
    eps = 0.05
    
    s = 4
    
    tic = time.time()
    out_1 = Fast_guided_filter(p, I, r, s, eps)
    toc1 = time.time()
    out_2, _ = guided_filter(p, I, r, eps)
    toc2 = time.time()
   
    plt.figure('fast_gf')
    plt.imshow(out_1, cmap='gray')
    print('>>> gf - time cost: {:.2} min'.format((toc2 - toc1) / 60))
    
    
    plt.figure('gf')
    plt.imshow(out_2, cmap='gray')
    print('>>> fast_gf - time cost: {:.2} min'.format((toc1 - tic) / 60))
    
    
        
    