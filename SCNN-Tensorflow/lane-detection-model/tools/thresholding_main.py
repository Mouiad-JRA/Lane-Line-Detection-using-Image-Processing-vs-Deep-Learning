import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
from perspective_regionofint_main import *
from skimage import img_as_ubyte
import json
#grad threshold sobel x/y
# Define a function that takes an image, gradient orientation, kernel
# and threshold min / max values.
def abs_sobel_thresh(img, orient='x', sobel_kernel = 3, thresh = (0,255)):
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = sobel_kernel))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output


# Define a function to threshold an image for a given range and Sobel kernel for directionsobel
# TODO these are changed in the other file
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output.astype(np.uint8)

#used for the white lanes histogram equalisation thresholding 
def adp_thresh_grayscale(gray, thr = 250):
    img = cv2.equalizeHist(gray)
    ret, thrs = cv2.threshold(img, thresh=thr, maxval=255, type=cv2.THRESH_BINARY)
    return thrs

#Color thresholding, takes saturation and value images in single channel and corresponding threshold values
# TODO these are also changed
def color_thr(s_img, l_img, s_threshold = (0,255), l_threshold = (0,255)):
    s_binary = np.zeros_like(s_img).astype(np.uint8)
    s_binary[(s_img > s_threshold[0]) & (s_img <= s_threshold[1])] = 1
    l_binary = np.zeros_like(l_img).astype(np.uint8)
    l_binary[(l_img > l_threshold[0]) & (l_img <= l_threshold[1])] = 1
    col = ((s_binary == 1) | (l_binary == 1))
    return col

with open(r'C:\Users\Mouiad\Desktop\Codes-for-Lane-Detection\SCNN-Tensorflow\lane-detection-model\Settings\config_2.JSON') as config_file:
        data = json.load(config_file)
grad_thx_mins=data['grad_thx_min']
grad_thx_maxs=data['grad_thx_max']
grad_thy_mins=data['grad_thy_min']
grad_thy_maxs=data['grad_thy_max']
mag_th_mins=data['mag_th_min']
mag_th_maxs=data['mag_th_max']
dir_th_mins=data['dir_th_min']
dir_th_maxs=data['dir_th_max']
s_threshold_mins=data['s_threshold_min']
s_threshold_maxs=data['s_threshold_max']
l_threshold_mins=data['l_threshold_min']
l_threshold_maxs=data['l_threshold_max']
k_sizes=data['k_size']
adp_thrs=data['adp_thr']
#the main thresholding operaion is performed here 
def thresholding(img, grad_thx_min =grad_thx_mins, grad_thx_max =grad_thx_maxs,
                 grad_thy_min =grad_thy_mins, grad_thy_max = grad_thy_maxs, mag_th_min = mag_th_mins,
                 mag_th_max = mag_th_maxs, dir_th_min  =dir_th_mins, dir_th_max = dir_th_maxs, 
                 s_threshold_min = s_threshold_mins, s_threshold_max = s_threshold_maxs, 
                 l_threshold_min = l_threshold_mins, l_threshold_max = l_threshold_maxs,  k_size = k_sizes, 
                 adp_thr = adp_thrs):
    imshape = img.shape
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hls[:,:,2].astype(np.uint8)
    l_channel = hls[:,:,1].astype(np.uint8)
    ksize = k_size 
    gradx = abs_sobel_thresh(l_channel, orient='x', sobel_kernel=ksize, thresh=(grad_thx_min,grad_thx_max))
    grady = abs_sobel_thresh(l_channel, orient='y', sobel_kernel=ksize, thresh=(grad_thy_min, grad_thy_max))
    mag_binary = mag_thresh(l_channel, sobel_kernel=ksize, mag_thresh=(mag_th_min, mag_th_max))
    dir_binary = dir_threshold(l_channel, sobel_kernel=ksize, thresh=(dir_th_min, dir_th_max))
    combined = np.zeros_like(gradx)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    s_binary = color_thr(s_channel, l_channel, s_threshold=(s_threshold_min,s_threshold_max), 
                         l_threshold= (l_threshold_min,l_threshold_max)).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.uint8)
    adp = adp_thresh_grayscale(gray, adp_thr)/255
    color_binary = np.zeros_like(gradx)
    color_binary[(combined == 1) | (s_binary == 1) | (adp == 1)] = 1
    color_binary = np.dstack(( color_binary,color_binary,color_binary)).astype(np.float32)
    vertices = np.array([[(.4*imshape[1], .4*imshape[0]), (.6*imshape[1],.4*imshape[0]),
                        (.8*imshape[1],imshape[0]),(.23*imshape[1], imshape[0])]], dtype=np.int32)
    color_binary = region_of_interest(color_binary.astype(np.uint8), vertices)
    return color_binary.astype(np.float32), combined, s_binary

#img=cv2.imread(r"C:\Users\Mouiad\Desktop\Codes-for-Lane-Detection\SCNN-Tensorflow\lane-detection-model\data\CULane\driver_100_30frame\05250510_0307.MP4\00030.jpg")
#img2=cv2.imread(r"C:\Users\Mouiad\Desktop\SACES_Second_test\train_images\um_000000.png")
#x,y,z=thresholding(img)
#cv2.imshow("r",x)


