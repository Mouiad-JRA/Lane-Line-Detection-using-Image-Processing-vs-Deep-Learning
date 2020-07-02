import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
from ipywidgets import widgets
from IPython.display import display
from IPython.display import Image
from ipywidgets import interactive, interact, fixed
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from skimage import img_as_ubyte
from thresholding_main import *
from perspective_regionofint_main import *
from sliding_main import *
from scipy import misc

def draw_on_original(undist, left_fitx, right_fitx, ploty, Minv):
    # Create an image to draw the lines on
    color_warp = np.zeros_like(undist).astype(np.uint8)
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane with low confidence region in yollow
    cv2.fillPoly(color_warp, np.int_([pts]), (255, 255,0))
    # confidence region in green
    start_width = int(undist.shape[0] * 55.5 / 100.0)
    pts_left = np.array([np.transpose(np.vstack([left_fitx[start_width:], ploty[start_width:]]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx[start_width:], ploty[start_width:]])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 0, 255))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.4, 0)
    return result

def pipeline(img):
    thresh_combined, grad_th, col_th = thresholding(img)  
    perspective, Minv = perspective_transform(thresh_combined)
    perspective = cv2.cvtColor(perspective, cv2.COLOR_RGB2GRAY).astype(np.uint8)
    slides_pers, left_fitx, right_fitx, ploty, avg_cur, dist_centre_val = for_sliding_window(perspective)
    mapped_lane = draw_on_original(img, left_fitx, right_fitx, ploty, Minv)  # should be undist
    L=draw_on_original_Left(img,  left_fitx, ploty, Minv)
    R=draw_on_original_Right(img,  right_fitx, ploty, Minv)
    L=cv2.resize(L, (800,288))
    #print(L.shape)
    R=cv2.resize(R, (800,288))
    #print(R.shape)
    # font and text for drawing the offset and curvature
    """curvature = "Estimated lane curvature %.2fm" % (avg_cur)
    dist_centre = "Estimated offset from lane center %.2fm" % (dist_centre_val)
    font = cv2.FONT_HERSHEY_COMPLEX
    # using cv2 for drawing text/images in diagnostic pipeline.
    # else return the original mapped imaged with the curvature and offset drawn
    cv2.putText(mapped_lane, curvature, (30, 60), font, 1.2, (0, 255, 255), 2)
    cv2.putText(mapped_lane, dist_centre, (30, 120), font, 1.2, (0, 255, 255), 2)"""
    return mapped_lane
def pipeline1(img):
    thresh_combined, grad_th, col_th = thresholding(img)  
    perspective, Minv = perspective_transform(thresh_combined)
    perspective = cv2.cvtColor(perspective, cv2.COLOR_RGB2GRAY).astype(np.uint8)
    slides_pers, left_fitx, right_fitx, ploty, avg_cur, dist_centre_val = for_sliding_window(perspective)
    mapped_lane = draw_on_original(img, left_fitx, right_fitx, ploty, Minv)  # should be undist
    L=draw_on_original_Left(img,  left_fitx, ploty, Minv)
    R=draw_on_original_Right(img,  right_fitx, ploty, Minv)
    L=cv2.resize(L, (800,288))
    #print(L.shape)
    R=cv2.resize(R, (800,288))
    #print(R.shape)
    # font and text for drawing the offset and curvature
    """curvature = "Estimated lane curvature %.2fm" % (avg_cur)
    dist_centre = "Estimated offset from lane center %.2fm" % (dist_centre_val)
    font = cv2.FONT_HERSHEY_COMPLEX
    # using cv2 for drawing text/images in diagnostic pipeline.
    # else return the original mapped imaged with the curvature and offset drawn
    cv2.putText(mapped_lane, curvature, (30, 60), font, 1.2, (0, 255, 255), 2)
    cv2.putText(mapped_lane, dist_centre, (30, 120), font, 1.2, (0, 255, 255), 2)"""
    return mapped_lane,L,R
def draw_on_original_Left(undist,  left_fitx, ploty, Minv):
    # Create an image to draw the lines on
    color_warp = np.zeros_like(undist).astype(np.uint8)
    # confidence region in green
    start_width = int(30)
    pts_left = np.array([np.transpose(np.vstack([left_fitx[start_width:], ploty[start_width:]]))])
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts_left]), (255, 255, 255))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    global newwarp1
    newwarp1 = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    return newwarp1

def draw_on_original_Right(undist,  right_fitx, ploty, Minv):
    # Create an image to draw the lines on
    color_warp = np.zeros_like(undist).astype(np.uint8)
    # confidence region in green
    start_width = int(30)
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx[start_width:], ploty[start_width:]])))])
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts_right]), (255, 255, 255))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    global newwarp
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    return newwarp

def ones_or_zeros_for_Right():
    if np.mean(newwarp)>0:
        return True
    return False

def ones_or_zeros_for_Left():
    if np.mean(newwarp1)>0:
        return True
    return False