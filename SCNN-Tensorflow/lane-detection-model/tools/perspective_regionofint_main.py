import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#perspective transform on undistorted images
def perspective_transform(img):
    imshape = img.shape
    #vertices = np.array([[(.5281*imshape[1], .6586*imshape[0]), (.5619*imshape[1],.8213*imshape[0]),
    #(.376*imshape[1],.8213*imshape[0]),(.4251*imshape[1], .6586*imshape[0])]], dtype=np.float32)
    vertices = np.array([[(.30731 * imshape[1], .77627 * imshape[0]), (.63841 * imshape[1], .75762 * imshape[0]),
                          (.42743 * imshape[1], .55932 * imshape[0]), (.49695 * imshape[1], .55593 * imshape[0])]], dtype=np.float32)
    src = np.float32(vertices)
    #dst = np.array([[(.5281*imshape[1], .6586*imshape[0]), (.5281*imshape[1],.8213*imshape[0]),
    #(.376*imshape[1],.8213*imshape[0]),(.376*imshape[1], .6586*imshape[0])]], dtype=np.float32)
    dst = np.array([[(.2737*imshape[1], .8711*imshape[0]), (.6841*imshape[1],.8254*imshape[0]),
                       (.2737*imshape[1],.03379*imshape[0]),(.7*imshape[1], .033*imshape[0])]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    img_size = (imshape[1], imshape[0]) 
    perspective_img = cv2.warpPerspective(img, M, img_size, flags = cv2.INTER_LINEAR)
    return perspective_img, Minv

#region of interest
def region_of_interest(img, vertices):
    #defining a blank mask to start with
    mask = np.zeros_like(img, dtype=np.uint8)
    if len(img.shape) > 2:
        channel_count = img.shape[2]  #  depending on out image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255  
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    mask=cv2.imread(r"C:\Users\Mouiad\Desktop\Codes-for-Lane-Detection\SCNN-Tensorflow\lane-detection-model\Settings\mask.png")
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


