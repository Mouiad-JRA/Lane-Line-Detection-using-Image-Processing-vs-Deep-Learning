<p align="center">
 <b>Self Driving Car</b>
<p align="center">
<img width="550" height="350" src="https://user-images.githubusercontent.com/66889657/92144799-e749ed00-ee1f-11ea-8396-3517ef8faea4.jpg">

 
# Lane Line Detection using Image Processing vs Deep Learning
Lane line detection technique is used in many self-driving autonomous vehicles as well as line-following robots,our project is to develop a software to Detect lane lines in a variety of conditions, including changing road surfaces, curved roads, and variable lighting. An image processing  based detection and a deep learning AI detection were to be implemented, evaluated and compared in the scope of this project.
[Paper](https://www.researchgate.net/publication/344123734_Lane_Line_Detection_using_Image_Processing_Deep_Learning_comparative_study) for this project.
# Structure
 An easy to use sandbox for lane detection in the proposed methods, which is an easy-to-use graphical user interface (GUI) that contains two sections:
1. Image processing section.
2. Deep Learning section.
# Requirements
1)  Hardware: For testing, GPU with 3G memory suffices.
2)  Anaconda
3)  Spyder IDE
4)  Pycharm
5)  Opencv (for tools/lane_evaluation), version 3.2.0.7 (2.4.x should also work).
6)  Matplotlib
7)  Numpy==1.13.1
8)  easydict==1.6
9) matplotlib==2.0.2
10) glog==0.3.1
11) scikit_learn==0.19.1 
# Befor we start let's get to know the Data set
<p align="center">
<img src="https://user-images.githubusercontent.com/66889657/92105221-30cc1500-edeb-11ea-9532-65de6abccba4.JPG" width="425"/> <img src="https://user-images.githubusercontent.com/66889657/92105229-345f9c00-edeb-11ea-87fe-1a01aefeeaee.JPG" width="425"/> 

[Xingang](https://github.com/XingangPan) is the researcher who built this data set.
CULane is a large scale challenging dataset for academic research on traffic lane detection. It is collected by cameras mounted on six different vehicles driven by different drivers in Beijing. More than 55 hours of videos were collected and 133,235 frames were extracted. Data examples are shown above. We have divided the dataset into 88880 for training set, 9675 for validation set, and 34680 for test set. The test set is divided into normal and 8 challenging categories, which correspond to the 9 examples above.
</br>
For each frame, we manually annotate the traffic lanes with cubic splines. For cases where lane markings are occluded by vehicles or are unseen, we still annotate the lanes according to the context, as shown in (2). We also hope that algorithms could distinguish barriers on the road, like the one in (1). Thus the lanes on the other side of the barrier are not annotated. In this dataset we focus our attention on the detection of four lane markings, which are paid most attention to in real applications. Other lane markings are not annotated.
The whole dataset is available at [CULane](https://xingangpan.github.io/projects/CULane.html).
# To run this project:
Open computer_vison_1.py in spyder or pycharm and run it.
## First: Image processing section
### This section consists of the following stages:
1. Camera Calibration & Distortion Correction Stage.
2. Creating a Thresholded binary image using color transforms and gradients.
3. Apply a perspective transform to rectify binary image ("birds-eye view").
4. Detect lane pixels and fit a polynomial expression to find the lane boundary.
5. Determine the curvature of the lane and vehicle position with respect to center.
6. Overlay the detected lane boundaries back onto the original image.
### Let us now discuss each of these section stages in detail.
# Step 1: Camera Calibration Stage & Distortion Correction Stage
Real cameras use curved lenses to form an image, and light rays often bend a little too much or too little at the edges of these lenses. This creates an effect that distorts the edges of images, so that lines or objects appear more or less curved than they actually are. This is called radial distortion, which is the most common type of distortion.
There are three coefficients needed to correct radial distortion: k1, k2, and k3. To correct the appearance of radially distorted points in an image, one can use a correction formula mentioned below.
![Camera Calibration](https://user-images.githubusercontent.com/66889657/92038347-ecedf700-ed7b-11ea-9b0c-49ba9b6ae774.JPG)
In the following equations, (x,y) is a point in a distorted image, to undistort these points, OpenCV calculates r, which is the known distance between a point in an undistorted (corrected) image (xcorrected ,ycorrected) and the center of the image distortion, which is often the center of that image (xc ,yc ).<br/>
This center point (xc ,yc) is sometimes referred to as the distortion center, these points are pictured above.
<br/>
The ``` do_calibration() ``` function performs the following operations:
<br/>
1. Read chessboad images and convert to gray scale
2. Find the chessboard corners.
3. Perform the ``` cv2.calibrateCamera() ``` to compute the distortion co-efficients and camera matrix that we need to transform the 3d object points to 2d image points.
4. Store the calibration values to use it later.
## Now Distortion Correction:
Using the distortion co-efficients and camera matrix obtained from the camera calibration stage we undistort the images using the ```cv2.undistort``` function.
### A sample chessboard image and corresponding undistorted image is shown below: 
![Distortion Correction](https://user-images.githubusercontent.com/66889657/92039345-95e92180-ed7d-11ea-8797-2b400a364774.JPG)
###### By perfoming the distortion correction we see that the chessboard lines appear to be parallel compared to the original raw image.<br/>
### Another sample image and corresponding undistorted image is shown below:
![Distortion Correction ](https://user-images.githubusercontent.com/66889657/92040062-ced5c600-ed7e-11ea-8fb7-e3c5553bbdc9.JPG)
# Note : we didn't apply this step because the data set image were already un-distorted.
# Step 2: Create a Thresholded binary image using color transforms and gradients
In the thresholding binary image stage, multiple transformations were applied and later combined to get the best binary image for lane detection, various thresholding operations used are explained below.<br/>
1. <b> Saturation thresholding </b>: The images are transformed to HSL color space to obtain the saturation values, the yellow color lanes are best detected in the  saturation color space.
2. <b>Histogram equalized thresholding</b>: The images are transformed to gray scale and histogram is equalized using the opencv library functions, the white color lanes are best detected using this operation.
3. <b>Gradient Thresholding </b>: The Sobel operator is applied to get the gradients in the x and y direction which are also used to get the magnitude and direction thresholded images.
4. <b>Region of Interest </b>: Region of Interest operation is a process of masking unwanted portions of an image, thus keeping only essential part of the image Combining the above thresholding step to get the best  binary image for lane detection: To obtain the clear distinctive lanes in the binary image, threshold parameters for the above operation have to be fine tuned.
![Thresholded binary image](https://user-images.githubusercontent.com/66889657/92040862-cf229100-ed7f-11ea-9fc2-4fd397f19def.JPG)

### in the previous figure:
+ image 1: Image sample.
+ image 2: Sobel-X.
+ image 3: Sobel-Y.
+ image 4: magintude.
+ image 5: direction.
+ image 6: Using mask.
+ image 7: The final output of this stage.
## The thresholds are as follows:
<p align="center">
 <img src="https://user-images.githubusercontent.com/66889657/92143110-76a1d100-ee1d-11ea-92c2-d13e029a4d70.jpg">

# Step 3: Perspective transformation
A perspective transform maps the points in a given image to different, desired, image points with a new perspective. For this project, perspective transformation is applied to get a bird’s-eye view like transform, that let’s us view a lane from above; this will be useful for calculating the lane curvature later on.<br/>
The source and destination points for the perspective transformation were taken through experimentation on data samples.
## The following images shows the perspective transformation from source to destination.
+ Image 1 - Image with parallel lanes.
---
![perspective transformation](https://user-images.githubusercontent.com/66889657/92042269-85877580-ed82-11ea-99fe-5d24d63f51f7.JPG)
---
+ Image 2 - Image with lanes, here lanes appear parallel in normal view, but on perspective transformation we can clearly see that the lanes are curved.
---
![perspective transformation](https://user-images.githubusercontent.com/66889657/92042358-b667aa80-ed82-11ea-8d58-4d29895affc6.JPG)
---
#  Step 4: Detect lane pixels and fit to find the lane boundary.
After applying calibration, thresholding, and a perspective transform to a road image, we have a binary image where the lane lines stand out clearly as shown above. Next a polynomial curve is fitted to the lanes.<br/>This is defined in the function ```for_sliding_window()``` included in the file ```sliding_main.py```
For this, we first take a histogram along all the columns in the lower half of the image.<br/>
The histogram plot is shown below.
<p align="center">
 <img width="550" height="350" src="https://user-images.githubusercontent.com/66889657/92042691-6210fa80-ed83-11ea-9563-5df9c040a530.JPG">
 </p>
<br/>
With this histogram,we are adding up the pixel values along each column in the image.<br/>  In the thresholded binary image, pixels are either 0 or 1, so the two most prominent peaks in this histogram will be good indicators of the x-position of the base of the lane lines.<br/>
we use that as a starting point to search for the lines. <br/>
From that point, we use a sliding window, placed around the line centers, to find and follow the lines up to the top of the frame. The sliding window technique can be shown as in the below image: 
<p align="center">
<img width="550" height="350" src="https://user-images.githubusercontent.com/66889657/92042866-c16f0a80-ed83-11ea-94fa-0b2802148ba6.JPG">

In the above image the sliding windows are shown in green, left lanes are red colored, right lanes are blue colored and the polynomial fits are yellow lines.
#  Step 5: Determine the curvature of the lane and vehicle position with respect to center.</br>
The curvature of the lanes f(y) are calculated by using the formulae R(curve)
<p align="center">
 <img width="550" height="350" src="https://user-images.githubusercontent.com/66889657/92043096-1743b280-ed84-11ea-8ac9-0d28150f6206.JPG">

The vehicle position is calculated as the difference  between the image center and the lane center.
</br>
#  Step 6: Overlay the detected lane boundaries back onto the original image.</br>
 Now we overlay the detected lanes on the original images using inverse perspective transform.
# Second Deep Learning Section
## Not: This Section contains Tensorflow implementation of "Spatial As Deep: Spatial CNN for Traffic Scene Understanding" (SCNN-Tensorflow), you can find The original repo for the Section [here](https://github.com/cardwing/Codes-for-Lane-Detection)
### This section consists of Simple steps:
1. After we install the necessary packages in the [Requirements](https://github.com/Mouiad-JRA/Lane-Line-Detection-using-Image-Processing-vs-Deep-Learning/blob/master/Requirements)
2. Download VGG-16:<br/>
Download the vgg.npy [here](https://github.com/machrisaa/tensorflow-vgg) and put it in SCNN-Tensorflow/lane-detection-model/data.
3. Pre-trained model for testing:<br/>
Download it from [here](https://drive.google.com/file/d/1-E0Bws7-v35vOVfqEXDTJdfovUTQ2sf5/view).
# Results of applying the previous two sections and comparing between them
<br/>
Finally you've to run the file computer_vison_1.py and the following GUI will appear:
<p align="center">
<img width="550" height="350" src="https://user-images.githubusercontent.com/66889657/92126949-7992c680-ee09-11ea-9694-04b49b0d6793.jpg">
</p>
In order to apply Image processing you must press the Computer vision button and the following GUI will appear:
<p align="center">
<img width="550" height="350" src="https://user-images.githubusercontent.com/66889657/92127230-c4144300-ee09-11ea-9503-7e07f91cbd6b.jpg">
</p>
Now to apply image processing technology, First choose an image:
<p align="center">
<img width="550" height="350" src="https://user-images.githubusercontent.com/66889657/92128316-f2dee900-ee0a-11ea-8b0b-91f615956a19.jpg">
</p>
after that we can see the result as the following image shows:
<p align="center">
<img width="550" height="350" src="https://user-images.githubusercontent.com/66889657/92128582-36d1ee00-ee0b-11ea-99d8-cca38fd877ee.jpg">
</p>
Now to apply the SCNN we go back to the first GUI and press the Deep learning button, then the following GUI appears:
<p align="center">
<img width="550" height="350" src="https://user-images.githubusercontent.com/66889657/92139824-fbd6b700-ee18-11ea-9429-28a6650e6ae3.jpg">
</p>
After that you must choose an image to apply SCNN and the result'll be as the following image:
<p align="center">
<img width="550" height="350" src="https://user-images.githubusercontent.com/66889657/92140047-4b1ce780-ee19-11ea-8c37-74f286a54a12.jpg">
</p>
If you want to visually compare the two technologies, we can go back to the original interface and choose the compare button, and the following interface appears:
<p align="center">
<img width="550" height="350" src="https://user-images.githubusercontent.com/66889657/92140235-83bcc100-ee19-11ea-9ebf-e51cdd796ef3.jpg">
</p>
Then you must choose an image from the data set to apply the two technologies and the result'll be as the following image:
<p align="center">
<img src="https://user-images.githubusercontent.com/66889657/92140715-2412e580-ee1a-11ea-8755-be7cdb4d331f.jpg">
</p>
Now we will compare the SCNN output and the Image processing output using some scenarios as the following figure:
<p align="center">
<img src="https://user-images.githubusercontent.com/66889657/92140981-7c49e780-ee1a-11ea-9dee-a1ccb9426f7d.jpg">
</p>
From the previous output images we can say the following:
 we could say that both SCNN and image processing gave us an acceptable output on the previous scenarios, but notice that the SCNN has outperformed Image processing.

# Discussion
The results clearly show that SCNN network has outperformed image processing significantly in most of the scenarios we have in the dataset, in the scenario of images that contain a deviation we noticed that the SCNN gave much better results than Image processing on the same images, the same is true for scenarios that contain shadows, night, Since the image processing suffers from several problems, the most important of which are shadows and the sunshine, which the image processing technique couldn’t overcome, but the SCNN did.
# Shortcomings in Image Processing
we have observed some problems with the current Image Processing  Implementation:</br>
+ when the lane is covered by some shadow, our code originally failed to detect it, we managed to fix this issue by applying the HSL color filtering as another pre-processing step, but it didn't give good results as we expected.
+ Straight lines do not work when there are curves on the road, so to solve this problem we used Bird Eye Transform and it was difficult to get the correct parameters, we are not sure that we got the best settings.
+ For the sake of comparison, we wanted to use Intersection-Over-Union (Iou) and F1 Score , we succeeded applying them on SCNN, but in terms of image processing, we tried in various ways, but we could not achieve Iou and F-Score on this section,  so we decided the comparison should be visual.

# Conclusion
This was an exciting and challenging first project that got us to understand a lot more about color spaces, image processing and revise some linear algebra too, for the image processing section on one hand and using SCNN for lane detection on the other hand.</br>
We are looking forward to even more challenging projects that stretch our knowledge in all these fields and more, and help us in understanding how a self-driving car is built!
# Credits & References:
 [Codes-for-Lane-Detection](https://github.com/cardwing/Codes-for-Lane-Detection)
 <br/>
   </br>
 [SCNN](https://github.com/XingangPan/SCNN)
  <br/>
   </br>
 [CULane Dataset](https://xingangpan.github.io/projects/CULane.html)
  <br/>
    </br>
 [Mastering OpenCV 4 - Third Edition](https://www.packtpub.com/free-ebooks/application-development/mastering-opencv-4-third-edition/9781789533576)
  <br/>
    </br>
 [Learning OpenCV 3](https://www.oreilly.com/library/view/learning-opencv-3/9781491937983)
  <br/>
  </br>
 [Udacity Advance Lane-Detection of the Road in Autonomous Driving](https://mc.ai/udacity-advance-lane-detection-of-the-road-in-autonomous-driving)
  <br/>
  </br>
