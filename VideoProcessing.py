# -*- coding: utf-8 -*-
"""
Frame Extraction and Saving from Video
"""
'''
~ MATLAB ~
Focal Length:          fc = [ 1237.36828   1275.40537 ] ± [ 5.56050   5.78613 ]
Principal point:       cc = [ 954.79824   560.46217 ] ± [ 3.16465   3.22632 ]
Skew:             alpha_c = [ -0.00099 ] ± [ 0.00039  ]   => angle of pixel axes = 90.05701 ± 0.02233 degrees
Distortion:            kc = [ -0.30409   0.10001   -0.00065   -0.00156  0.00000 ] ± [ 0.00525   0.00635   0.00062   0.00059  0.00000 ]
Pixel error:          err = [ 0.20609   0.21830 ]

~ openCV ~
Camera Matrix:			K = [1237.36828, 0, 954.79824],[0, 1275.40537, 560.46217],[0, 0, 1]
Distortion Coefficient	d = [-0.30409, 0.10001, -0.00065, -0.00156, 0] 

(Reference: https://www.graceunderthesea.com/thesis/camera-calibration-to-undistort-images)
'''
import os
import cv2
import argparse
import numpy as np
import glob

print ("[PACKAGE] openCV version: " + cv2.__version__)
pathIN = "F:/CourseSubject/FYP/FYP 2 Progress/imstCode/ImageStitch/video/GOPROHERO3.mp4"
pathOUT = "F:/CourseSubject/FYP/FYP 2 Progress/imstCode/ImageStitch/image/ExtImg"
pathOUT2 = "F:/CourseSubject/FYP/FYP 2 Progress/imstCode/ImageStitch/image/ExtImg/ProcessedImg"

# copy parameters to arrays
K = np.array([[1237.36828, 0, 954.79824],[0, 1275.40537, 560.46217],[0, 0, 1]])
d = np.array([-0.30409, 0.10001, -0.00065, -0.00156, 0]) # just use first two terms

def extImages(pathIN, pathOUT):
    # grab the paths to the input video and starts extracting
    print("[INFO] loading video...")
    vidObj = cv2.VideoCapture(pathIN)
    
    # measure fps of the video
    fps = int(vidObj.get(cv2.CAP_PROP_FPS))
    print("[INFO] video FPS...", fps)
    
    
    # Used as couter variable
    cnt = 0
    IMGcounter = 1
    
    # checks whether frames were extracted
    success = 1
    
    while success:
            # videoObj object calls read
            # function extract frames
            success, image = vidObj.read()
            # save 1 frame out of 50 frame
            if cnt%50 == 0:
                print("[INFO] extracting frame %d ..." % IMGcounter)
                cv2.imwrite(os.path.join(pathOUT, "frame%d.jpg" % IMGcounter), image)
                #saves the frames with fram-count
                h, w = image.shape[:2]
                # undistort the images
                newCamera, roi = cv2.getOptimalNewCameraMatrix(K, d, (w,h), 0)
                undistort_img = cv2.undistort(image, K, d, None, newCamera)
                
                # crop the image
                x, y, w, h = roi
                undistort_img = undistort_img[y:y+h, x:x+w]
                cv2.imwrite(os.path.join(pathOUT2, "frame%d.jpg" % IMGcounter), undistort_img)
                IMGcounter += 1
            cnt += 1
    print("[INFO] Done extracting...")
   
""""""""""""""""""""""""""""""""""""""""""""" MAIN FUNCTION """""""""""""""""""""""""""""""""""""""""""""""""""""""""
if __name__=="__main__":
    '''
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=str, required=True,
    	help="Path to input directory of video to extract")
    ap.add_argument("-o", "--output", type=str, required=True,
    	help="Path to the extracted images")
    args = vars(ap.parse_args())
    extImages(args["video"], args["output"])
    '''
    extImages(pathIN, pathOUT)