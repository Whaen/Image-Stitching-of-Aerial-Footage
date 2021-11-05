# -*- coding: utf-8 -*-
"""
Author: Ng Wei Haen
Topic: Image Stitching of Aerial Footage
Start Date: 17 Dec 2020
"""

import cv2
import numpy as np
import glob
import os
import time
#import math
from colorama import Style, Back
import xlsxwriter as xls
"""
Important Parameter
-------------------
detector_type (string): type of determine, "sift" or "orb"
                        Defaults to "sift".
matcher_type (string): type of determine, "flann" or "bf"
                       Defaults to "flann".
resize_ratio (int) = number needed to decrease the input images size
output_height_times (int): determines the output height based on input image height. 
                           Defaults to 2.
output_width_times (int): determines the output width based on input image width. 
                           Defaults to 4.
            
"""
detector_type = "sift"
matcher_type = "flann"
resize_ratio = 3
output_height_times = 20
output_width_times = 15
gms = False
visualize = True

image_dir = "image/Input"
key_frame = "image/Input/frame1.jpg"
output_dir = "image/Input"
    
class ImageStitching:
    def __init__(self, first_image, 
                 output_height_times = output_height_times, 
                 output_width_times = output_width_times, 
                 detector_type = detector_type, 
                 matcher_type = matcher_type):
        """This class processes every frame and generates the panorama

        Args:
            first_image (image for the first frame): first image to initialize the output size
            output_height_times (int, optional): determines the output height based on input image height. Defaults to 2.
            output_width_times (int, optional): determines the output width based on input image width. Defaults to 4.
            detector_type (str, optional): the detector for feature detection. It can be "sift" or "orb". Defaults to "sift".
        """
        self.detector_type = detector_type
        self.matcher_type = matcher_type
        if detector_type == "sift":
            # SIFT feature detector
            self.detector = cv2.xfeatures2d.SIFT_create(nOctaveLayers = 3,
                                                        contrastThreshold = 0.04,
                                                        edgeThreshold = 10,
                                                        sigma = 1.6)
            if matcher_type == "flann":
                # FLANN: the randomized kd trees algorithm
                FLANN_INDEX_KDTREE = 1
                flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
                search_params = dict (checks=200)
                self.matcher = cv2.FlannBasedMatcher(flann_params,search_params)
                
            else:
                # Brute-Force matcher
                self.matcher = cv2.BFMatcher()
        elif detector_type == "orb":
            # ORB feature detector
            self.detector = cv2.ORB_create()
            self.detector.setFastThreshold(0)
            if matcher_type == "flann":
                FLANN_INDEX_LSH = 6
                flann_params= dict(algorithm = FLANN_INDEX_LSH,
                                   table_number = 6, # 12
                                   key_size = 12,     # 20
                                   multi_probe_level = 1) #2
                search_params = dict (checks=200)
                self.matcher = cv2.FlannBasedMatcher(flann_params,search_params)
            else:
                # Brute-Force-Hamming matcher
                self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.record = []
        self.visualize = visualize
        self.output_img = np.zeros(shape=(int(output_height_times * first_image.shape[0]), 
                                          int(output_width_times*first_image.shape[1]), 
                                          first_image.shape[2]))

        self.process_first_frame(first_image)

        # output image offset
        self.w_offset = int(self.output_img.shape[0]/2 - first_image.shape[0]/2)
        self.h_offset = int(self.output_img.shape[1]/2 - first_image.shape[1]/2)

        self.output_img[self.w_offset:self.w_offset+first_image.shape[0],
                        self.h_offset:self.h_offset+first_image.shape[1], :] = first_image
        a = self.output_img
        heightM, widthM = a.shape[:2]
        a = cv2.resize(a, (int(widthM / 4), 
                           int(heightM / 4)), 
                       interpolation=cv2.INTER_AREA)
        # cv2.imshow('output', a)
        self.H_old = np.eye(3)
        self.H_old[0, 2] = self.h_offset
        self.H_old[1, 2] = self.w_offset

    def process_first_frame(self, first_image):
        """processes the first frame for feature detection and description

        Args:
            first_image (cv2 image/np array): first image for feature detection
        """
        self.base_frame_rgb = first_image
        base_frame_gray = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
        base_frame = cv2.GaussianBlur(base_frame_gray, (5,5), 0)
        self.base_features, self.base_desc = self.detector.detectAndCompute(base_frame, None)
    
    def process_adj_frame(self, next_frame_rgb):
        """gets an image and processes that image for mosaicing

        Args:
            next_frame_rgb (np array): input of current frame for the mosaicing
        """
        self.next_frame_rgb = next_frame_rgb
        next_frame_gray = cv2.cvtColor(next_frame_rgb, cv2.COLOR_BGR2GRAY)
        next_frame = cv2.GaussianBlur(next_frame_gray, (5,5), 0)
        self.next_features, self.next_desc = self.detector.detectAndCompute(next_frame, None)
        
        
        self.matchingNhomography(self.next_desc, self.base_desc)
        
        if len(self.matches) < 4:
            return
        
        print ("\n")
        self.warp(self.next_frame_rgb, self.H)
        
        # For record purpose: save into csv file later
        self.record.append([len(self.base_features), len(self.next_features), 
                            self.no_match_lr, self.no_GMSmatches, self.inlier, self.inlierRatio, self.reproError])
        
        # loop preparation
        self.H_old = self.H
        self.base_features = self.next_features
        self.base_desc = self.next_desc
        self.base_frame_rgb = self.next_frame_rgb

    def matchingNhomography(self, next_desc, base_desc):
        """matches the descriptors

        Args:
            next_desc (np array): current frame descriptor
            base_desc (np array): previous frame descriptor

        Returns:
            array: and array of matches between descriptors
        """
        # matching
        if self.detector_type == "sift":
            pair_matches = self.matcher.knnMatch(next_desc, trainDescriptors = base_desc, 
                                                 k = 2)

            """
                Store all the good matches as per Lowe's ratio test'
                The Lowe's ratio is refer to the journal "Distinctive 
                Image Features from Scale-Invariant Keypoints" by 
                David G. Lowe.
            """
            lowe_ratio = 0.8
            matches = []
            for m, n in pair_matches:
                if m.distance < n.distance * lowe_ratio:
                    matches.append(m)
            self.no_match_lr = len(matches)
            # Rate of matches (Lowe's ratio test)
            rate = float(len(matches) / ((len(self.base_features) + len(self.next_features))/2))
            print (f"Rate of matches (Lowe's ratio test): {Back.RED}%f{Style.RESET_ALL}" % rate)
            

        elif self.detector_type == "orb":
            if self.matcher_type == "flann":
                matches = self.matcher.match(next_desc, base_desc)
                '''
                lowe_ratio = 0.8
                matches = []
                for m, n in pair_matches:
                    if m.distance < n.distance * lowe_ratio:
                        matches.append(m)
                '''
                self.no_match_lr = len(matches)
                # Rate of matches (Lowe's ratio test)
                rate = float(len(matches) / (len(base_desc) + len(next_desc)))
                print (f"Rate of matches (Lowe's ratio test): {Back.RED}%f{Style.RESET_ALL}" % rate)
            else:
                pair_matches = self.matcher.match(next_desc, base_desc)
                # Rate of matches (before Lowe's ratio test)
                self.no_match_lr = len(pair_matches)
                rate = float(len(pair_matches) / (len(base_desc) + len(next_desc)))
                print (f"Rate of matches: {Back.RED}%f{Style.RESET_ALL}" % rate)

        
        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)
        
        # OPTIONAL: used to remove the unmatch pair match
        matches = cv2.xfeatures2d.matchGMS(self.next_frame_rgb.shape[:2], 
                                            self.base_frame_rgb.shape[:2], 
                                            self.next_features, 
                                            self.base_features, matches, 
                                            withScale = False, withRotation = False, 
                                            thresholdFactor = 6.0) if gms else matches
        self.no_GMSmatches = len(matches) if gms else 0
        # Rate of matches (GMS)
        rate = float(self.no_GMSmatches / (len(base_desc) + len(next_desc)))
        print (f"Rate of matches (GMS): {Back.CYAN}%f{Style.RESET_ALL}" % rate)

        # OPTIONAL: Obtain the maximum of 20 best matches
        # matches = matches[:min(len(matches), 20)]
        
        # Visualize the matches.
        if self.visualize:
            match_img = cv2.drawMatches(self.next_frame_rgb, self.next_features, self.base_frame_rgb, 
                                        self.base_features, matches, None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow('matches', match_img)
        
        self.H, self.status, self.reproError = self.findHomography(self.next_features, self.base_features, matches)
        print ('inlier/matched = %d / %d' % (np.sum(self.status), len(self.status)))
        self.inlier = np.sum(self.status)
        self.inlierRatio = float(np.sum(self.status)) / float(len(self.status))
        print ('inlierRatio = ', self.inlierRatio)
        # len(status) - np.sum(status) = number of detected outliers
        
        ''' 
            TODO - 
                To minimize or get rid of cumulative homography error is use block bundle adjustnment
                Suggested from "Multi View Image Stitching of Planar Surfaces on Mobile Devices"
                Using 3-dimentional multiplication to find cumulative homography is very sensitive
                to homography error.
        '''
        # 3-dimensional multiplication to find cumulative homography to the reference keyframe
        self.H = np.matmul(self.H_old, self.H) 
        self.H = self.H/self.H[2,2]
        self.matches = matches
        return matches
    
    @ staticmethod
    def findHomography(base_features, next_features, matches):
        """gets two matches and calculate the homography between two images

        Args:
            base_features (np array): keypoints of image 1
            next_features (np_array): keypoints of image 2
            matches (np array): matches between keypoints in image 1 and image 2

        Returns:
            np arrat of shape [3,3]: Homography matrix
        """
        
        kp1 = []
        kp2 = []
        for match in matches:
            kp1.append(base_features[match.queryIdx])
            kp2.append(next_features[match.trainIdx])
        p1_array = np.array([k.pt for k in kp1])
        p2_array = np.array([k.pt for k in kp2])
        
        homography, status = cv2.findHomography(p1_array, p2_array, method = cv2.RANSAC, 
                                                    ransacReprojThreshold = 5.0,
                                                    mask = None,
                                                    maxIters = 2000,
                                                    confidence = 0.995)
        
        #### Finding the euclidean distance error ####
        list1 = np.array(p2_array)    
        list2 = np.array(p1_array)
        list2 = np.reshape(list2, (len(list2), 2))
        ones = np.ones(len(list1))
        TestPoints = np.transpose(np.reshape(list1, (len(list1), 2)))
        print ("Length:", np.shape(TestPoints), np.shape(ones))
        TestPointsHom = np.vstack((TestPoints, ones))
        print ("Homogenous Points:", np.shape(TestPointsHom))
    
        projectedPointsH = np.matmul(homography, TestPointsHom)  # projecting the points in test image to collage image using homography matrix    
        projectedPointsNH = np.transpose(np.array([np.true_divide(projectedPointsH[0,:], projectedPointsH[2,:]), np.true_divide(projectedPointsH[1,:], projectedPointsH[2,:])]))
        
        print ("list2 shape:", np.shape(list2))
        print ("NH Points shape:", np.shape(projectedPointsNH))
        print ("Raw Error Vector:", np.shape(np.linalg.norm(projectedPointsNH-list2, axis=1)))
        Error = int(np.sum(np.linalg.norm(projectedPointsNH-list2, axis=1)))
        print ("Total Error:", Error)
        AvgError = np.divide(np.array(Error), np.array(len(list1)))
        print ("Average Error:", AvgError)
        
        ################## 
        return homography, status, AvgError

    def warp(self, next_frame_rgb, H):
        """ warps the current frame based of calculated homography H

        Args:
            next_frame_rgb (np array): current frame
            H (np array of shape [3,3]): homography matrix

        Returns:
            np array: image output of mosaicing
        """
        warped_img = cv2.warpPerspective(
            next_frame_rgb, H, (self.output_img.shape[1], self.output_img.shape[0]), 
            flags=cv2.INTER_LINEAR)
            
        transformed_corners = self.get_transformed_corners(next_frame_rgb, H)
        warped_img = self.draw_border(warped_img, transformed_corners)
        
        self.output_img[warped_img > 0] = warped_img[warped_img > 0]
        output_temp = np.copy(self.output_img)
        output_temp = self.draw_border(output_temp, transformed_corners, color=(0, 0, 255))
        
        # Visualize the stitched result
        if self.visualize:
            output_temp_copy = output_temp/255.
            output_temp_copy = cv2.normalize(output_temp_copy, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)  # convert float64 to unit8
            size = 720
            heightM, widthM = output_temp_copy.shape[:2]
            ratio = size / float(heightM)
            output_temp_copy = cv2.resize(output_temp_copy, (int(ratio * widthM), size), interpolation=cv2.INTER_AREA)
            cv2.imshow('output',  output_temp_copy)

        return self.output_img

    @ staticmethod
    def get_transformed_corners(next_frame_rgb, H):
        """finds the corner of the current frame after warp

        Args:
            next_frame_rgb (np array): current frame
            H (np array of shape [3,3]): Homography matrix 

        Returns:
            [np array]: a list of 4 corner points after warping
        """
        corner_0 = np.array([0, 0])
        corner_1 = np.array([next_frame_rgb.shape[1], 0])
        corner_2 = np.array([next_frame_rgb.shape[1], next_frame_rgb.shape[0]])
        corner_3 = np.array([0, next_frame_rgb.shape[0]])

        corners = np.array([[corner_0, corner_1, corner_2, corner_3]], dtype=np.float32)
        transformed_corners = cv2.perspectiveTransform(corners, H)

        transformed_corners = np.array(transformed_corners, dtype=np.int32)
        # output_temp = np.copy(output_img)
        # mask = np.zeros(shape=(output_temp.shape[0], output_temp.shape[1], 1))
        # cv2.fillPoly(mask, transformed_corners, color=(1, 0, 0))
        # cv2.imshow('mask', mask)

        return transformed_corners

    def draw_border(self, image, corners, color=(0, 0, 0)):
        """This functions draw rectancle border

        Args:
            image ([type]): current mosaiced output
            corners (np array): list of corner points
            color (tuple, optional): color of the border lines. Defaults to (0, 0, 0).

        Returns:
            np array: the output image with border
        """
        for i in range(corners.shape[1]-1, -1, -1):
            cv2.line(image, tuple(corners[0, i, :]), tuple(
                corners[0, i-1, :]), thickness=5, color=color)
        return image
    
    @staticmethod
    def stitchedimg_crop(stitched_img):
        """This functions crop the black edge

        Args:
            stitched_img (np array): stitched image with black edge

        Returns:
            np array: the output image with no black edge
        """
        stitched_img = cv2.normalize(stitched_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)  # convert float64 to unit8
        # Crop black edges
        stitched_img_gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(stitched_img_gray, 1, 255, cv2.THRESH_BINARY)
        dino, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        print ("Cropping black edge of stitched image ...")
        print ("Found %d contours...\n" % (len(contours)))
        
        max_area = 0
        best_rect = (0,0,0,0)
    
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
    
            deltaHeight = h-y
            deltaWidth = w-x
            if deltaHeight < 0 or deltaWidth < 0:
                deltaHeight = h+y
                deltaWidth = w+x
            
            area = deltaHeight * deltaWidth
    
            if ( area > max_area and deltaHeight > 0 and deltaWidth > 0):
                max_area = area
                best_rect = (x,y,w,h)
        
        if ( max_area > 0 ):
            final_img_crop = stitched_img[best_rect[1]:best_rect[1]+best_rect[3],
                    best_rect[0]:best_rect[0]+best_rect[2]]
        
        return final_img_crop

def main():
    images = sorted(glob.glob(image_dir + "/*.jpg"), 
                    key=lambda x: int(os.path.splitext(os.path.basename(x))[0][5:]))
    # read the first frame
    first_frame = cv2.imread(key_frame)
    heightM, widthM = first_frame.shape[:2]
    first_frame = cv2.resize(first_frame, (int(widthM / resize_ratio), 
                                            int(heightM / resize_ratio)), 
                              interpolation=cv2.INTER_AREA)
    
    image_stitching = ImageStitching(first_frame)
    round = 2
    for next_img_path in images[1:]:
        print (f'Reading {Back.YELLOW}%s{Style.RESET_ALL}...' % next_img_path)
        next_frame_rgb = cv2.imread(next_img_path)
        heightM, widthM = next_frame_rgb.shape[:2]
        next_frame_rgb = cv2.resize(next_frame_rgb, (int(widthM / resize_ratio), 
                                           int(heightM / resize_ratio)), 
                               interpolation=cv2.INTER_AREA)
        
        print ("Stitching %d / %d of image ..." % (round,len(images)))
        # process each frame
        image_stitching.process_adj_frame(next_frame_rgb)
        
        round += 1
        if round > len(images):
            print ("Please press 'q' to continue the process ...")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # cv2.imwrite('mosaic.jpg', image_stitching.output_img)
    final_img_crop = image_stitching.stitchedimg_crop(image_stitching.output_img)

    print ("Image stitching done ...")
    cv2.imwrite("%s/Normal.JPG" % output_dir, final_img_crop)
    
    # Save important results into csv file
    tuplelist = tuple(image_stitching.record)
    workbook = xls.Workbook('Normal.xlsx') 
    worksheet = workbook.add_worksheet("Normal") 
    row = 0
    col = 0
    worksheet.write(row, col, 'number_pairs')
    worksheet.write(row, col + 1, 'basefeature')
    worksheet.write(row, col + 2, 'nextfeature') 
    worksheet.write(row, col + 3, 'no_match_lr')
    worksheet.write(row, col + 4, 'match_rate')
    worksheet.write(row, col + 5, 'no_GMSmatches (OFF)')
    worksheet.write(row, col + 6, 'gms_match_rate')
    worksheet.write(row, col + 7, 'inlier')
    worksheet.write(row, col + 8, 'inlierratio')
    worksheet.write(row, col + 9, 'reproerror')
    row += 1
    number = 1
    # Iterate over the data and write it out row by row. 
    for basefeature, nextfeature, no_match_lr, no_GMSmatches, inlier, inlierratio, reproerror in (tuplelist): 
        worksheet.write(row, col, number) 
        worksheet.write(row, col + 1, basefeature)
        worksheet.write(row, col + 2, nextfeature) 
        worksheet.write(row, col + 3, no_match_lr)
        match_rate = no_match_lr / ((basefeature+nextfeature)/2)
        worksheet.write(row, col + 4, match_rate)
        worksheet.write(row, col + 5, no_GMSmatches)
        gms_match_rate = no_GMSmatches / ((basefeature+nextfeature)/2)
        worksheet.write(row, col + 6, gms_match_rate)
        worksheet.write(row, col + 7, inlier)
        worksheet.write(row, col + 8, inlierratio)
        worksheet.write(row, col + 9, reproerror)
        number += 1
        row += 1
      
    workbook.close()

""""""""""""""""""""""""""""""""""""""""""""" Main """""""""""""""""""""""""""""""""""""""
if __name__ == "__main__":
    program_start = time.process_time()
    main()
    program_end = time.process_time()
    print (f'Program elapsed time: {Back.GREEN}%s s{Style.RESET_ALL}\n' % str(program_end-program_start))
   
