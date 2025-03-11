import matplotlib.pyplot as plt
from matplotlib import animation, lines
import sys
import numpy as np

import imageio
import cv2
from skimage.morphology import medial_axis
from skimage.morphology import skeletonize

import astropy.units as u
from fil_finder import FilFinder2D

#############################################################################################
############################# Functions and Class around videos #############################

def read_video(filename):

    """ Extract video data from filename and return it in a numpy array format """

    reader = imageio.get_reader(filename,  'ffmpeg')
    image_series = []     
    for num, image in enumerate(reader):
        image_series.append(image)    

    return reader.get_meta_data()["fps"], np.array(image_series, dtype=np.uint8)[:,:,:,0]

def moving_average(x, w):
    """ Performs a moving average for a unidimensional array """
    return np.convolve(x, np.ones(w), 'valid') / w

def ImportROI(ROI_filename):
    """ Imports a region of interest as the complementary of a full red object from one image filename.
    Extract ROI in terms of binary matrix (0 for non ROI, 1 for ROI)
    """

    # import
    ROI_image = np.array(cv2.imread(ROI_filename))

    # Object to return
    return_shape = (ROI_image.shape[0], ROI_image.shape[1])
    ROI_matrix = np.zeros(shape=return_shape, dtype=bool)

    # Extract ROI curves in terms of pixel coordinates.
    # red_curve = np.array(np.where(np.array(ROI[:,:,2]!=ROI[:,:,1]) + np.array(ROI[:,:,0]!=ROI[:,:,1])))

    ROI_matrix = ROI_image[:,:,2]!=ROI_image[:,:,1]
    ROI_matrix += ROI_image[:,:,0]!=ROI_image[:,:,1]
    ROI_matrix = ~ROI_matrix
    
    if ROI_matrix.shape == return_shape:
        return ROI_matrix
    else:
        print("Error: ROI matrix is not of image size")
        return -1

##### Image functions #####

def KeepROI(image, ROI_matrix):
    """ Puts to zero image pixels that are not in ROI value. """

    image = np.multiply(image, ROI_matrix)
    return image

def erode(image, kernel): 
    return cv2.erode(src=image, kernel=kernel, iterations=1)

def open(image, kernel):
    return cv2.morphologyEx(src=image, kernel = kernel, op=cv2.MORPH_OPEN)

def close(image, kernel):
    return cv2.morphologyEx(src=image, kernel = kernel, op=cv2.MORPH_CLOSE)

MorphOpDict = {
    "dilation" : cv2.dilate,
    "erosion" : erode,
    "opening" : open,
    "closing" : close
}

def MorphologicalOperations(data, Ope_list):
    for k in range(len(Ope_list)):
        MorphOp = MorphOpDict[Ope_list[k][0]]
        kernel = Ope_list[k][1]
        data = MorphOp(data, kernel=kernel)
    return data

def ClusterFilter(image, cluster_size):
    """ Identifies connected components for each frame, and puts to 0 pixels
    belonging to clusters of size lower or equal to cluster_size. """

    retval, labels = cv2.connectedComponents(image)
    num = labels.max()
    for i in range(1, num+1):
        pts =  np.where(labels == i)
        # print("pts:", pts)
        if len(pts[0]) <= cluster_size:
            labels[pts] = 0
            image[pts] = 0

    return

def CoM(image):
    """ Computes and returns the center of mass 2D coordinates of a gray-scale image. """

    M=cv2.moments(image)
    c_x = M["m10"]/M["m00"]
    c_y = M["m01"]/M["m00"]

    return [c_x, c_y]

def Extract_Axonemes(image):
    """Extract the axoneme from one image, as un unordered list of coordinates."""

    # Unordered list of points
    NoN = np.nonzero(image)

    # Rearranging as a list of coordinates
    axoneme_disordered = []
    for k in range(len(NoN[0])):
       axoneme_disordered.append([NoN[1][k],NoN[0][k]])

    return axoneme_disordered

def basal_end(coordinates, point, orientation="none", orientation_threshold=-1):
    """ Identify, in a unordered list of coordinates the closest point to a given point,
    either left or right-located, or without orientation.
    Applied this code to 
    - identify the basal end from center of mass coordinates.
    - use it recursively to order the unordered list of points of both axonemes.
    If orientation is specified, one will look at the basal end on the left side or the right side of the point. """

    res_coord_index = -1
    min_dist = 126**2
    for k in range(len(coordinates)):
        
        dist = min_dist+1
        if (orientation=="left" and orientation_threshold>-1 and coordinates[k][0]<orientation_threshold) or (orientation=="right" and orientation_threshold>-1 and coordinates[k][0]>orientation_threshold) or (orientation=="none" and orientation_threshold==-1):
                dist = (coordinates[k][0]-point[0])**2 + (coordinates[k][1]-point[1])**2
                # print("coordinates[k]", coordinates[k], "point", point, "orientation threshold", orientation_threshold)     

        if dist<min_dist:
            res_coord_index = k
            min_dist = dist
    
    # Deletion of minimum from coordinates and returning it at the same time
    if res_coord_index<0:
        return None
    else:
        return coordinates.pop(res_coord_index)

###########################

def Order_axonemes(unordered_coordinates, point):

    axoneme_left = []
    axoneme_right = []

    # print("length of unordered list: ", len(unordered_coordinates))
    # Determine basal ends from point and store it in two newly created lists
    # print("Basal end, left axoneme:")
    bl = basal_end(unordered_coordinates, point, 'left', point[0])
    if bl!=None:
        axoneme_left.append(bl)
    br = basal_end(unordered_coordinates, point, 'right', point[0])
    if br!=None:
        axoneme_right.append(br)
    # print("Basal end, right axoneme:")

    # print("axoneme basal ends: ", axoneme_left, axoneme_right)

    # Recursively fill ordered axonemes with same function
    while len(unordered_coordinates)>0:
        # print("new basal end, left axoneme: ")
        bl=basal_end(unordered_coordinates, axoneme_left[-1], 'left', axoneme_right[0][0])
        if bl!=None:
            axoneme_left.append(bl)
        if len(unordered_coordinates)>0:
            # print("new basal end, right axoneme: ")
            br=basal_end(unordered_coordinates, axoneme_right[-1], 'right', axoneme_left[0][0])
            if br!=None:
                axoneme_right.append(br)

    # print("lenght of sum of both axonemes: ", len(axoneme_left)+len(axoneme_right))

    return axoneme_left, axoneme_right

##### Video class and functions #####

class Video:
    
    def __init__(self, filename=None, video_array=None, length=None, width=None, height=None):

        if filename!=None:
            """ Extract video data from filename and initialize a Video class object with it """

            self.fps, self.data = read_video(filename)
            self.length = self.data.shape[0]
            self.width = self.data.shape[1]
            self.height = self.data.shape[2]

        elif video_array!=None:
            """ Initialize a Video class object with a numpy array as data """

            self.data = video_array
            self.fps=30
            if length==None:
                self.length = self.data.shape[0]
            else:
                self.length = length
            if width==None:
                self.width = self.data.shape[1]
            else:
                self.width = width
            if height==None:
                self.height = self.data.shape[2]
            else:
                self.height=height

        elif length!=None and width!=None and height!=None:
            self.fps=30
            self.length = length
            self.width = width
            self.height = height
            self.data = np.zeros((length,width,height), dtype=np.uint8)            
        
        else:
            self.fps=30
            self.length = 1501
            self.width = 128
            self.height = 128
            self.data = np.zeros((1501,128,128), dtype=np.uint8)

    # Filtering

    def Keep_ROI(self, ROI_matrix):
        """ Returns a video with all images truncated to their region of interest,
        i.e., all pixel that is non-ROI is put to zero. """

        for t in range(self.length):
            self.data[t] = KeepROI(self.data[t], ROI_matrix)
        
        return 

    def MovingAverage(self, w):
        """ Returns the moving average of a video. 
        Note that video length is altered by the window size w. """

        MA_video = Video(length=self.length-w+1, width = self.width, height = self.height)
        for i in range(128):
            for j in range(128):
                MA_video.data[:,i,j]=moving_average(self.data[:,i,j], w)
        return MA_video

    def deMovingAverage(self, w):
        """ Returns the result of the substraction of self by its moving average. 
        Note that video length is altered by the windows size w. """

        deMA_video = self.MovingAverage(w)
        for t in range(deMA_video.length):
            deMA_video.data[t]=self.data[t+int((w-1)/2)]-deMA_video.data[t] # Tested for w odd only.

        return deMA_video

    def MedianFilter(self, w):
        """ Applies a median filter to self, with size of the kernel being w. """

        for t in range(self.length):
            self.data[t] = cv2.medianBlur(self.data[t], w)
        return

    def Threshold(self):
        """ Applies a black-and-white (binary) threshold to self. """

        for t in range(self.length):
            ret, self.data[t] = cv2.threshold(self.data[t],127,255,cv2.THRESH_BINARY)
        return

    def MorphOps(self, Ope_list):
        """ Applies successive morphological operations to self. 
        Operations are stored in a list like [['erosion', np.ones((2,2),np.uint8)], ['...', ...], ['opening', np.ones((2,2),np.uint8)]]"""
        # Note: this could be made better by first defining the functions and then using them for each t, instead of defining them self.length time.
        for t in range(self.length):
            self.data[t] = MorphologicalOperations(self.data[t], Ope_list)
    
    def ClusterThreshold(self, cluster_size):
        """ Identifies connected components for each frame, and puts to 0 pixels
        belonging to clusters of size lower or equal to cluster_size. """

        for t in range(self.length):
            ClusterFilter(self.data[t], cluster_size)

        return

    def Skeletonize(self, mode):
        if mode=="skeleton":
            for t in range(self.length):
                self.data[t] = skeletonize(self.data[t]/255)*255
        elif mode=="topskeleton":
            for t in range(self.length):
                skel, distance = medial_axis(self.data[t], return_distance=True)
                self.data[t] = distance * skel
        return

    # Analysis

    def FindSkeletons(self):
        """ To change. Supposed to extract the longest branch of a skeleton."""
        for t in range(self.length):
            fil = FilFinder2D(self.data[t], distance=250 * u.pc, mask=self.data[t])
            fil.create_mask(verbose=False, use_existing_mask=True) # Identify structure from image
            fil.medskel(verbose=False) # Skeletonize
            fil.analyze_skeletons(branch_thresh=40* u.pix, skel_thresh=10 * u.pix, prune_criteria='length') # Pruning by length
            self.data[t]=fil.skeleton_longpath
        return


    def CenterOfMass(self):
        """ Computes and returns the time-averaged center of mass
         2D coordinates of a gray-scale image. """
        
        c_x=0
        c_y=0

        for t in range(self.length):
            com_x, com_y = CoM(self.data[t])
            c_x+=com_x
            c_y+=com_y
        c_x/=self.length
        c_y/=self.length

        return [c_x, c_y]
    

    def ExtractAxonemes(self):
        """ Extract from a video the axonemes at each timeframe, 
        in the form of an array of unordered lists. """

        axoneme_disordered_video = []
        for t in range(self.length):
            axoneme_disordered_video.append(Extract_Axonemes(self.data[t]))
        return axoneme_disordered_video


    def ExtractAndOrder(self):
        """ Extracts an unordered list of points representing the two axonemes,
            Computes the center of mass between the two axonemes,
            Orders and separates the two axonemes from the basal end to the distal end."""

        axoneme_disordered_video = self.ExtractAxonemes()
        center_of_mass = self.CenterOfMass()

        axoneme_left_video=[]
        axoneme_right_video=[]    
        for t in range(self.length):
            a, b = Order_axonemes(axoneme_disordered_video[t], center_of_mass)
            axoneme_left_video.append(np.array(a))
            axoneme_right_video.append(np.array(b))

        return axoneme_left_video, axoneme_right_video

    # Write a video file

    def Write(self, filename, fps):

        writer = imageio.get_writer(filename + '.avi', fps=fps)
        for t in range(self.length):
            writer.append_data(self.data[t])
        writer.close()

    # Visualisation
    # 
    #

#####################################

#############################################################################################
#############################################################################################




############################################################################################
################################## Test of the code above ##################################

if __name__=="__main__":

    ##### Import video #####
    chlamy_video = Video(filename='chlamy_video.avi')

    w_ma=151
    chlamy_video = chlamy_video.deMovingAverage(w_ma)
    
    image_test = chlamy_video.data[0]

    # plt.imshow(image_test, interpolation='nearest')
    # plt.show()

    ##### Import ROI #####
    ROI_matrix = ImportROI('full_ROI.png').astype(int)
    # np.set_printoptions(threshold=sys.maxsize)
    # print("ROI_matrix: ", ROI_matrix)
    # plt.imshow(ROI_matrix, interpolation='nearest')
    # plt.show()

    #####################
    ##### Filtering #####

    chlamy_video.Threshold()

    morph_ope_list=[]
    k_erosion=np.ones((2,2),np.uint8)
    morph_ope_list.append(['erosion', k_erosion])
    # k_opening=np.ones((2,2),np.uint8)
    # morph_ope_list.append(['opening', k_opening])
    # k_closing=np.ones((2,2),np.uint8)
    # morph_ope_list.append(['closing', k_closing])

    chlamy_video.MorphOps(morph_ope_list)

    chlamy_video.Keep_ROI(ROI_matrix)

    chlamy_video.ClusterThreshold(15)

    #####################
    #####################

    ###########################
    ##### Skeletonization #####

    chlamy_video.Skeletonize("skeleton") # Normal skeletonization
    # chlamy_video.Skeletonize("topskeleton") # Topological skeletonization

    # chlamy_video.FindSkeletons()

    # test_skeleton = chlamy_video.data[0]

    # # Show the longest path
    # fil = FilFinder2D(test_skeleton, distance=250 * u.pc, mask=test_skeleton)
    # fil.create_mask(verbose=False, use_existing_mask=True) # Identify structure from image
    # fil.medskel(verbose=False) # Skeletonize
    # fil.analyze_skeletons(branch_thresh=40* u.pix, skel_thresh=10 * u.pix, prune_criteria='length') # Pruning by length

    # # Show the longest path
    # plt.imshow(fil.skeleton, cmap='gray')
    # plt.contour(fil.skeleton_longpath, colors='r')
    # plt.axis('off')
    # plt.show()

    ###########################
    ###########################


    #####################################
    ##### Extract curve coordinates #####

    axoneme_disordered = Extract_Axonemes(chlamy_video.data[10])
    
    # for k in range(len(axoneme_disordered)):
    #     plt.plot(axoneme_disordered[k][0],axoneme_disordered[k][1], "b+")
    # plt.imshow(chlamy_video.data[0], interpolation="nearest")
    # plt.show()

    ## Extract basal body skeleton end as a center of mass average

    # Extract center of mass coordinates

    c_x, c_y = chlamy_video.CenterOfMass()
    print("center of mass average:", c_x, c_y)

    # Identify basal ends in the list of coordinates

    # plt.plot(c_x, c_y,'ro')
    # #plt.plot(b_x, b_y,'bo')
    # plt.plot(b_x_left, b_y_left,'go')
    # plt.plot(b_x_right, b_y_right, 'co')
    # plt.imshow(chlamy_video.data[100], interpolation="nearest")
    # plt.show()
    # exit()

    ## Reordering --> Needs to separate the two axonemes having identified the origins
    
    axoneme_left, axoneme_right = Order_axonemes(axoneme_disordered, [c_x, c_y])
    axoneme_left=np.array(axoneme_left)
    axoneme_right=np.array(axoneme_right)
    ## Test to print left axoneme, then right axoneme

    # print(axoneme_left)
    plt.plot(c_x, c_y,'ro')
    plt.imshow(chlamy_video.data[10], interpolation="nearest")

    
    plt.plot(axoneme_left[:,0], axoneme_left[:,1], 'bx')
    plt.plot(axoneme_right[:,0], axoneme_right[:,1], 'gx')
    plt.show()
    exit()



    #####################################
    #####################################


    ###################
    ##### Writing #####
    filename='written_video'
    chlamy_video.Write(filename, fps=chlamy_video.fps)
    ###################
    ###################




############################################################################################
############################################################################################