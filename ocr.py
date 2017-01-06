import cv2
import numpy as np
import math
import os

knn = 3
num_score = 5

def mapCharsToRects(list_rectangles_corners, img):
    ''''list_rectangle_corners' is a list of lists of 2-tuples, repesenting the
    x,y coordinates of each corner of a rectangle. 'img' is the original opencv
    Mat image that the camera took. It is not processed in any way.'''
    ret_list = []
    
    for rect_corners in list_rectangles_corners:
        character_image = getImageSliceV3(rect_corners, img)
        char_des = getDescriptors(character_image)
        n_descriptors, w_descriptors = loadCharacterDescriptors()
        n_scores = []
        for desc in n_descriptors:
            n_scores.append(compareDescriptors(char_des, desc))
        w_scores = []
        for desc in w_descriptors:
            w_scores.append(compareDescriptors(char_des, desc))
        n_scores.sort()
        w_scores.sort()
        print ("n_Scores: ", n_scores)
        print ("w_Scores: ", w_scores)
        # number of times it matched as "N" vs "W"
        n = 0
        w = 0
        for i in range(knn):
            if n_scores[0] < w_scores[0]:
                n += 1
                del n_scores[0]
            else:
                w += 1
                del w_scores[0]
        if n > w:
            ret_list.append("N")
        else:
            ret_list.append("W")
    return ret_list

def compareDescriptors(des1, des2):
    # create BFMatcher object
    bf = cv2.BFMatcher()
    
    # Match descriptors.
    matches = bf.match(des1,des2)
    
    return scoreMatches(matches)

def getKeypointsAndDescriptors(img):
    '''Returns a 2-tuple of the (keypoints, descriptors).'''
    sift = cv2.SIFT()
    return sift.detectAndCompute(img, None)

def getDescriptors(img):
    '''Returns just the descriptors.'''
    sift = cv2.SIFT()
    kp, des = sift.detectAndCompute(img, None)
    return des

def loadCharacterDescriptors():
    '''Loads in the Descriptors for the test images. Change the path
    to match your local PC.'''
    ns = []
    ws = []
    for i in range(1,4):
        ns.append(getDescriptors(
                  cv2.imread(r"C:\Users\black\workspace\OCR RoboSub\OCR\n" \
                             + str(i) + ".jpg")))
        ws.append(getDescriptors(
                  cv2.imread(r"C:\Users\black\workspace\OCR RoboSub\OCR\w" \
                             + str(i) + ".jpg")))
    return (ns, ws)

def drawDescriptors(img1, img2):
    kp1, des1 = getKeypointsAndDescriptors(img1)
    kp2, des2 = getKeypointsAndDescriptors(img2)
    
    # create BFMatcher object
    bf = cv2.BFMatcher()
    
    # Match descriptors.
    matches = bf.match(des1,des2)
    
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    img3 = drawMatches(gray1, kp1, gray2, kp2, matches[:10])

    #show(img3)
    
# sums the first 10 (smallest) distances.
def scoreMatches(matches):
    matches = sorted(matches, key = lambda x:x.distance)
    return sum([match.distance for match in matches[:num_score]])

# credit for this function:
# https://stackoverflow.com/questions/20259025/module-object-has-no-attribute-drawmatches-opencv-python
def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    i = 0
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        #print mat.distance
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (i, i, i), 1)
        i += 255 / 10


    # Show the image
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out

# Gets below image.
def getImageSliceV2(rect_corners, img):
    ''''rect_corners' is a list of 2-tuples, repesenting the
    x,y coordinates of each corner of a rectangle. 'img' is the original opencv
    Mat image that the camera took. It is not processed in any way.
    Returns the slice of the image that the character should be within.'''
    # Currently assuming that the corners are presented in the order top-left,
    # top-right, bottom-left, bottom-right
    y1 = rect_corners[0][1]
    y2 = rect_corners[1][1]
    y3 = rect_corners[2][1]
    y4 = rect_corners[3][1]
    
    x1 = rect_corners[0][0]
    x2 = rect_corners[1][0]
    x3 = rect_corners[2][0]
    x4 = rect_corners[3][0]
    
    cols = img.shape[1]
    
    top = min(y3, y4)
    #print top
    bottom = max(y3, y4) + (abs(y1 - y3) + abs(y2 - y4)) / 2
    #print bottom
    left = max(0, min(min(x1, x3), min(x1, x3) + (x4 - x2 + x3 - x1) / 2))
    #print left
    right = min(cols, max(max(x2, x4), max(x2, x4) + (x4 - x2 + x3 - x1) / 2))
    #print right
    return img[top : bottom, left : right]
  
# Gets above image.
def getImageSliceV3(rect_corners, img):
    ''''rect_corners' is a list of 2-tuples, repesenting the
    x,y coordinates of each corner of a rectangle. 'img' is the original opencv
    Mat image that the camera took. It is not processed in any way.
    Returns the slice of the image that the character should be within.'''
    # Currently assuming that the corners are presented in the order top-left,
    # top-right, bottom-left, bottom-right
    y1 = rect_corners[0][1]
    y2 = rect_corners[1][1]
    y3 = rect_corners[2][1]
    y4 = rect_corners[3][1]
    
    x1 = rect_corners[0][0]
    x2 = rect_corners[1][0]
    x3 = rect_corners[2][0]
    x4 = rect_corners[3][0]
    
    cols = img.shape[1]
    
    top = max(0, min(y1, y2) - (abs(y1 - y3) + abs(y2 - y4)) / 2)
    #print top
    bottom = max(y1, y2)
    #print bottom
    left = max(0, min(min(x1, x3), min(x1, x3) - (x4 - x2 + x3 - x1) / 2))
    #print left
    right = min(cols, max(max(x2, x4), max(x2, x4) - (x4 - x2 + x3 - x1) / 2))
    #print right
    return img[top : bottom, left : right]
  
def getImageSlice(rect_corners, img):
    ''''rect_corners' is a list of 2-tuples, repesenting the
    x,y coordinates of each corner of a rectangle. 'img' is the original opencv
    Mat image that the camera took. It is not processed in any way.
    Returns the slice of the image that the character should be within.'''
    # Currently assuming that the corners are presented in the order top-left,
    # top-right, bottom-left, bottom-right
    
    # Finding how far in the x and y direction we must look for the letter
    # Takes the average of the distances between top_left and bot_left, and 
    # top_right and bot_right. 
    debug = True
    displace = divide2Tuple(
               add2Tuple(subtract2Tuple(rect_corners[2], rect_corners[0]), 
                         subtract2Tuple(rect_corners[3], rect_corners[1])),
               2)
    
    if debug:
        print rect_corners[0][1]
        print rect_corners[2][1]
    min_y_top = min(rect_corners[0][1], rect_corners[2][1])
    max_y_top = max(rect_corners[0][1], rect_corners[2][1])
    if debug:
        print min_y_top
        print max_y_top
    # How skewed is the rectangle?
    skew_angle = math.degrees(math.atan2(displace[1], displace[0]))
    if debug:
        print skew_angle
    rows,cols = img.shape[0:2]
    # Create a new image, basically from the bounding box of the rectangle.
    # Note: if at all skewed, this will be larger than the intended area,
    # which is perfectly fine.
    if debug:
        print displace
    # Need to add in checks for if things are out of bounds.
    if True: #displace[0] <= 0:
        x_left = max(0, rect_corners[0][0] + displace[0])
        y_left = max(0, min_y_top + displace[1])
        y_right = min(rows, max_y_top + displace[1])
        x_right = min(cols, rect_corners[1][0] + displace[0])
        if debug:
            print cols
            print x_left
            print x_right
            print
            print rows
            print y_left
            print y_right
        img_skewed = img[y_left : y_right, x_left : x_right]
    else:
        img_skewed = img[min_y_top + displace[1] : max_y_top + displace[1],
                     rect_corners[0][0] : rect_corners[1][0] + displace[0]]
    #return img_skewed
    # Rotating the matrix 
    return rotate(img_skewed, -skew_angle)

def getUnrotatedImageSlice(rect_corners, img):
    ''''rect_corners' is a list of 2-tuples, repesenting the
    x,y coordinates of each corner of a rectangle. 'img' is the original opencv
    Mat image that the camera took. It is not processed in any way.
    Returns the slice of the image that the character should be within.'''
    # Currently assuming that the corners are presented in the order top-left,
    # top-right, bottom-left, bottom-right
    
    # Finding how far in the x and y direction we must look for the letter
    # Takes the average of the distances between top_left and bot_left, and 
    # top_right and bot_right. 
    displace = divide2Tuple(
               add2Tuple(subtract2Tuple(rect_corners[2], rect_corners[0]), 
                         subtract2Tuple(rect_corners[3], rect_corners[1])),
               2)
    min_y_top = min(rect_corners[0][1], rect_corners[2][1])
    max_y_top = max(rect_corners[0][1], rect_corners[2][1])
    # How skewed is the rectangle?
    skew_angle = math.degrees(math.atan2(displace[1], displace[0]))
    rows,cols = img.shape[0:2]
    # Create a new image, basically from the bounding box of the rectangle.
    # Note: if at all skewed, this will be larger than the intended area,
    # which is perfectly fine.
    # Need to add in checks for if things are out of bounds.
    if True: #displace[0] <= 0:
        x_left = max(0, rect_corners[0][0] + displace[0])
        y_left = max(0, min_y_top + displace[1])
        y_right = min(rows, max_y_top + displace[1])
        x_right = min(cols, rect_corners[1][0] + displace[0])
        img_skewed = img[y_left : y_right, x_left : x_right]
    else:
        img_skewed = img[min_y_top + displace[1] : max_y_top + displace[1],
                     rect_corners[0][0] : rect_corners[1][0] + displace[0]]
    return img_skewed

def rotate(img, angle_degrees):
    rows,cols = img.shape[0:2]
    rotMatrix = cv2.getRotationMatrix2D((0, 0), angle_degrees, 1)
    return cv2.warpAffine(img, rotMatrix, (cols, rows))

def subtract2Tuple(tuple1, tuple2):
    return (tuple1[0] - tuple2[0], tuple1[1] - tuple2[1])

def add2Tuple(tuple1, tuple2):
    return (tuple1[0] + tuple2[0], tuple1[1] + tuple2[1])

def divide2Tuple(tuple1, divisor):
    return (int(tuple1[0] / divisor), int(tuple1[1] / divisor))

def getCharacterContour(img_slice):
    ''''img_slice' is the slice of the original image that contians the 
    character. Returns the contour of the character, to be processed later.'''
    gray = cv2.cvtColor(img_slice, cv2.COLOR_BGR2GRAY)
    
def show(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
# Code courtesy of skyuuka on StackOverFlow
def process_image(imagename, resultname):
    '''Processes and writes code to a file. Untested.'''
    img = cv2.imread(imagename, 0)
    sift =  cv2.SIFT()
    k, des = sift.detectAndCompute(img, None)
    k, des = pack_keypoint(k, des) #
    write_features_to_file(resultname, k, des)
    
def pack_keypoint(keypoints, descriptors):
    kpts = np.array([[kp.pt[0], kp.pt[1], kp.size,
                  kp.angle, kp.response, kp.octave,
                  kp.class_id]
                 for kp in keypoints])
    desc = np.array(descriptors)
    return kpts, desc

def unpack_keypoint(array):
    try:
        kpts = array[:,:7]
        desc = array[:,7:]
        keypoints = [cv2.KeyPoint(x, y, _size, _angle, _response, int(_octave), int(_class_id))
                 for x, y, _size, _angle, _response, _octave, _class_id in list(kpts)]
        return keypoints, np.array(desc)
    except(IndexError):
        return np.array([]), np.array([])
    
def read_features_from_file(filename):
    """ Read feature properties and return in matrix form. """
    if os.path.getsize(filename) <= 0:
        return np.array([]), np.array([])
    f = np.load(filename)
    if f.size == 0:
        return np.array([]), np.array([])
    f = np.atleast_2d(f)
    return f[:,:7], f[:,7:] # feature locations, descriptors

def write_features_to_file(filename, locs, desc):
    np.save(filename, np.hstack((locs,desc)))
