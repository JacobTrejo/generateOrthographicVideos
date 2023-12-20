from Programs.Config import Config
from scipy.io import loadmat
import random
import time
import numpy as np

# Configuration variables
imageSizeY, imageSizeX = Config.imageSizeY, Config.imageSizeX
# theta_array = loadmat('Inputs/generated_pose_all_2D_50k.mat')
# theta_array = theta_array['generated_pose']

# This part is to get the indices, this part can be hardcoded #########
amount_of_boxes = 9
temp_arr = np.array(range(amount_of_boxes))
original, offset = np.meshgrid(temp_arr, temp_arr)
original += offset
original = np.remainder(original, amount_of_boxes)
indices_for_permutation = list(np.concatenate([original[rowIdx, :] for rowIdx in range(amount_of_boxes)], axis=0))
# end of getting indices ################################################


# Rotate along x axis. Angles are accepted in radians
def rotx(angle):
    M = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    return M


# Rotate along y axis. Angles are accepted in radians
def roty(angle):
    M = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    return M


# Rotate along z axis. Angles are accepted in radians
def rotz(angle):
    M = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    return M

####################### New Functions which got added

def theta_array_to_pts(theta_array,x, seglen):
    theta = x[2]
    hp = x[0: 2]
    dt = x[2: 11]
    # dt = np.concatenate( ((np.ones((500000)) * theta)[:, None], theta_array), axis=1)
    dt = np.concatenate( ((np.ones((theta_array.shape[0])) * theta)[:, None], theta_array), axis=1)
    # pt = np.zeros((2, 10))
    # theta = np.zeros((9, 1))
    # theta[0] = dt[0]

    # pt[:, 0] = hp
    theta = np.cumsum(dt, axis=1)
    dx = np.cos(theta)
    dy = np.sin(theta)
    dx = np.cumsum(dx, axis=1)
    dy = np.cumsum(dy, axis=1)
    dx *= seglen
    dy *= seglen
    dx += hp[0]
    dy += hp[1]
    # backbone = np.stack((dx, dy), axis = 1)

    # pt = np.zeros((500000,2,10))
    pt = np.zeros((theta_array.shape[0],2,10))
    pt[:,0,1:] = dx
    pt[:,1,1:] = dy

    # pt[:,0] = hp
    pt[:,0,0] = hp[0]
    pt[:,1,0] = hp[1]
    return pt

def theta_array_pts_to_boxes(pts):
    centers = pts[..., 0:9] + pts[..., 1:10]
    centers *= .5
    lenghts = pts[..., 0:9] - pts[..., 1:10]
    lenghts = np.abs(lenghts)
    threshold = 3
    padding = .7
    lenghts[lenghts < threshold] = threshold
    lenghts = lenghts * (1 + padding)
    # only half should be added to each center, like radius = .5 * diameter
    lenghts *= .5
    # Lets make the Box around the head bigger
    lenghts[...,0] *= 3

    # top_bottom = np.zeros((500000, 4, 9))
    top_bottom = np.zeros((pts.shape[0], 4, 9))
    top_bottom[:, :2, :] = centers + lenghts
    top_bottom[:, 2:, :] = centers - lenghts

    return top_bottom

def theta_array_which_crash(theta_array_boxes, boxes):
    """
        function which will check which delta thetas cause a crash
    :param theta_array_boxes: thetha arrays as shells
    :param boxes: the shell of the fish
    :return: 500000 array stating which theta array idx are cause crashes or not
    """
    # NOTE: this function is not vectorized with regard to boxes for all the fish, because it will crash
    # To do so would easily require 2 arrays size greater than 30,000,000,000

    # creating the permutations of the boxes in order to check every combination in a vectorized manner
    permutated_fish_box = boxes[:, indices_for_permutation]
    theta_array_boxes_stacked = np.concatenate(
        [theta_array_boxes, theta_array_boxes, theta_array_boxes, theta_array_boxes, theta_array_boxes,
         theta_array_boxes, theta_array_boxes, theta_array_boxes, theta_array_boxes],
        axis=2)

    # permutated_fish_box_big = np.ones((500000, 4, 9 * 9))
    permutated_fish_box_big = np.ones((theta_array_boxes.shape[0], 4, 9 * 9))
    permutated_fish_box_big *= permutated_fish_box

    # Array of size 500000, which has the indices of which delta thetas cause a crash and which do not
    which_crashed = np.any((theta_array_boxes_stacked[:, 0, :] > permutated_fish_box_big[:, 2, :]) *
                           (theta_array_boxes_stacked[:, 2, :] < permutated_fish_box_big[:, 0, :]) *
                           (theta_array_boxes_stacked[:, 1, :] > permutated_fish_box_big[:, 3, :]) *
                           (theta_array_boxes_stacked[:, 3, :] < permutated_fish_box_big[:, 1, :]), axis=1)
    return which_crashed


def x_seglen_to_pts_short(x, seglen):
    theta = x[2]
    hp = x[0:2]
    dt = x[3:11]
    dt = np.concatenate((np.array( theta)[None], np.array(dt)), axis=0)
    theta = np.cumsum(dt, axis = 0)
    dx = np.cos(theta)
    dy = np.sin(theta)
    dx = np.cumsum(dx, axis=0)
    dy = np.cumsum(dy, axis=0)
    dx *= seglen
    dy *= seglen
    dx += hp[0]
    dy += hp[1]
    pt = np.zeros((2,10))
    pt[0,1:] = dx
    pt[1,1:] = dy
    pt[0,0] = hp[0]
    pt[1,0] = hp[1]
    return pt

def pts_to_boxes_short(pts):
    padding = .7
    threshold = 3
    # number of keypoints - 1
    amountOfBoxes = 10 - 1

    centers = pts[:,:-1] + pts[:,1:]
    centers *= .5
    lenghts = pts[:,:-1] - pts[:,1:]
    lenghts = np.abs(lenghts)

    lenghts[lenghts < threshold] = threshold
    lenghts = lenghts * (1 + padding)
    # only half should be added to each center, like radius = .5 * diameter
    lenghts *= .5
    # Lets make the Box around the head bigger
    lenghts[:,0] *= 3


    boxes = np.zeros((4,amountOfBoxes))
    boxes[:2,:] = centers + lenghts
    boxes[2:,:] = centers - lenghts
    return boxes

# def get_good_dtheta_indices(x,y,theta0,seglen, overlappingFishVectList, fishVectToOverlapp):
#     theta_array_boxes = theta_array_pts_to_boxes( theta_array_to_pts(theta_array, [x,y,theta0], seglen) )
#     f2o_seglen, f2o_z, f2o_xvect = fishVectToOverlapp[0], fishVectToOverlapp[1], fishVectToOverlapp[2:]
#     # f2o_seglen, f2o_xvect = fishVectToOverlapp[0], fishVectToOverlapp[1:]
#
#     f2o_boxes = pts_to_boxes_short( x_seglen_to_pts_short(f2o_xvect, f2o_seglen) )
#     # In this case we want them to actually crash so that they overlap
#     start = time.time()
#     which_thetas_overlap = theta_array_which_crash(theta_array_boxes, f2o_boxes)
#     for overlappingFishVect in overlappingFishVectList:
#         o_seglen, o_z, o_xvect = overlappingFishVect[0], overlappingFishVect[1], overlappingFishVect[2:]
#         # o_seglen, o_xvect = overlappingFishVect[0], overlappingFishVect[1:]
#         o_boxes = pts_to_boxes_short(x_seglen_to_pts_short(o_xvect, o_seglen))
#         # we are inverting the ones that crash because those are bad since they crash with the surrounding fish
#         which_thetas_overlap *= np.invert( theta_array_which_crash(theta_array_boxes, o_boxes) )
#     end = time.time()
#     duration = end - start
#     print('duration of loop', duration)
#     return which_thetas_overlap

def get_good_dtheta_indices_chuncks(x,y,theta0,seglen, overlappingFishVectList, fishVectToOverlapp,chunk):
    """
        Works just like the function above only thing is that you now only consider chunks of the theta array
        this is so that it runs faster since considering the full array takes almost 10 seconds, this takes a fraction
        of a second
    :param x: x coordinate of the fish
    :param y: y coordinate of the fish
    :param theta0: theta0 coordinate of the fish
    :param seglen: segment length
    :param overlappingFishVectList: the current list of overlapping fish
    :param fishVectToOverlapp: the fish you want to overlap
    :param chunk: number between 0 - 9, representing which chuck of the theta array when split into 10 pieces
    :return: an array of size 50,000 representing the good indices of a chunk of the theta array
    """
    pts = theta_array_to_pts(theta_array[50000 * chunk :50000 * (chunk + 1),...], [x,y,theta0], seglen)
    which_are_in_bounds = np.any( (pts[:,0,:] < Config.imageSizeX) * (pts[:,0,:] > 0) * \
                          (pts[:,1,:] < Config.imageSizeY) * (pts[:,1,:] > 0), axis=1 )

    # theta_array_boxes = theta_array_pts_to_boxes(
    #     theta_array_to_pts(theta_array[50000 * chunk :50000 * (chunk + 1),...], [x,y,theta0], seglen) )

    theta_array_boxes = theta_array_pts_to_boxes( pts )

    f2o_seglen, f2o_z, f2o_xvect = fishVectToOverlapp[0], fishVectToOverlapp[1], fishVectToOverlapp[2:]
    # f2o_seglen, f2o_xvect = fishVectToOverlapp[0], fishVectToOverlapp[1:]

    f2o_boxes = pts_to_boxes_short( x_seglen_to_pts_short(f2o_xvect, f2o_seglen) )
    # In this case we want them to actually crash so that they overlap
    start = time.time()
    which_thetas_overlap = theta_array_which_crash(theta_array_boxes, f2o_boxes)
    for overlappingFishVect in overlappingFishVectList:
        o_seglen, o_z, o_xvect = overlappingFishVect[0], overlappingFishVect[1], overlappingFishVect[2:]
        # o_seglen, o_xvect = overlappingFishVect[0], overlappingFishVect[1:]
        o_boxes = pts_to_boxes_short(x_seglen_to_pts_short(o_xvect, o_seglen))
        # we are inverting the ones that crash because those are bad since they crash with the surrounding fish
        which_thetas_overlap *= np.invert( theta_array_which_crash(theta_array_boxes, o_boxes) )
    which_thetas_overlap *= which_are_in_bounds

    return which_thetas_overlap


####################### End of New Functions


def x_seglen_to_3d_points_vectorized(x, seglen):
    hp = x[0: 2]
    dt = x[2: 11]
    pt = np.zeros((2, 10))
    theta = np.zeros((9, 1))
    theta[0] = dt[0]
    pt[:, 0] = hp
    theta = np.cumsum(dt)
    dx = np.cos(theta)
    dy = np.sin(theta)
    dx = np.cumsum(dx)
    dy = np.cumsum(dy)
    dx *= seglen
    dy *= seglen
    dx += hp[0]
    dy += hp[1]
    # backbone = np.stack((dx, dy), axis = 1)

    pt = np.zeros((2,10))
    pt[0,1:] = dx
    pt[1,1:] = dy
    pt[:,0] = hp

    size_lut = 49
    size_half = (size_lut + 1) / 2

    dh1 = pt[0, 1] - np.floor(pt[0, 1])
    dh2 = pt[1, 1] - np.floor(pt[1, 1])

    d_eye = seglen

    XX = size_lut
    YY = size_lut
    ZZ = size_lut

    c_eyes = 1.9
    c_belly = 0.98
    c_head = 1.04
    canvas = np.zeros((XX, YY, ZZ))

    theta, gamma, phi = x[2], 0, 0

    R = rotz(theta) @ roty(phi) @ rotx(gamma)

    # Initialize points of the ball and stick model in the canvas
    pt_original = np.zeros((3, 3))
    # pt_original_1 is the mid-point in Python's indexing format
    pt_original[:, 1] = np.array([np.floor(XX / 2) + dh1, np.floor(YY / 2) + dh2, np.floor(ZZ / 2)])
    pt_original[:, 0] = pt_original[:, 1] - np.array([seglen, 0, 0], dtype=pt_original.dtype)
    pt_original[:, 2] = pt_original[:, 1] + np.array([seglen, 0, 0], dtype=pt_original.dtype)

    eye1_c = np.array([[c_eyes * pt_original[0, 0] + (1 - c_eyes) * pt_original[0, 1]],
                       [c_eyes * pt_original[1, 0] + (1 - c_eyes) * pt_original[1, 1] + d_eye / 2],
                       [pt_original[2, 1] - seglen / 8]], dtype=pt_original.dtype)
    eye1_c = eye1_c - pt_original[:, 1, None]
    eye1_c = np.matmul(R, eye1_c) + pt_original[:, 1, None]

    eye2_c = np.array([[c_eyes * pt_original[0, 0] + (1 - c_eyes) * pt_original[0, 1]],
                       [c_eyes * pt_original[1, 0] + (1 - c_eyes) * pt_original[1, 1] - d_eye / 2],
                       [pt_original[2, 1] - seglen / 8]], dtype=pt_original.dtype)
    eye2_c = eye2_c - pt_original[:, 1, None]
    eye2_c = np.matmul(R, eye2_c) + pt_original[:, 1, None]

    eye1_c[0] = eye1_c[0] - (size_half - 1) + pt[0, 1]
    eye1_c[1] = eye1_c[1] - (size_half - 1) + pt[1, 1]
    eye2_c[0] = eye2_c[0] - (size_half - 1) + pt[0, 1]
    eye2_c[1] = eye2_c[1] - (size_half - 1) + pt[1, 1]

    pt = np.concatenate([pt, eye1_c[0: 2], eye2_c[0: 2]], axis=1)
    return pt


def x_seglen_to_3d_points(x, seglen):
    """
        Function that turns the x vector into the 3d points of the fish
    :param x:
    :param seglen:
    :return:
    """
    hp = x[0: 2]
    dt = x[2: 11]
    pt = np.zeros((2, 10))
    theta = np.zeros((9, 1))
    theta[0] = dt[0]
    pt[:, 0] = hp

    for n in range(0, 9):
        R = np.array([[np.cos(dt[n]), -np.sin(dt[n])], [np.sin(dt[n]), np.cos(dt[n])]])
        if n == 0:
            vec = np.matmul(R, np.array([seglen, 0], dtype=R.dtype))
        else:
            vec = np.matmul(R, vec)
            theta[n] = theta[n - 1] + dt[n]
        pt[:, n + 1] = pt[:, n] + vec

    # Now calculating the eyes
    size_lut = 49
    size_half = (size_lut + 1) / 2

    dh1 = pt[0, 1] - np.floor(pt[0, 1])
    dh2 = pt[1, 1] - np.floor(pt[1, 1])

    d_eye = seglen

    XX = size_lut
    YY = size_lut
    ZZ = size_lut

    c_eyes = 1.9
    c_belly = 0.98
    c_head = 1.04
    canvas = np.zeros((XX, YY, ZZ))

    theta, gamma, phi = x[2], 0, 0

    R = rotz(theta) @ roty(phi) @ rotx(gamma)

    # Initialize points of the ball and stick model in the canvas
    pt_original = np.zeros((3, 3))
    # pt_original_1 is the mid-point in Python's indexing format
    pt_original[:, 1] = np.array([np.floor(XX / 2) + dh1, np.floor(YY / 2) + dh2, np.floor(ZZ / 2)])
    pt_original[:, 0] = pt_original[:, 1] - np.array([seglen, 0, 0], dtype=pt_original.dtype)
    pt_original[:, 2] = pt_original[:, 1] + np.array([seglen, 0, 0], dtype=pt_original.dtype)

    eye1_c = np.array([[c_eyes * pt_original[0, 0] + (1 - c_eyes) * pt_original[0, 1]],
                       [c_eyes * pt_original[1, 0] + (1 - c_eyes) * pt_original[1, 1] + d_eye / 2],
                       [pt_original[2, 1] - seglen / 8]], dtype=pt_original.dtype)
    eye1_c = eye1_c - pt_original[:, 1, None]
    eye1_c = np.matmul(R, eye1_c) + pt_original[:, 1, None]

    eye2_c = np.array([[c_eyes * pt_original[0, 0] + (1 - c_eyes) * pt_original[0, 1]],
                       [c_eyes * pt_original[1, 0] + (1 - c_eyes) * pt_original[1, 1] - d_eye / 2],
                       [pt_original[2, 1] - seglen / 8]], dtype=pt_original.dtype)
    eye2_c = eye2_c - pt_original[:, 1, None]
    eye2_c = np.matmul(R, eye2_c) + pt_original[:, 1, None]

    eye1_c[0] = eye1_c[0] - (size_half - 1) + pt[0, 1]
    eye1_c[1] = eye1_c[1] - (size_half - 1) + pt[1, 1]
    eye2_c[0] = eye2_c[0] - (size_half - 1) + pt[0, 1]
    eye2_c[1] = eye2_c[1] - (size_half - 1) + pt[1, 1]

    pt = np.concatenate([pt, eye1_c[0: 2], eye2_c[0: 2]], axis=1)

    return pt


def addBoxes(pt, padding=.7):
    """
        Function that adds boxes to the fish given its backbone points
    :param pt:
    :param padding:
    :return:
    """
    number_of_eyes = 2

    threshold = Config.minimumSizeOfBox
    # minus one assuming the boxes are in the center
    amount_of_boxes = pt.shape[1] - number_of_eyes - 1

    dimensions_of_center = 2
    dimensions_of_box = 2
    pointsAndBoxes = np.zeros((dimensions_of_box + dimensions_of_center, amount_of_boxes))
    # TODO:vectorize
    for pointIdx in range(amount_of_boxes):
        fPX, fPY = pt[:, pointIdx]
        sPX, sPY = pt[:, pointIdx + 1]

        c = np.array([(fPX + sPX) / 2, (fPY + sPY) / 2])
        l = np.array([np.abs(fPX - sPX), np.abs(fPY - sPY)])

        # Even if is close to zero there should still be some length/width to the box
        l[l < threshold] = threshold

        # Adding the padding
        l = (padding + 1) * l

        cX, cY = c
        lX, lY = l

        pointsAndBoxes[:, pointIdx] = [cX, cY, lX, lY]

    # Now the eyes
    for eyeIdx in range(number_of_eyes):
        lenghtOfEyes = 9  # ?, maybe make it depend on seglen
        cX, cY = pt[:, -number_of_eyes + eyeIdx]
        lX, lY = lenghtOfEyes, lenghtOfEyes

        box_for_eyes = np.array([[cX], [cY], [lX], [lY]])
        pointsAndBoxes = np.concatenate((pointsAndBoxes, box_for_eyes), axis=1)
    return pointsAndBoxes



def doesThisFishInterfereWithTheAquarium(fishVect, fishVectList):
    """
        Function to check if the fish in question will crash with any of the other fishes
    :param fishVect:
    :param fishVectList:
    :return:
    """
    # Preprocessing
    fishVect = np.array(fishVect)
    if len(fishVectList) == 0: return False
    fishVectList = np.array(fishVectList)

    # concatenating to avoid repetitions
    fishVectList = np.concatenate((fishVect[np.newaxis, :], fishVectList), axis=0)

    boxesList = []
    for fishVectIdx in range(fishVectList.shape[0]):
        fishVect = fishVectList[fishVectIdx, ...]

        seglen, z, x = fishVect[0], fishVect[1], fishVect[2:]
        pt = x_seglen_to_3d_points(x, seglen)
        boxes_covering_fish = addBoxes(pt)

        # Converting them to top bottom format
        boxes_covering_fish_top_bottom_format = np.copy(boxes_covering_fish)
        boxes_covering_fish_top_bottom_format[0, :] = \
            boxes_covering_fish[0, :] + (boxes_covering_fish_top_bottom_format[2, :] / 2)
        boxes_covering_fish_top_bottom_format[1, :] = \
            boxes_covering_fish[0, :] - (boxes_covering_fish_top_bottom_format[2, :] / 2)
        # For the y - dimension
        boxes_covering_fish_top_bottom_format[2, :] = \
            boxes_covering_fish[1, :] + (boxes_covering_fish_top_bottom_format[3, :] / 2)
        boxes_covering_fish_top_bottom_format[3, :] = \
            boxes_covering_fish[1, :] - (boxes_covering_fish_top_bottom_format[3, :] / 2)
        # boxes_covering_fish is in format bigX, smallX, bigY, smallY

        boxesList.append(boxes_covering_fish_top_bottom_format)
    boxesInQuestion = boxesList[0]
    boxesList = boxesList[1:]
    cat = np.concatenate(boxesList, axis=1)
    number_of_boxes_per_fish = 11

    # boxes Arrays have the first row representing the biggest X value, the second row representing the smallest X value
    # the third row the biggest Y value, the fourth row the smallest Y value

    # Checking if they crashed in the x dimension
    top_x_coors_boxes_in_question = boxesInQuestion[0, :]
    bottom_x_coors_boxes = cat[1, :]
    grid = np.array(np.meshgrid(top_x_coors_boxes_in_question, bottom_x_coors_boxes))
    tops, bottoms = grid[0, ...], grid[1, ...]
    points_where_top_edge_is_above_bottom_edge = tops > bottoms

    bottom_x_coors_boxes_in_question = boxesInQuestion[1, :]
    top_x_coors_boxes = cat[0, :]
    grid = np.array(np.meshgrid(bottom_x_coors_boxes_in_question, top_x_coors_boxes))
    bottoms, tops = grid[0], grid[1]
    points_where_bottom_edge_is_above_top_edge = bottoms < tops

    points_where_they_crash_in_the_x_dimension = points_where_top_edge_is_above_bottom_edge * \
                                                 points_where_bottom_edge_is_above_top_edge

    # Checking if they crashed in the y dimension
    top_x_coors_boxes_in_question = boxesInQuestion[2, :]
    bottom_x_coors_boxes = cat[3, :]
    grid = np.array(np.meshgrid(top_x_coors_boxes_in_question, bottom_x_coors_boxes))
    tops, bottoms = grid[0, ...], grid[1, ...]
    points_where_top_edge_is_above_bottom_edge = tops > bottoms

    bottom_x_coors_boxes_in_question = boxesInQuestion[3, :]
    top_x_coors_boxes = cat[2, :]
    grid = np.array(np.meshgrid(bottom_x_coors_boxes_in_question, top_x_coors_boxes))
    bottoms, tops = grid[0], grid[1]
    points_where_bottom_edge_is_above_top_edge = bottoms < tops

    points_where_they_crash_in_the_y_dimension = points_where_top_edge_is_above_bottom_edge * \
                                                 points_where_bottom_edge_is_above_top_edge

    points_where_they_crashed = points_where_they_crash_in_the_x_dimension * points_where_they_crash_in_the_y_dimension

    if np.any(points_where_they_crashed):
        # print('Invalid Fish')
        return True
    else:
        # print('Valid Fish')
        return False


def isThisAGoodFishVectList(fishVectList):
    """
        Funtion that checks whether the fishVectList has fishes that are crashing into each other
    :param fishVectList:
    :return:
    """
    boxesList = []
    for fishVectIdx in range(fishVectList.shape[0]):
        fishVect = fishVectList[fishVectIdx, ...]

        seglen, z, x = fishVect[0], fishVect[1], fishVect[2:]
        pt = x_seglen_to_3d_points(x, seglen)
        boxes_covering_fish = addBoxes(pt)

        # Converting them to top bottom format
        boxes_covering_fish_top_bottom_format = np.copy(boxes_covering_fish)
        boxes_covering_fish_top_bottom_format[0, :] = \
            boxes_covering_fish[0, :] + (boxes_covering_fish_top_bottom_format[2, :] / 2)
        boxes_covering_fish_top_bottom_format[1, :] = \
            boxes_covering_fish[0, :] - (boxes_covering_fish_top_bottom_format[2, :] / 2)
        # For the y - dimension
        boxes_covering_fish_top_bottom_format[2, :] = \
            boxes_covering_fish[1, :] + (boxes_covering_fish_top_bottom_format[3, :] / 2)
        boxes_covering_fish_top_bottom_format[3, :] = \
            boxes_covering_fish[1, :] - (boxes_covering_fish_top_bottom_format[3, :] / 2)
        # boxes_covering_fish is in format bigX, smallX, bigY, smallY

        boxesList.append(boxes_covering_fish_top_bottom_format)
    cat = np.concatenate(boxesList, axis=1)
    number_of_boxes_per_fish = 11

    # Considering if they crash in the x dimension
    top_x_coors = cat[0, :]
    bottom_x_coors = cat[1, :]
    grid = np.array(np.meshgrid(top_x_coors, bottom_x_coors))
    top_x_coors = grid[0, ...]
    bottom_x_coors = grid[1, ...]
    pointsWhereTheTopEdgeIsInFrontOfABottomEdges = top_x_coors > bottom_x_coors

    # now the other way around
    bottom_x_coors = cat[1, :]
    top_x_coors = cat[0, :]
    grid = np.array(np.meshgrid(bottom_x_coors, top_x_coors))
    bottom_x_coors = grid[0, ...]
    top_x_coors = grid[1, ...]
    pointsWhereTheBottomEdgesIsBeforeATopEdge = bottom_x_coors < top_x_coors

    didTheyCrashInXArray = pointsWhereTheTopEdgeIsInFrontOfABottomEdges * pointsWhereTheBottomEdgesIsBeforeATopEdge

    # Considering if they crashed in the y dimension
    top_y_coors = cat[2, :]
    bottom_y_coors = cat[3, :]
    grid = np.array(np.meshgrid(top_y_coors, bottom_y_coors))
    top_y_coors = grid[0, ...]
    bottom_y_coors = grid[1, ...]
    pointsWhereTheTopEdgeIsInFrontOfABottomEdges = top_y_coors > bottom_y_coors

    bottom_y_coors = cat[3, :]
    top_y_coors = cat[2, :]
    grid = np.array(np.meshgrid(bottom_y_coors, top_y_coors))
    bottom_y_coors = grid[0, ...]
    top_y_coors = grid[1, ...]
    pointsWhereTheBottomEdgesIsBeforeATopEdge = bottom_y_coors < top_y_coors

    didTheyCrashInYArray = pointsWhereTheTopEdgeIsInFrontOfABottomEdges * pointsWhereTheBottomEdgesIsBeforeATopEdge
    didTheyCrash = didTheyCrashInXArray * didTheyCrashInYArray

    # TODO: vetorize this part
    for fishIdx in range(fishVectList.shape[0]):
        startIdx = fishIdx * number_of_boxes_per_fish
        endIdx = startIdx + number_of_boxes_per_fish
        didTheyCrash[startIdx:endIdx, startIdx:endIdx] = 0

    didTheyCrash = np.any(didTheyCrash)
    print("Crashed :'(") if didTheyCrash else print('Safe')
    if didTheyCrash:
        return False
    else:
        return True


def are_pts_in_bounds(pts):
    """
        Checks if the pts are in within [0,imageSizeX) and [0, imageSizeY)
        pts must have the first row representing the x coors
        and the second row representing the y coors
    :param pts: numpy array
    :return: boolean
    """
    # Turning it into an int to make the definition of it being in bounds concise
    are_xs_in_bounds = np.all(np.logical_and(np.ceil(pts[0, :]) < imageSizeX, pts[0, :] >= 0))
    are_ys_in_bounds = np.all(np.logical_and(np.ceil(pts[1, :]) < imageSizeY, pts[1, :] >= 0))

    is_pts_in_bounds = are_xs_in_bounds and are_ys_in_bounds
    return is_pts_in_bounds


def is_atleast_one_point_in_bounds(pts):
    """
        Function makes sure atleast one of the points are within 0,imageSizeX) and [0, imageSizeY)
    :param pts: numpy array
    :return: boolean
    """
    is_one_x_in = np.logical_and(np.ceil(pts[0, :]) < imageSizeX, pts[0, :] >= 0)
    is_one_y_in = np.logical_and(np.ceil(pts[1, :]) < imageSizeY, pts[1, :] >= 0)
    is_atleast_one_in = np.any(np.logical_and(is_one_x_in, is_one_y_in))
    return is_atleast_one_in


def is_fish_on_edge(pts):
    """
        Function which detects wheter a fish in on an edge.  It is defined being on an edge
        if one of the points is in the image and if one of the points is not in the image
    :param pts: numpy array
    :return: boolean
    """
    # is_one_x_in = np.logical_and( np.ceil( pts[0,:]) < imageSizeX, pts[0,:] >= 0 )
    # is_one_y_in = np.logical_and( np.ceil( pts[1,:]) < imageSizeY, pts[1,:] >= 0 )
    # is_atleast_one_in =np.any( np.logical_and( is_one_x_in, is_one_y_in) )
    is_atleast_one_in = is_atleast_one_point_in_bounds(pts)

    # Assuming of course that atleast one is in
    is_atleast_one_outside = not are_pts_in_bounds(pts)

    is_on_edge = is_atleast_one_in and is_atleast_one_outside

    return is_on_edge


def generateRandomConfiguration(fishInView, fishInEdges, OverlappingFish):
    averageSizeOfFish = Config.averageSizeOfFish
    fishVectList = []

    # generating fishes that are completely in the view, based on their keypoints
    for _ in range(fishInView):
        # Loop to keep on searching in case we randomly generate a fish that was not in the view
        while True:
            # Generating random fishVect
            xVect = np.zeros((11))
            fishlen = (np.random.rand(1) - 0.5) * 30 + averageSizeOfFish
            idxlen = np.floor((fishlen - 62) / 1.05) + 1
            seglen = 5.6 + idxlen * 0.1
            seglen = seglen[0]
            # These fish are on the top plane
            z = 1
            # seglen = 7.1

            x, y = np.random.randint(0, imageSizeX), np.random.randint(0, imageSizeY)
            theta_array_idx = np.random.randint(0, 500000)
            dtheta = theta_array[theta_array_idx, :]
            xVect[:2] = [x, y]
            xVect[2] = np.random.rand(1)[0] * 2 * np.pi
            xVect[3:] = dtheta
            fishVect = np.zeros((13))
            fishVect[0] = seglen
            fishVect[1] = z
            fishVect[2:] = xVect

            # Checking if it is in bounds
            pts = x_seglen_to_3d_points(xVect, seglen)
            if are_pts_in_bounds(pts) == False:
                continue

            # Checking if it interferes
            if doesThisFishInterfereWithTheAquarium(fishVect, fishVectList):
                continue

            # The fish passed all the requirements, so we can add it to the fishVectList
            fishVectList.append(fishVect)
            break

    # Generating fishes in the edges
    # TODO: might want to try a different way, perhaps with shifting the fish
    for _ in range(fishInEdges):

        while True:
            # The possible edge choices for one view are: the left edge, the right edge, the top and the bottom
            # let the numbers 0 - 3 represent these choices respectively
            edgeIdx = np.random.randint(0, 4)
            xVect = np.zeros((11))

            if edgeIdx < 2:
                # We got the left or right, which means the x coordinate is constrained
                if edgeIdx == 0:
                    # We want to generate points around left edge
                    x = (np.random.rand() - .5) * averageSizeOfFish
                    # Got to add some offset if we want to generate all possible conformations
                    # y = np.random.random_integers(0 - averageSizeOfFish/2, imageSizeY + averageSizeOfFish/2)
                    y = np.random.randint(0 - averageSizeOfFish / 2, imageSizeY + averageSizeOfFish / 2)

                else:
                    # We want to generate a fish along the right edge
                    x = (np.random.rand() - .5) * averageSizeOfFish + imageSizeX
                    # Got to add some offset if we want to generate all possible conformations
                    # y = np.random.random_integers(0 - averageSizeOfFish/2, imageSizeY + averageSizeOfFish/2)
                    y = np.random.randint(0 - averageSizeOfFish / 2, imageSizeY + averageSizeOfFish / 2)
            else:
                # We got the top or bottom edge, which means the y coordinate is constrained
                if edgeIdx == 2:
                    # We want to generate a fish around the top edge
                    # x = np.random.random_integers(0 - averageSizeOfFish / 2, imageSizeX + averageSizeOfFish / 2)
                    x = np.random.randint(0 - averageSizeOfFish / 2, imageSizeX + averageSizeOfFish / 2)
                    y = (np.random.rand() - .5) * averageSizeOfFish
                else:
                    # We want to generate a fish around the bottom edge
                    # x = np.random.random_integers(0 - averageSizeOfFish / 2, imageSizeX + averageSizeOfFish / 2)
                    x = np.random.randint(0 - averageSizeOfFish / 2, imageSizeX + averageSizeOfFish / 2)
                    y = (np.random.rand() - .5) * averageSizeOfFish + imageSizeY
            theta_array_idx = np.random.randint(0, 500000)
            dtheta = theta_array[theta_array_idx, :]
            xVect[:2] = [x, y]
            xVect[2] = np.random.rand(1)[0] * 2 * np.pi
            xVect[3:] = dtheta
            fishlen = (np.random.rand(1) - 0.5) * 30 + averageSizeOfFish
            idxlen = np.floor((fishlen - 62) / 1.05) + 1
            seglen = 5.6 + idxlen * 0.1
            seglen = seglen[0]
            fishVect = np.zeros((13))
            fishVect[0] = seglen
            fishVect[1] = 1
            fishVect[2:] = xVect

            pts = x_seglen_to_3d_points(xVect, seglen)
            is_on_edge = is_fish_on_edge(pts)
            if is_on_edge:
                if not doesThisFishInterfereWithTheAquarium(fishVect, fishVectList):
                    fishVectList.append(fishVect)
                    break

    # Generating Overlapping Fishes
    # NOTE: Currently overlapping fish is defined as creating fishes that will overlap the previous fishes
    # might be better if we redefine it something like pairs of overlapping fish, or maybe we can add more pairs

    # clipping the value to stay consistent with the definition above
    OverlappingFish = min(fishInView + fishInEdges, OverlappingFish)

    # Inputs deciding what is the maximum offset when causing the fishes to overlap

    maxOverlappingOffset = Config.maxOverlappingOffset

    overLappingFishVectList = []
    # We also want to overlap the fish in the edges
    fishesToOverlapIdices = random.sample([*range(len(fishVectList))], OverlappingFish)
    for overLappingFishIdx in range(OverlappingFish):

        fishVectToOverlap = fishVectList[fishesToOverlapIdices[overLappingFishIdx]]
        ogSeglen = fishVectToOverlap[0]
        ogXVect = fishVectToOverlap[2:]
        ogPts = x_seglen_to_3d_points(ogXVect, ogSeglen)

        startTime = time.time()
        while True:
            currentTime = time.time()
            duration = currentTime - startTime
            if duration > 120:
                # It is taking too long to generate a fish that satisfies the configuration, lets move on to the next
                break
            # The keypoint idx of the original fish we want to overlap
            ogFishKeypointToOverlap = np.random.randint(0, 12)
            ogPoint = ogPts[:, ogFishKeypointToOverlap]

            # The keypoint of the generated fish we want to use to cause the overlap
            genFishKeypointToOverlap = np.random.randint(0, 12)


            # Generating the fish
            xVect = np.zeros((11))
            fishlen = (np.random.rand(1) - 0.5) * 30 + averageSizeOfFish
            idxlen = np.floor((fishlen - 62) / 1.05) + 1
            seglen = 5.6 + idxlen * 0.1
            seglen = seglen[0]
            # seglen = 7.1

            x, y = np.random.randint(0, imageSizeX), np.random.randint(0, imageSizeY)
            theta_array_idx = np.random.randint(0, 500000)
            dtheta = theta_array[theta_array_idx, :]
            xVect[:2] = [x, y]
            xVect[2] = np.random.rand(1)[0] * 2 * np.pi
            xVect[3:] = dtheta
            fishVect = np.zeros((13))
            fishVect[0] = seglen
            # These fish are on the bottom plane
            fishVect[1] = 2
            fishVect[2:] = xVect
            pts = x_seglen_to_3d_points(xVect, seglen)
            point = pts[:, genFishKeypointToOverlap]
            distance = ogPoint - point
            # adding some randomness
            xOffSet = ((2 * np.random.rand()) - 1) * maxOverlappingOffset
            yOffSet = ((2 * np.random.rand()) - 1) * maxOverlappingOffset
            distance += np.array([xOffSet, yOffSet])
            # Shifting the fish to cause the overlap
            xVect[0] += distance[0]
            xVect[1] += distance[1]
            fishVect[2:] = xVect

            if not doesThisFishInterfereWithTheAquarium(fishVect, overLappingFishVectList):

                # Adding the part that at least one of the points of the fish should be visible
                # this is to stop the fish from disappearing when overlapping the fish on the edges
                # NOTE: it might be a good idea to not have the fish overlap the ones on the edges
                if is_atleast_one_point_in_bounds(pts):
                    overLappingFishVectList.append(fishVect)
                    break

    # fish_list = []
    # # Transforming the fish vectors into fish objects
    # for fishVect in fishVectList + overLappingFishVectList:
    #     fish = Fish(fishVect)
    #     fish_list.append(fish)

    # return fishVectList, overLappingFishVectList
    return fishVectList + overLappingFishVectList

# def generateRandomConfigurationNoLag(fishInView, fishInEdges, OverlappingFish):
#     averageSizeOfFish = Config.averageSizeOfFish
#     fishVectList = []
#
#     # generating fishes that are completely in the view, based on their keypoints
#     for _ in range(fishInView):
#         # Loop to keep on searching in case we randomly generate a fish that was not in the view
#         while True:
#             # Generating random fishVect
#             xVect = np.zeros((11))
#             fishlen = (np.random.rand(1) - 0.5) * 30 + averageSizeOfFish
#             idxlen = np.floor((fishlen - 62) / 1.05) + 1
#             seglen = 5.6 + idxlen * 0.1
#             seglen = seglen[0]
#             # These fish are on the top plane
#             z = 1
#             # seglen = 7.1
#
#             x, y = np.random.randint(0, imageSizeX), np.random.randint(0, imageSizeY)
#             theta_array_idx = np.random.randint(0, 500000)
#             dtheta = theta_array[theta_array_idx, :]
#             xVect[:2] = [x, y]
#             xVect[2] = np.random.rand(1)[0] * 2 * np.pi
#             xVect[3:] = dtheta
#             fishVect = np.zeros((13))
#             fishVect[0] = seglen
#             fishVect[1] = z
#             fishVect[2:] = xVect
#
#             # Checking if it is in bounds
#             pts = x_seglen_to_3d_points(xVect, seglen)
#             if are_pts_in_bounds(pts) == False:
#                 continue
#
#             # Checking if it interferes
#             if doesThisFishInterfereWithTheAquarium(fishVect, fishVectList):
#                 continue
#
#             # The fish passed all the requirements, so we can add it to the fishVectList
#             fishVectList.append(fishVect)
#             break
#
#     # Generating fishes in the edges
#     # TODO: might want to try a different way, perhaps with shifting the fish
#     for _ in range(fishInEdges):
#
#         while True:
#             # The possible edge choices for one view are: the left edge, the right edge, the top and the bottom
#             # let the numbers 0 - 3 represent these choices respectively
#             edgeIdx = np.random.randint(0, 4)
#             xVect = np.zeros((11))
#
#             if edgeIdx < 2:
#                 # We got the left or right, which means the x coordinate is constrained
#                 if edgeIdx == 0:
#                     # We want to generate points around left edge
#                     x = (np.random.rand() - .5) * averageSizeOfFish
#                     # Got to add some offset if we want to generate all possible conformations
#                     # y = np.random.random_integers(0 - averageSizeOfFish/2, imageSizeY + averageSizeOfFish/2)
#                     y = np.random.randint(0 - averageSizeOfFish / 2, imageSizeY + averageSizeOfFish / 2)
#
#                 else:
#                     # We want to generate a fish along the right edge
#                     x = (np.random.rand() - .5) * averageSizeOfFish + imageSizeX
#                     # Got to add some offset if we want to generate all possible conformations
#                     # y = np.random.random_integers(0 - averageSizeOfFish/2, imageSizeY + averageSizeOfFish/2)
#                     y = np.random.randint(0 - averageSizeOfFish / 2, imageSizeY + averageSizeOfFish / 2)
#             else:
#                 # We got the top or bottom edge, which means the y coordinate is constrained
#                 if edgeIdx == 2:
#                     # We want to generate a fish around the top edge
#                     # x = np.random.random_integers(0 - averageSizeOfFish / 2, imageSizeX + averageSizeOfFish / 2)
#                     x = np.random.randint(0 - averageSizeOfFish / 2, imageSizeX + averageSizeOfFish / 2)
#                     y = (np.random.rand() - .5) * averageSizeOfFish
#                 else:
#                     # We want to generate a fish around the bottom edge
#                     # x = np.random.random_integers(0 - averageSizeOfFish / 2, imageSizeX + averageSizeOfFish / 2)
#                     x = np.random.randint(0 - averageSizeOfFish / 2, imageSizeX + averageSizeOfFish / 2)
#                     y = (np.random.rand() - .5) * averageSizeOfFish + imageSizeY
#             theta_array_idx = np.random.randint(0, 500000)
#             dtheta = theta_array[theta_array_idx, :]
#             xVect[:2] = [x, y]
#             xVect[2] = np.random.rand(1)[0] * 2 * np.pi
#             xVect[3:] = dtheta
#             fishlen = (np.random.rand(1) - 0.5) * 30 + averageSizeOfFish
#             idxlen = np.floor((fishlen - 62) / 1.05) + 1
#             seglen = 5.6 + idxlen * 0.1
#             seglen = seglen[0]
#             fishVect = np.zeros((13))
#             fishVect[0] = seglen
#             fishVect[1] = 1
#             fishVect[2:] = xVect
#
#             pts = x_seglen_to_3d_points(xVect, seglen)
#             is_on_edge = is_fish_on_edge(pts)
#             if is_on_edge:
#                 if not doesThisFishInterfereWithTheAquarium(fishVect, fishVectList):
#                     fishVectList.append(fishVect)
#                     break
#
#     # Generating Overlapping Fishes
#     # NOTE: Currently overlapping fish is defined as creating fishes that will overlap the previous fishes
#     # might be better if we redefine it something like pairs of overlapping fish, or maybe we can add more pairs
#
#     # clipping the value to stay consistent with the definition above
#     OverlappingFish = min(fishInView + fishInEdges, OverlappingFish)
#
#     # Inputs deciding what is the maximum offset when causing the fishes to overlap
#
#     maxOverlappingOffset = Config.maxOverlappingOffset
#
#     overLappingFishVectList = []
#     # We also want to overlap the fish in the edges
#     fishesToOverlapIdices = random.sample([*range(len(fishVectList))], OverlappingFish)
#     for overLappingFishIdx in range(OverlappingFish):
#
#         fishVectToOverlap = fishVectList[fishesToOverlapIdices[overLappingFishIdx]]
#         ogSeglen = fishVectToOverlap[0]
#         ogXVect = fishVectToOverlap[2:]
#         ogPts = x_seglen_to_3d_points(ogXVect, ogSeglen)
#
#         startTime = time.time()
#         while True:
#             currentTime = time.time()
#             duration = currentTime - startTime
#             if duration > 120:
#                 # It is taking too long to generate a fish that satisfies the configuration, lets move on to the next
#                 break
#             # The keypoint idx of the original fish we want to overlap
#             ogFishKeypointToOverlap = np.random.randint(0, 12)
#             ogPoint = ogPts[:, ogFishKeypointToOverlap]
#
#             point_of_new_fish = ogPoint + np.array([ 2*(np.random.rand() -1) * maxOverlappingOffset,
#                                                      2*(np.random.rand() -1) * maxOverlappingOffset])
#
#             fishlen = (np.random.rand(1) - 0.5) * 30 + averageSizeOfFish
#             idxlen = np.floor((fishlen - 62) / 1.05) + 1
#             seglen = 5.6 + idxlen * 0.1
#             seglen = seglen[0]
#             theta0 = np.random.rand(1)[0] * 2 * np.pi
#
#             good_dtheta_indices = get_good_dtheta_indices( point_of_new_fish[0] , point_of_new_fish[1] ,theta0,seglen,overLappingFishVectList,fishVectToOverlap)
#
#             amount_of_good = len((theta_array)[good_dtheta_indices, ...])
#             # None of those points where good
#             if amount_of_good == 0 : continue
#
#             random_dtheta = (theta_array)[good_dtheta_indices, ...][np.random.randint(0, amount_of_good)]
#
#             fishVect = np.zeros((13))
#             fishVect[0] = seglen
#             fishVect[1] = 2
#             fishVect[2] = point_of_new_fish[0]
#             fishVect[3] = point_of_new_fish[1]
#             fishVect[4] = theta0
#             fishVect[5:] = random_dtheta
#
#             overLappingFishVectList.append(fishVect)
#             break
#             # # The keypoint of the generated fish we want to use to cause the overlap
#             # genFishKeypointToOverlap = np.random.randint(0, 12)
#             #
#             #
#             # # Generating the fish
#             # xVect = np.zeros((11))
#             #
#             # # seglen = 7.1
#             #
#             # x, y = np.random.randint(0, imageSizeX), np.random.randint(0, imageSizeY)
#             # theta_array_idx = np.random.randint(0, 500000)
#             # dtheta = theta_array[theta_array_idx, :]
#             # xVect[:2] = [x, y]
#             # xVect[2] = np.random.rand(1)[0] * 2 * np.pi
#             # xVect[3:] = dtheta
#             # fishVect = np.zeros((13))
#             # fishVect[0] = seglen
#             # # These fish are on the bottom plane
#             # fishVect[1] = 2
#             # fishVect[2:] = xVect
#             # pts = x_seglen_to_3d_points(xVect, seglen)
#             # point = pts[:, genFishKeypointToOverlap]
#             # distance = ogPoint - point
#             # # adding some randomness
#             # xOffSet = ((2 * np.random.rand()) - 1) * maxOverlappingOffset
#             # yOffSet = ((2 * np.random.rand()) - 1) * maxOverlappingOffset
#             # distance += np.array([xOffSet, yOffSet])
#             # # Shifting the fish to cause the overlap
#             # xVect[0] += distance[0]
#             # xVect[1] += distance[1]
#             # fishVect[2:] = xVect
#             #
#             # if not doesThisFishInterfereWithTheAquarium(fishVect, overLappingFishVectList):
#             #
#             #     # Adding the part that at least one of the points of the fish should be visible
#             #     # this is to stop the fish from disappearing when overlapping the fish on the edges
#             #     # NOTE: it might be a good idea to not have the fish overlap the ones on the edges
#             #     if is_atleast_one_point_in_bounds(pts):
#             #         overLappingFishVectList.append(fishVect)
#             #         break
#
#     # fish_list = []
#     # # Transforming the fish vectors into fish objects
#     # for fishVect in fishVectList + overLappingFishVectList:
#     #     fish = Fish(fishVect)
#     #     fish_list.append(fish)
#
#     # return fishVectList, overLappingFishVectList
#     return fishVectList + overLappingFishVectList


def generateRandomConfigurationNoLagChunks(fishInView, fishInEdges, OverlappingFish):
    averageSizeOfFish = Config.averageSizeOfFish
    fishVectList = []

    # generating fishes that are completely in the view, based on their keypoints
    for _ in range(fishInView):
        # Loop to keep on searching in case we randomly generate a fish that was not in the view
        while True:
            # Generating random fishVect
            xVect = np.zeros((11))
            fishlen = (np.random.rand(1) - 0.5) * 30 + averageSizeOfFish
            idxlen = np.floor((fishlen - 62) / 1.05) + 1
            seglen = 5.6 + idxlen * 0.1
            seglen = seglen[0]
            # These fish are on the top plane
            z = 1
            # seglen = 7.1

            x, y = np.random.randint(0, imageSizeX), np.random.randint(0, imageSizeY)
            theta_array_idx = np.random.randint(0, 500000)
            dtheta = theta_array[theta_array_idx, :]
            xVect[:2] = [x, y]
            xVect[2] = np.random.rand(1)[0] * 2 * np.pi
            xVect[3:] = dtheta
            fishVect = np.zeros((13))
            fishVect[0] = seglen
            fishVect[1] = z
            fishVect[2:] = xVect

            # Checking if it is in bounds
            pts = x_seglen_to_3d_points(xVect, seglen)
            if are_pts_in_bounds(pts) == False:
                continue

            # Checking if it interferes
            if doesThisFishInterfereWithTheAquarium(fishVect, fishVectList):
                continue

            # The fish passed all the requirements, so we can add it to the fishVectList
            fishVectList.append(fishVect)
            break

    # Generating fishes in the edges
    # TODO: might want to try a different way, perhaps with shifting the fish
    for _ in range(fishInEdges):

        while True:
            # The possible edge choices for one view are: the left edge, the right edge, the top and the bottom
            # let the numbers 0 - 3 represent these choices respectively
            edgeIdx = np.random.randint(0, 4)
            xVect = np.zeros((11))

            if edgeIdx < 2:
                # We got the left or right, which means the x coordinate is constrained
                if edgeIdx == 0:
                    # We want to generate points around left edge
                    x = (np.random.rand() - .5) * averageSizeOfFish
                    # Got to add some offset if we want to generate all possible conformations
                    # y = np.random.random_integers(0 - averageSizeOfFish/2, imageSizeY + averageSizeOfFish/2)
                    y = np.random.randint(0 - averageSizeOfFish / 2, imageSizeY + averageSizeOfFish / 2)

                else:
                    # We want to generate a fish along the right edge
                    x = (np.random.rand() - .5) * averageSizeOfFish + imageSizeX
                    # Got to add some offset if we want to generate all possible conformations
                    # y = np.random.random_integers(0 - averageSizeOfFish/2, imageSizeY + averageSizeOfFish/2)
                    y = np.random.randint(0 - averageSizeOfFish / 2, imageSizeY + averageSizeOfFish / 2)
            else:
                # We got the top or bottom edge, which means the y coordinate is constrained
                if edgeIdx == 2:
                    # We want to generate a fish around the top edge
                    # x = np.random.random_integers(0 - averageSizeOfFish / 2, imageSizeX + averageSizeOfFish / 2)
                    x = np.random.randint(0 - averageSizeOfFish / 2, imageSizeX + averageSizeOfFish / 2)
                    y = (np.random.rand() - .5) * averageSizeOfFish
                else:
                    # We want to generate a fish around the bottom edge
                    # x = np.random.random_integers(0 - averageSizeOfFish / 2, imageSizeX + averageSizeOfFish / 2)
                    x = np.random.randint(0 - averageSizeOfFish / 2, imageSizeX + averageSizeOfFish / 2)
                    y = (np.random.rand() - .5) * averageSizeOfFish + imageSizeY
            theta_array_idx = np.random.randint(0, 500000)
            dtheta = theta_array[theta_array_idx, :]
            xVect[:2] = [x, y]
            xVect[2] = np.random.rand(1)[0] * 2 * np.pi
            xVect[3:] = dtheta
            fishlen = (np.random.rand(1) - 0.5) * 30 + averageSizeOfFish
            idxlen = np.floor((fishlen - 62) / 1.05) + 1
            seglen = 5.6 + idxlen * 0.1
            seglen = seglen[0]
            fishVect = np.zeros((13))
            fishVect[0] = seglen
            fishVect[1] = 1
            fishVect[2:] = xVect

            pts = x_seglen_to_3d_points(xVect, seglen)
            is_on_edge = is_fish_on_edge(pts)
            if is_on_edge:
                if not doesThisFishInterfereWithTheAquarium(fishVect, fishVectList):
                    fishVectList.append(fishVect)
                    break

    # Generating Overlapping Fishes
    # NOTE: Currently overlapping fish is defined as creating fishes that will overlap the previous fishes
    # might be better if we redefine it something like pairs of overlapping fish, or maybe we can add more pairs

    # clipping the value to stay consistent with the definition above
    OverlappingFish = min(fishInView + fishInEdges, OverlappingFish)

    # Inputs deciding what is the maximum offset when causing the fishes to overlap

    maxOverlappingOffset = Config.maxOverlappingOffset

    overLappingFishVectList = []
    # We also want to overlap the fish in the edges
    fishesToOverlapIdices = random.sample([*range(len(fishVectList))], OverlappingFish)
    for overLappingFishIdx in range(OverlappingFish):

        fishVectToOverlap = fishVectList[fishesToOverlapIdices[overLappingFishIdx]]
        ogSeglen = fishVectToOverlap[0]
        ogXVect = fishVectToOverlap[2:]
        ogPts = x_seglen_to_3d_points(ogXVect, ogSeglen)

        startTime = time.time()
        while True:
            currentTime = time.time()
            duration = currentTime - startTime
            if duration > 120:
                # It is taking too long to generate a fish that satisfies the configuration, lets move on to the next
                break
            # The keypoint idx of the original fish we want to overlap
            ogFishKeypointToOverlap = np.random.randint(0, 12)
            ogPoint = ogPts[:, ogFishKeypointToOverlap]

            point_of_new_fish = ogPoint + np.array([ 2*(np.random.rand() -1) * maxOverlappingOffset,
                                                     2*(np.random.rand() -1) * maxOverlappingOffset])

            fishlen = (np.random.rand(1) - 0.5) * 30 + averageSizeOfFish
            idxlen = np.floor((fishlen - 62) / 1.05) + 1
            seglen = 5.6 + idxlen * 0.1
            seglen = seglen[0]
            theta0 = np.random.rand(1)[0] * 2 * np.pi

            # the chunk number represents which part of the theta array we are considering
            chunk = np.random.randint(0,10)
            good_dtheta_indices = get_good_dtheta_indices_chuncks(
                point_of_new_fish[0],point_of_new_fish[1],theta0,seglen,overLappingFishVectList,fishVectToOverlap,chunk)

            amount_of_good = len((theta_array[50000 * chunk:50000 * (chunk + 1),...])[good_dtheta_indices, ...])
            # None of those points where good
            if amount_of_good == 0 : continue

            random_dtheta = (theta_array[50000 * chunk :50000 * (chunk + 1),...])[good_dtheta_indices, ...][np.random.randint(0, amount_of_good)]

            fishVect = np.zeros((13))
            fishVect[0] = seglen
            fishVect[1] = 2
            fishVect[2] = point_of_new_fish[0]
            fishVect[3] = point_of_new_fish[1]
            fishVect[4] = theta0
            fishVect[5:] = random_dtheta

            overLappingFishVectList.append(fishVect)
            break
            # # The keypoint of the generated fish we want to use to cause the overlap
            # genFishKeypointToOverlap = np.random.randint(0, 12)
            #
            #
            # # Generating the fish
            # xVect = np.zeros((11))
            #
            # # seglen = 7.1
            #
            # x, y = np.random.randint(0, imageSizeX), np.random.randint(0, imageSizeY)
            # theta_array_idx = np.random.randint(0, 500000)
            # dtheta = theta_array[theta_array_idx, :]
            # xVect[:2] = [x, y]
            # xVect[2] = np.random.rand(1)[0] * 2 * np.pi
            # xVect[3:] = dtheta
            # fishVect = np.zeros((13))
            # fishVect[0] = seglen
            # # These fish are on the bottom plane
            # fishVect[1] = 2
            # fishVect[2:] = xVect
            # pts = x_seglen_to_3d_points(xVect, seglen)
            # point = pts[:, genFishKeypointToOverlap]
            # distance = ogPoint - point
            # # adding some randomness
            # xOffSet = ((2 * np.random.rand()) - 1) * maxOverlappingOffset
            # yOffSet = ((2 * np.random.rand()) - 1) * maxOverlappingOffset
            # distance += np.array([xOffSet, yOffSet])
            # # Shifting the fish to cause the overlap
            # xVect[0] += distance[0]
            # xVect[1] += distance[1]
            # fishVect[2:] = xVect
            #
            # if not doesThisFishInterfereWithTheAquarium(fishVect, overLappingFishVectList):
            #
            #     # Adding the part that at least one of the points of the fish should be visible
            #     # this is to stop the fish from disappearing when overlapping the fish on the edges
            #     # NOTE: it might be a good idea to not have the fish overlap the ones on the edges
            #     if is_atleast_one_point_in_bounds(pts):
            #         overLappingFishVectList.append(fishVect)
            #         break

    # fish_list = []
    # # Transforming the fish vectors into fish objects
    # for fishVect in fishVectList + overLappingFishVectList:
    #     fish = Fish(fishVect)
    #     fish_list.append(fish)

    # return fishVectList, overLappingFishVectList
    return fishVectList + overLappingFishVectList










