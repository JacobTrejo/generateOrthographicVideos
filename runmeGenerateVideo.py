from Programs.Config import Config
from Programs.Aquarium import Aquarium
from Programs.programsForGeneratingFish import x_seglen_to_3d_points, addBoxes
import os
import shutil
import multiprocessing
import numpy as np
import time
from scipy.io import loadmat
import cv2 as cv
import pdb

fish = loadmat('x_files_4_dim_rotated/000.mat')
fish2 = loadmat('x_files_4_dim_rotated/001.mat')
fish3 = loadmat('x_files_4_dim_rotated/100.mat')['x_vid'][:100,:]
fish =  fish['x_vid']
fish2 = fish2['x_vid']

homepath = Config.dataDirectory

if not os.path.exists(homepath[:-1]):
   os.makedirs(homepath[:-1])
# # Not resting it no more because it is strange, should try looking for a better function
# else:
#    # reset it
#    shutil.rmtree(homepath)
#    os.makedirs(homepath[:-1])

folders = [Config.videoDirectory,Config.poseAnnotationsDirectory,Config.fishNumbersDirectory]
subFolders = []
for folder in folders:
   subPath = homepath + folder
   if not os.path.exists(subPath):
       os.makedirs(subPath)

fishVectList = np.load('Inputs/fish_list/zebrafish_000000.npy')

#print(fishVectList.shape)

fishVectList[:,0,2:] = fish[:100,:]
fishVectList[:,1,2:] = fish2[:100,:]

fishVectList[...,2] += 100
fishVectList[...,3] += 100
fishVectList[:,0,2:4] += 50
fishVectList[:,0,4] += np.pi

fishVectList[:,0,1] = 1
fishVectList[:,1,1] = 2

# Giving it extra motion
#fishVectList[:,0,2] += np.array(list(range(100))) 
#fishVectList[:,0,3] += np.array(list(range(100)))

# Adding a 3d fish
#temp = np.zeros((100,13))
#temp[:,2:] = anikets3
#anikets3 = temp

temp = np.zeros((100,2))
fish3 = np.concatenate((temp, fish3), axis=1)
# Setting the fish length
fish3[:,0] = fishVectList[:,0,0]
# Moving it away from the corner
fish3[:, 2:4] += 90
fish3[:,1] = 3

fishVectList = np.concatenate( (fishVectList, fish3[:, None, :]), axis = 1)


# Just Checking to make sure a z dimension was passed, if not we will add one with the first fish being closest to the camera
amount_of_frames, amount_of_fish, size_of_x_vector = fishVectList.shape[:3]


if size_of_x_vector == 12:
    # it is missing the z componenet, lets add it after the segment length
    newFishVectList = np.zeros((amount_of_frames, amount_of_fish, 13))
    newFishVectList[...,0] = fishVectList[..., 0]
    newFishVectList[...,2:] = fishVectList[...,1:]
    newFishVectList[...,1] = np.stack([  np.array(list(range(amount_of_fish))) for _ in range(amount_of_frames) ], axis = 0   )
    fishVectList = newFishVectList    


aquarium = Aquarium(0, fishVectList = fishVectList)
aquarium.draw_video()





