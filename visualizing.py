import numpy as np
import cv2 as cv
from Programs.Config import Config


videoName = 'zebrafish_000000.mp4'
name = videoName[:-4]

fish_numbers = np.load(Config.dataDirectory + Config.fishNumbersDirectory + name + '.npy')

pose_annotations = np.load(Config.dataDirectory + Config.poseAnnotationsDirectory + name + '.npy')
amount_of_frames = len(pose_annotations)
amount_of_fish = len(pose_annotations[0])


# cap = cv.VideoCapture(videoName)
cap = cv.VideoCapture(Config.dataDirectory + Config.videoDirectory + videoName)

out = cv.VideoWriter('visualizing_result.avi', cv.VideoWriter_fourcc('M','J','P','G'), Config.fps, (Config.imageSizeX, Config.imageSizeY))


for frame_idx in range(amount_of_frames):
    ret, im = cap.read()
    for fishIdx in range(amount_of_fish):
        fish_number = fish_numbers[frame_idx, fishIdx]

        pts = pose_annotations[frame_idx, fishIdx]
        # Its necessary to pass in integers
        pts = pts.astype(int)
        if (Config.shouldMask and fish_number != 0) or not(Config.shouldMask):
            for ptIdx in range(12):
                color = (0,255,0)
                if ptIdx > 9: color = (0,0,255)
                im = cv.circle(im, (pts[0,ptIdx], pts[1,ptIdx]), radius=2, color=color, thickness=-1)

        if fish_number != 0:
            org = (pts[0,1], pts[1,1] - 10)
            font = cv.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            thickness = 1

            im = cv.putText(im, str(int( fish_number)), org, font, fontScale, [0,255,0], thickness, cv.LINE_AA)

    out.write(im)
out.release()
cap.release()





