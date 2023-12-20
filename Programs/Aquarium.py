import random
import warnings
from Programs.Config import Config
import numpy as np
import cv2 as cv
# from Programs.programsForGeneratingFish import generateRandomConfiguration, generateRandomConfigurationNoLag, generateRandomConfigurationNoLagChunks
from Programs.Auxilary import add_noise_static_noise, add_noise_static_noise_background, add_patchy_noise, mergeViews, createDepthArr
from Programs.Auxilary import roundHalfUp, compute_var_mat_for_patchy_noise, add_patchy_noise_from_vars_mat, return_fish_random_parameters
# from Programs.programsForDrawingImage import f_x_to_model_bigger
from Programs.programsForDrawingImageNoRandom import f_x_to_model_bigger
import scipy.signal as sig
from scipy.ndimage.morphology import binary_dilation
import time
from tqdm import tqdm

# Configuration variables
imageSizeY, imageSizeX = Config.imageSizeY, Config.imageSizeX

class Aquarium:
    # Static Variables
    aquariumVariables = ['fishInAllViews', 'fishInEdges','overlapping']
    fishVectListKey = 'fishVectList'

    def overloaded_constructor(self, **kwargs):
        aquariumVariablesDict = {'fishInAllViews':0, 'fishInEdges':0, 'overlapping':0}

        wasAnAquariumVariableDetected = False
        wasAnAquariumPassed = False
        for key in kwargs:
            if key in Aquarium.aquariumVariables:
                aquariumVariablesDict[key] = kwargs.get(key)
                wasAnAquariumVariableDetected = True
            if key is Aquarium.fishVectListKey:
                wasAnAquariumPassed = True
                fishVectList = kwargs.get(key)

        # if not wasAnAquariumPassed:
        #     if wasAnAquariumVariableDetected:
        #         fishVectList = self.generateFishListGivenVariables(aquariumVariablesDict)
        #     else:
        #         fishesInView = np.random.randint(0, Config.maxFishesInView)
        #         fishesInEdge = np.random.poisson(Config.averageFishInEdges)
        #         overlappingFish = 0
        #         for _ in range(fishesInView + fishesInEdge):
        #             shouldItOverlap = True if np.random.rand() < Config.overlappingFishFrequency else False
        #             if shouldItOverlap: overlappingFish += 1
        #         # fishVectList = generateRandomConfiguration(fishesInView, fishesInEdge, overlappingFish)
        #         fishVectList = generateRandomConfigurationNoLagChunks(fishesInView, fishesInEdge, overlappingFish)

        return fishVectList

    # def generateFishListGivenVariables(self, aquariumVariablesDict):
    #     fishInAllViews = aquariumVariablesDict.get('fishInAllViews')
    #     overlapping = aquariumVariablesDict.get('overlapping')
    #     fishInEdges = aquariumVariablesDict.get('fishInEdges')
    #     # fishVectList = generateRandomConfiguration(fishInAllViews, fishInEdges, overlapping)
    #     # fishVectList = generateRandomConfigurationNoLag(fishInAllViews, fishInEdges, overlapping)
    #     fishVectList = generateRandomConfigurationNoLagChunks(fishInAllViews, fishInEdges, overlapping)
    #     return fishVectList

    def __init__(self, frame_idx, **kwargs):
        # Getting the configuration settings
        maxFishesInView = Config.maxFishesInView
        averageFishInEdges = Config.averageFishInEdges
        overlappingFishFrequency = Config.overlappingFishFrequency
        self.shouldAddStaticNoise = Config.shouldAddStaticNoise
        self.shouldAddPatchyNoise = Config.shouldAddStaticNoise
        self.shouldSaveAnnotations = Config.shouldSaveAnnotations
        self.shouldSaveImages = Config.shouldSaveImages

        self.fishVectList = self.overloaded_constructor(**kwargs)
        self.amount_of_frames = len(self.fishVectList)
        self.amount_of_fish = len(self.fishVectList[0])
        self.random_fish_parameters = np.array([ return_fish_random_parameters() for _ in range(self.amount_of_fish) ])
        self.fish_numbers_array = np.zeros((self.amount_of_frames, self.amount_of_fish))
        self.previous_visibility = np.zeros((self.amount_of_fish))
        # variable to track which fish have currently been seen
        self.max_fish_seen = 0
        self.pose_annotations = np.zeros((self.amount_of_frames, self.amount_of_fish, 2, 12))
        self.fish_list = []
        # for fishVect in fishVectList:
        #     fish = Fish(fishVect)
        #     self.fish_list.append(fish)

        # self.filter_size = 2 * roundHalfUp(np.random.rand()) + 3
        # self.sigma = np.random.rand() + 0.5
        # self.gN1 = (np.random.rand() * np.random.normal(50, 10)) / 255
        # self.gN2 = (np.random.rand() * 50 + 20) / 255 ** 2
        # self.next_frame_to_update = np.random.poisson(Config.updateRate)
        self.filter_size = None
        self.sigma = None
        self.gN1 = None
        self.gN2 = None
        self.finalVarsMat = None
        self.next_frame_to_update = 0
        self.frame_count_since_last_update = 0

        self.finalMask = np.zeros((imageSizeY, imageSizeX))

        self.strIdxInFormat = format(frame_idx, '06d')

        self.views_list = []
        self.finalViews = []
        self.frame_idx = frame_idx
        # NOTE: the following variable is more of a constant
        self.amount_of_cameras = 1

    def should_update_randomness(self):
        if self.frame_count_since_last_update == self.next_frame_to_update:
            self.filter_size = 2 * roundHalfUp(np.random.rand()) + 3
            self.sigma = np.random.rand() + 0.5
            self.gN1 = (np.random.rand() * np.random.normal(50, 10)) / 255
            self.gN2 = (np.random.rand() * 50 + 20) / 255 ** 2
            self.frame_count_since_last_update = 0
            self.next_frame_to_update = np.random.poisson(Config.updateRate)
            self.finalVarsMat = compute_var_mat_for_patchy_noise(self.finalViews[0][0], self.fish_list)
        self.frame_count_since_last_update += 1


    def add_static_noise_to_views(self):
        for viewIdx, view in enumerate(self.finalViews):
            graymodel = view[0]
            depth = view[1]
            #noisey_graymodel = add_noise_static_noise(graymodel, self.filter_size, self.sigma, self.gN1, self.gN2)
            noisey_graymodel = add_noise_static_noise_background(graymodel, self.filter_size, self.sigma, self.gN1, self.gN2)
            #noisey_graymodel = add_noise_static_noise_background(graymodel)
            # TODO: dont use tuples since they are immutable
            noisey_view = (noisey_graymodel, depth)
            # updating
            self.finalViews[viewIdx] = noisey_view

    # def add_static_noise_to_views(self):
    #     for viewIdx, view in enumerate(self.finalViews):
    #         graymodel = view[0]
    #         depth = view[1]
    #         noisey_graymodel = add_noise_static_noise(graymodel)
    #         # TODO: dont use tuples since they are immutable
    #         noisey_view = (noisey_graymodel, depth)
    #         # updating
    #         self.finalViews[viewIdx] = noisey_view

    # def add_patchy_noise_to_views(self):
    #     for viewIdx, view in enumerate(self.finalViews):
    #         graymodel = view[0]
    #         depth = view[1]
    #         noisey_graymodel = add_patchy_noise(graymodel, self.fish_list)
    #         # TODO: dont use tuples since they are immutable
    #         noisey_view = (noisey_graymodel, depth)
    #         # updating
    #         self.finalViews[viewIdx] = noisey_view


    def add_patchy_noise_to_views(self):
        for viewIdx, view in enumerate(self.finalViews):
            graymodel = view[0]
            depth = view[1]
            # noisey_graymodel = add_patchy_noise(graymodel, self.fish_list)
            noisey_graymodel = add_patchy_noise_from_vars_mat(graymodel, self.finalVarsMat)
            # TODO: dont use tuples since they are immutable
            noisey_view = (noisey_graymodel, depth)
            # updating
            self.finalViews[viewIdx] = noisey_view

    def save_annotations(self):
        biggestIdx4TrainingData = Config.biggestIdx4TrainingData
        dataDirectory = Config.dataDirectory

        subFolder = 'train/' if self.frame_idx < biggestIdx4TrainingData else 'val/'
        labelsPath = dataDirectory + '/' + 'labels/' + subFolder
        strIdxInFormat = format(self.frame_idx, '06d')
        filename = 'zebrafish_' + strIdxInFormat + '.txt'
        labelsPath += filename

        # Creating the annotations
        f = open(labelsPath, 'w')

        for fish in (self.fish_list):
            # for fish in (fishVectList + overlappingFishVectList):
            boundingBox = fish.boundingBox

            # Should add a method to the bounding box, boundingBox.isSmallFishOnEdge()
            if fish.is_valid_fish:
                f.write(str(0) + ' ')
                f.write(
                    str(boundingBox.getCenterX() / imageSizeX) + ' ' + str(boundingBox.getCenterY() / imageSizeY) + ' ')
                f.write(
                    str(boundingBox.getWidth() / imageSizeX) + ' ' + str(boundingBox.getHeight() / imageSizeY) + ' ')

                xArr = fish.xs
                yArr = fish.ys
                vis = fish.vis
                for pointIdx in range(12):
                    # Visibility is set to zero if they are out of bounds
                    # Just got to clip them so that YOLO does not throw an error
                    x = np.clip(xArr[pointIdx], 0, imageSizeX - 1)
                    y = np.clip(yArr[pointIdx], 0, imageSizeY - 1)
                    f.write(str(x / imageSizeX) + ' ' + str(y / imageSizeY)
                            + ' ' + str(int(vis[pointIdx])) + ' ')
                f.write('\n')

    def save_image(self):
        biggestIdx4TrainingData = Config.biggestIdx4TrainingData
        dataDirectory = Config.dataDirectory

        subFolder = 'train/' if self.frame_idx < biggestIdx4TrainingData else 'val/'
        imagesPath = dataDirectory + '/' + 'images/' + subFolder
        strIdxInFormat = format(self.frame_idx, '06d')
        filename = 'zebrafish_' + strIdxInFormat + '.png'
        imagesPath += filename
        cv.imwrite(imagesPath, self.finalViews[0][0])

    # For Debbuging
    def get_image(self):
        return self.finalViews[0][0]


    def draw(self):
        # drawing the fishes
        for fish in self.fish_list:
            fish.draw()
            self.views_list.append(fish.views)

        # merging the images
        if len(self.views_list) != 0:
            self.finalViews = mergeViews(self.views_list)
        else:
            for viewIdx in range(self.amount_of_cameras):
                view = (np.zeros((imageSizeY, imageSizeX)), np.zeros((imageSizeY, imageSizeX)))
                self.finalViews.append(view)

        # updating the visibility for the cases where a fish ends up covering another fish
        for fishIdx, fish in enumerate(self.fish_list):
            fish.update_visibility(self.finalViews)
            # You have update the fish list, because python is wierd
            self.fish_list[fishIdx] = fish

        if self.shouldAddStaticNoise:
            self.add_static_noise_to_views()

        if self.shouldAddPatchyNoise:
            self.add_patchy_noise_to_views()

    def draw_video(self):
        fourcc = cv.VideoWriter_fourcc(*'avc1')
        #fourcc = cv.VideoWriter_fourcc(*'DIVX')
        
        # Not this one
        # out = cv.VideoWriter('video.mp4',cv.VideoWriter_fourcc(*'DIVX'), 100, (bFrameSizeX, bFrameSizeY * 3),False)

        out = cv.VideoWriter(Config.dataDirectory + Config.videoDirectory +
                             "zebrafish_" + self.strIdxInFormat + '.mp4', fourcc, Config.fps, (imageSizeX, imageSizeY), False)
        print('Generating a video with ', self.amount_of_fish, 'fish for ',self.amount_of_frames, ' frames')

        for frame_number in tqdm(range(self.amount_of_frames)):
            frame = self.draw_frame(frame_number)
            out.write(frame.astype(np.uint8))

            # reseting some variables
            self.views_list = []
            self.finalViews = []

        out.release()
        np.save(Config.dataDirectory + Config.poseAnnotationsDirectory +
                'zebrafish_' + self.strIdxInFormat + '.npy', self.pose_annotations)
        np.save(Config.dataDirectory + Config.fishNumbersDirectory +
                'zebrafish_' + self.strIdxInFormat + '.npy', self.fish_numbers_array)

    def draw_frame(self, frame_number):
        fishVectListForFrame = self.fishVectList[frame_number]
        self.fish_list = []
        for fishIdx, fishVect in enumerate(fishVectListForFrame):
            fish = Fish(fishVect, self.random_fish_parameters[fishIdx])
            self.fish_list.append(fish)
        for fishIdx, fish in enumerate(self.fish_list):
            fish.draw()
            self.views_list.append(fish.views)
            self.pose_annotations[frame_number, fishIdx] = fish.pts

            # Setting the number of the fish
            was_this_fish_visible = self.previous_visibility[fishIdx]
            is_this_fish_visible = fish.visible
            if is_this_fish_visible:
                if was_this_fish_visible:
                    # It can still be marked as the same fish
                    self.fish_numbers_array[frame_number, fishIdx] = self.fish_numbers_array[frame_number - 1, fishIdx]
                else:
                    # This fish appeared
                    self.max_fish_seen += 1
                    self.fish_numbers_array[frame_number, fishIdx] = self.max_fish_seen
            # updating for the next frame
            self.previous_visibility[fishIdx] = is_this_fish_visible

        if len(self.views_list) != 0:
            self.finalViews = mergeViews(self.views_list)
        else:
            for viewIdx in range(self.amount_of_cameras):
                view = [np.zeros((imageSizeY, imageSizeX)), np.zeros((imageSizeY, imageSizeX))]
                self.finalViews.append(view)
        self.should_update_randomness()
        
        self.add_static_noise_to_views()
        #self.add_patchy_noise_to_views()
        
        if Config.shouldMask:
            for fishIdx, fish in enumerate(self.fish_list):
                # fish.mask
                self.finalMask = np.add(self.finalMask, fish.mask)
                self.finalMask[self.finalMask > 0] = 1
                # self.finalMask += fish.mask
            self.finalViews[0] = (self.finalMask.astype(int) * self.finalViews[0][0], self.finalViews[0][1])
            self.finalMask = np.zeros((imageSizeY, imageSizeX))
            print(np.max(self.finalViews[0][0]))
        return self.finalViews[0][0]


class Fish:
    class BoundingBox:
        BoundingBoxThreshold = Config.boundingBoxThreshold

        def __init__(self, smallY, bigY, smallX, bigX):
            self.smallY = smallY
            self.bigY = bigY
            self.smallX = smallX
            self.bigX = bigX

        def getHeight(self):
            return (self.bigY - self.smallY)

        def getWidth(self):
            return (self.bigX - self.smallX)

        def getCenterX(self):
            return ((self.bigX + self.smallX) / 2)

        def getCenterY(self):
            return ((self.bigY + self.smallY) / 2)

        def isValidBox(self):
            height = self.getHeight()
            width = self.getWidth()

            if (height <= Fish.BoundingBox.BoundingBoxThreshold) or (width <= Fish.BoundingBox.BoundingBoxThreshold):
                return False
            else:
                return True

    def __init__(self, fishVect, randomness_vector):
        self.seglen = fishVect[0]
        # self.seglen = 2.4
        self.z = fishVect[1]
        self.x = fishVect[2:]
        self.randomness_vector = randomness_vector
        self.visible = 1
        self.mask = np.zeros((imageSizeY, imageSizeX))

    def draw(self):
        graymodel, pts = f_x_to_model_bigger(self.x, self.seglen, Config.randomizeFish, imageSizeX, imageSizeY, self.randomness_vector)

        depth = np.ones(pts[0,:].shape) * self.z

        depth_im = createDepthArr(graymodel, pts[0,:], pts[1,:], depth)
        # TODO: fill out these depth images since for the orthographic projections the fish can have spots
        camera1View = (graymodel, depth_im)
        self.views = [camera1View]

        self.pts = pts
        self.graymodel = graymodel

        self.vis = np.zeros((pts.shape[1]))
        self.vis[self.valid_points_masks] = 1

        if len(self.vis[self.valid_points_masks]) < 12: self.visible = 0
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            if self.visible: self.mask = np.nan_to_num(graymodel / graymodel)

        # startTime = time.time()
        if Config.shouldAddNoiseToMask:
            outline = binary_dilation(np.nan_to_num(graymodel), np.ones((3, 3))) - self.mask

            yIndices, xIndices = np.nonzero(outline)
            amount_of_indices = len(yIndices)
            amount_of_points_to_dilate = \
                np.random.randint(0, np.clip( Config.maxAmountOfPointsToDilate, 0, amount_of_indices) + 1)
            if not(self.visible): amount_of_points_to_dilate = 0
            indices_4_indices = random.sample(range(amount_of_indices), amount_of_points_to_dilate)
            tempArr = np.zeros((amount_of_points_to_dilate, imageSizeY, imageSizeX))
            arrIndices = [*range(amount_of_points_to_dilate)]
            yIndices = list(yIndices[indices_4_indices])
            xIndices = list(xIndices[indices_4_indices])

            tempArr[arrIndices, yIndices, xIndices] = 1

            for point_to_dialate_idx in range(amount_of_points_to_dilate):
                radius = np.random.randint(1, Config.maxDilationRadius)
                tempArr[point_to_dialate_idx] =  binary_dilation( tempArr[point_to_dialate_idx], np.ones((radius, radius)) )

            if len(tempArr) != 0:
                tempArr = np.max(tempArr, axis=0)
                self.mask = np.maximum(tempArr, self.mask)
        # endTime = time.time()
        # print('duration: ', endTime - startTime)

        # marking the depth of the points, will be used later to find their visibility
        marked_depth_at_keypoints = depth_im[self.intYs[self.valid_points_masks],
                                             self.intXs[self.valid_points_masks]]
        self.depth = np.zeros(self.xs.shape)
        self.depth[self.valid_points_masks] = marked_depth_at_keypoints


        # Creating the bounding box
        nonzero_coors = np.array(np.where(graymodel > 0))
        try:
            smallY = np.min(nonzero_coors[0, :])
            bigY = np.max(nonzero_coors[0, :])
            smallX = np.min(nonzero_coors[1, :])
            bigX = np.max(nonzero_coors[1, :])
        except:
            smallY = 0
            bigY = 0
            smallX = 0
            bigX = 0
        self.boundingBox = Fish.BoundingBox(smallY, bigY, smallX, bigX)

    @property
    def xs(self):
        return self.pts[0, :]

    @property
    def ys(self):
        return self.pts[1, :]

    @property
    def intXs(self):
        return np.ceil(self.pts[0, :]).astype(int)

    @property
    def intYs(self):
        return np.ceil(self.pts[1, :]).astype(int)

    @property
    def valid_points_masks(self):
        xs = self.intXs
        ys = self.intYs
        xs_in_bounds = (xs < imageSizeX) * (xs >= 0)
        ys_in_bounds = (ys < imageSizeY) * (ys >= 0)
        return xs_in_bounds * ys_in_bounds

    def amount_of_vis_points(self):
        val_xs = self.pts[0, :][self.valid_points_masks]
        return val_xs.shape[0]

    def update_visibility(self, finalViews):
        finalView1 = finalViews[0]
        finalDepth = finalView1[1]

        previous_marked_depths = self.depth[self.valid_points_masks]
        final_marked_depths = finalDepth[self.intYs[self.valid_points_masks],
                                         self.intXs[self.valid_points_masks]]
        still_vis = final_marked_depths == previous_marked_depths

        # have to do it this way because python is wierd with the references
        tempVis = np.ones((self.vis).shape)
        tempVis[self.valid_points_masks] = still_vis
        self.vis *= tempVis

    @property
    def is_valid_fish(self):
        if (self.amount_of_vis_points() >= 1) and self.boundingBox.isValidBox():
            return True
        else:
            return False

