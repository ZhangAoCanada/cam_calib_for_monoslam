from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
from multiprocessing import Process

IF_RESIZE = False

def ReadTxtFile(txt_filepath):
    """
    Functionality:
        Read the manually selected calibration images index from the txt file.
    """
    all_index = []
    with open(txt_filepath) as f:
        for line in f:
            all_index.append([int(i) for i in line.split(",")])

    all_index = np.array(all_index)
    all_index = all_index[0]

    return all_index

def checkoutDir(dir_name):
    """
    Function:
        Checkout if the directory is exists, if not then create one

    Args:
        dir_name            ->          directory name
    """
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

def FindChessBoardCorner(images_selected, left_calibration_directory, chess_board_size,
                        grid_size, max_calibration_numimg = 10000, plot_corners = True):
    """
    Functionality:
        Find the chessboard of all the left and right calibrations images. If all points found, return all 
        necessary paramters for further calibration and rectification.
    """

    # flags of findChessboardCorners()
    chessboard_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)

    # prepare object points
    object_points = np.zeros((chess_board_size[0] * chess_board_size[1], 3), np.float32)
    object_points[:,:2] = grid_size * np.mgrid[0:chess_board_size[0], 0:chess_board_size[1]].T.reshape(-1,2).astype(np.float32)

    # find out how many manual seleted images
    num_seleted_images = len(images_selected)
    if max_calibration_numimg > num_seleted_images:
        max_calibration_numimg = num_seleted_images
    
    # arrays to store object points and image points from all the images
    objpoints = []
    objpointsL = []

    imgpointsL = []
    imgpointsL_single = []
    
    for i in range(max_calibration_numimg):
        # read selected images index
        current_img_number = images_selected[i]
        N = str(current_img_number)

        # read the left selected image
        img_path_l = left_calibration_directory + N + ".jpg"
        img_l = cv2.imread(img_path_l)
        if IF_RESIZE:
            img_l = cv2.resize(img_l, None, fx=0.5, fy=0.5)

        if img_l is None:
            continue

        # transfer the image into gray for better result
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)

        # pass image size for further use
        image_size = gray_l.shape[::-1]

        # find the chessboard on the gray image
        ret_l, corners_l = cv2.findChessboardCorners(gray_l, chess_board_size, chessboard_flags)

        # if found, add object points, image points
        if (ret_l == True):
            print("current number:\t", current_img_number)

            objpoints.append(object_points)
            objpointsL.append(object_points)

            cv2.cornerSubPix(gray_l, corners_l, (5,5), (-1,-1), criteria)

            imgpointsL.append(corners_l)
            imgpointsL_single.append(corners_l)
            
            # plot the chessboard corners to verify the qualtiy of the images
            if plot_corners:
                print(current_img_number)
                img_display_l = cv2.drawChessboardCorners(img_l, chess_board_size, corners_l, ret_l)
                chessboard_display = img_display_l
                small_display = cv2.resize(chessboard_display, (0,0), fx=0.5, fy=0.5) 
                cv2.imshow('display', small_display)
                # cv2.imwrite("/DATA/calib_ao/stereo_corners/" + str(current_img_number) + ".jpg", small_display)
                cv2.waitKey(500)
        
    if plot_corners:
        cv2.destroyAllWindows()

    # print the number of images that stored for rectification, left intrinsic calibration and 
    # right instrinsic calibration
    print("---------See how many images are saved-----------")
    print("Total image:\t", len(objpoints))

    return objpoints, imgpointsL, image_size, objpointsL, imgpointsL_single


def CalibrationAndRectification(objpoints, imgpointsL, image_size, objpointsL, imgpointsL_single):
    """
    Functionality:
        Use the result of the chessboard parameters in FindChessBoardCorner() to do the calibration and rectification.
    """
    # Do the single camera calibration first, in order to get a better result
    single_cali_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
    # ret_l, camMatrix_l, distCoe_l, revc_l, tvecs_l = cv2.calibrateCamera(objpointsL, imgpointsL_single, image_size, None, None, criteria = single_cali_criteria)
    ret_l, camMatrix_l, distCoe_l, revc_l, tvecs_l = cv2.calibrateCamera(objpoints, imgpointsL_single, image_size, None, None, criteria = single_cali_criteria)

    return camMatrix_l, distCoe_l

def SaveCalibrationAndRectificationParameters(chess_board_size, grid_size, \
            calibration_indexes, left_calibration_directory, calib_txt):
    """
    Main Function:
        Store all parameters as numpy arrays.
    """
    objpoints, imgpointsL, image_size, objpointsL, imgpointsL_single = FindChessBoardCorner(calibration_indexes, left_calibration_directory, 
                                                                                chess_board_size, grid_size, plot_corners = True)
    cam_Matrix, cam_Distort = CalibrationAndRectification(objpoints, imgpointsL, image_size, objpointsL, imgpointsL_single)
    
    print("Final calibration matrix is:")
    print(cam_Matrix)
    with open(calib_txt, "w") as ff:
        for i in range(cam_Matrix.shape[0]):
            for j in range(cam_Matrix.shape[1]):
                data = cam_Matrix[i,j]
                ff.write(str(data) +  " ")
            ff.write("\n")

def main(chess_board_size, grid_size, calibration_indexes, calib_txt):
    """
    Input necessary parameters here.
    """
    p = Process(target=SaveCalibrationAndRectificationParameters, args=(chess_board_size, \
                grid_size, calibration_indexes, left_calibration_directory, calib_txt))
    p.start()
    p.join()
    
if __name__ == "__main__":
    # how many corners there are in the chessboard
    chess_board_size = (8, 6)
    # size of each grid, please make the unit to "meters"
    grid_size = 0.030
    # create indexes
    all_index = ReadTxtFile("./selected.txt")
    # image directory
    left_calibration_directory = "./calib_images/"
    # calibration matrix file location
    calib_txt = "../mono_cv_cpp/intrinsic.txt"
    # main
    main(chess_board_size, grid_size, all_index, calib_txt)

