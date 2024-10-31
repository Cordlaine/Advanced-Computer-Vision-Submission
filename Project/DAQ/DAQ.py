import pyzed.sl as sl
import cv2
import numpy as np
import argparse
import os.path
import os


def take_image_zed(zed, dir, folders, number, show_only=False):
    imageL, imageR = sl.Mat(), sl.Mat()
    imageLun, imageRun = sl.Mat(), sl.Mat()
    depth_view = sl.Mat()
    depth = sl.Mat()
    disparity = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(imageL, sl.VIEW.LEFT)  # Get the left image
        zed.retrieve_image(imageR, sl.VIEW.RIGHT)
        zed.retrieve_image(imageLun, sl.VIEW.LEFT_UNRECTIFIED)  # Get the left image
        zed.retrieve_image(imageRun, sl.VIEW.RIGHT_UNRECTIFIED)  # Get the left image
        zed.retrieve_image(depth_view, sl.VIEW.DEPTH)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        zed.retrieve_measure(disparity, sl.MEASURE.DISPARITY)
    d = depth.get_data()
    dis = disparity.get_data()

    imL, imR = imageL.get_data(), imageR.get_data()
    imL_un, imR_un = imageLun.get_data(), imageRun.get_data()
    if show_only:
        h, w = int(imL.shape[0]*0.3), int(imL.shape[1]*0.3)
        cv2.imshow(f'ZED - RGB', np.hstack((cv2.resize(imL, (w, h)), cv2.resize(imR, (w, h)))))
        # cv2.imshow(f'ZED - right RGB {number}', imR)
        h, w = int(depth_view.get_data().shape[0]*0.5), int(depth_view.get_data().shape[1]*0.5)
        cv2.imshow(f'ZED - DEPTH_VIEW', cv2.resize(depth_view.get_data(), (w, h)))
    else:

        ret, corners = cv2.findChessboardCorners(cv2.cvtColor(imL, cv2.COLOR_RGB2GRAY), (10, 9), cv2.CALIB_CB_ADAPTIVE_THRESH)
        # ret, corners = cv2.findChessboardCorners(cv2.cvtColor(imL, cv2.COLOR_RGB2GRAY), (6,7), cv2.CALIB_CB_ADAPTIVE_THRESH)
        '''
        print('np.mean(d*dis)')
        k = d * dis
        k[k > 10000] = np.nan
        print(np.nanmean(k))
        '''
        print('ZED Chessboard:', ret)
        cv2.imwrite(os.path.join(dir, folders[0], number + ".png"), imL)
        cv2.imwrite(os.path.join(dir, folders[1], number + ".png"), imR)
        cv2.imwrite(os.path.join(dir, folders[3], number + ".png"), imL_un)
        cv2.imwrite(os.path.join(dir, folders[4], number + ".png"), imR_un)
        np.save(os.path.join(dir, folders[2], number + ".npy"), d)
        np.save(os.path.join(dir, folders[5], number + ".npy"), dis)

def zed_init(calibration_file=None):
    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
    init_params.camera_fps = 15  # Set fps at 30
    if calibration_file is not None:
        init_params.optional_opencv_calibration_file = calibration_file
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    return init_params

def yes_or_no(question):
    while "the answer is invalid":
        reply = str(input(question + ' (y/n)')).lower().strip()
        if reply[:1] == 'y':
            return True
        if reply[:1] == 'n':
            return False

if __name__ == '__main__':
    # Saving Data from ZED-Camera and 2xO3R-Camera
    # Args handling -> check help parameters to understand
    parser = argparse.ArgumentParser(description='Camera')
    parser.add_argument('--path', type=str, default='data/240718_test', help='path for saved images')
    parser.add_argument('--mode', type=str, default='recording', help='calibration or recording')
    parser.add_argument('--scene', type=str, default='scene_01', help='name of the scene')
    calib = True
    filter_default = True
    image_index_to_start = 1  # delfault = 1 == calib not zero

    args = parser.parse_args()
    dir = os.path.abspath(args.path)

    ''' ZED '''
    # Create a Camera object
    zed = sl.Camera()
    zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 50)

    stereoParam = {}
    #init_params = zed_init('zed_calibration.yml')
    init_params_default = zed_init()

    err = zed.open(init_params_default)#init_params)

    if err != sl.ERROR_CODE.SUCCESS:
        print('ZED nicht verbunden oder file not !')
        exit(1)
    calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
    # Focal length of the left eye in pixels
    intrinsics_zed = np.identity(3)
    distortion_zed = calibration_params.left_cam.disto
    intrinsics_zed[0, 0] = calibration_params.left_cam.fx
    intrinsics_zed[1, 1] = calibration_params.left_cam.fy
    intrinsics_zed[0, 2] = calibration_params.left_cam.cx
    intrinsics_zed[1, 2] = calibration_params.left_cam.cy
    # First radial distortion coefficient
    # print(intrinsics_zed)
    stereoParam['intrinsics_zedL'] = intrinsics_zed.tolist()
    stereoParam['distortion_zedL'] = distortion_zed.tolist()
    zed.close()


    if not os.path.exists(dir):
        os.mkdir(dir)
    else:
        print(f'|||||FILENAME:{dir} already existing||||||')
        print('if you proceed existing files may be overwritten')
        if not yes_or_no('proceed?'):
            exit(0)

    directory = {'ZED': ['RGB_left', 'RGB_right', 'depth','RGB_left_unrectified','RGB_right_unrectified','disparity'],
                 'ZED_default': ['RGB_left', 'RGB_right', 'depth', 'RGB_left_unrectified', 'RGB_right_unrectified',
                                 'disparity'],
                 'Calib': ''}

    for folder in directory:
        complete_dir = os.path.join(dir, folder)
        if not os.path.exists(complete_dir):
            os.mkdir(complete_dir)
        for subfolder in directory[folder]:
            complete_dir = os.path.join(dir, folder, subfolder)
            if not os.path.exists(complete_dir):
                os.mkdir(complete_dir)

    i = image_index_to_start

    while 'recording':
        err = zed.open(init_params_default)
        index = args.scene + '_' + f"{i:04}"
        while True:
            take_image_zed(zed, dir, directory['ZED'], number=index, show_only=True)
            k = cv2.waitKey(10)
            if k == 32:  # LEERTASTE
                break
            elif k == 13:  # ENTER
                print('ENDE')
                exit(0)
        take_image_zed(zed, os.path.join(dir, 'ZED'), directory['ZED'], number=index)
        i+=1
