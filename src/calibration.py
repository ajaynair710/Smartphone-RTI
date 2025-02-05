import pickle
import cv2 as cv
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from src.typing import Frame, Size2D
from src.stream import VideoStream
from utils.io import InputSanitizationUtils as ISUtils, PathUtils
import cv2

from numpy.typing import NDArray

from utils.misc import Timer, default
from utils.io import SilentLogger
from utils.io import BaseLogger
from src.typing import Views


@dataclass
class CalibratedCamera:
    ''' Dataclass to store camera calibration coefficients. '''

    camera_mat       : NDArray        # 3x3 Intrinsic Camera Matrix
    distortion_coeffs: NDArray        # 1x5 Distortion Coefficients
    params           : Dict[str, Any] # Camera Calibration Hyperparameters
    white_mask       : bool = False   # Whether to fill empty pixels with white

    @classmethod
    def from_points(
        cls,
        obj_points: List[NDArray],
        img_points: List[NDArray],
        size      : Size2D,
        params    : Dict[str, Any] | None = None,
        logger    : BaseLogger            = SilentLogger()
    ) -> 'CalibratedCamera':
        ''' Calibrates the camera using object and image points. '''

        WARN_THRESH = 1

        params_ : Dict = default(params, {})

        # Check if the number of object and image points are equal
        if len(obj_points) != len(img_points):
            logger.handle_error(
                msg=f"Number of object points and image points must be equal. Got {len(obj_points)} and {len(img_points)}",
                exception=ValueError
            )

        logger.info(msg=f"Starting camera calibration for {len(obj_points)} samples ...")
        timer = Timer()

        # Calibrate the camera
        ret, camera_mat, distortion_coeffs, _, _ = cv.calibrateCamera( # type: ignore
            obj_points, img_points, size, None, None                   # type: ignore
        )

        if not ret: logger.handle_error(msg="Camera calibration failed. ", exception=RuntimeError)
        else:
            logger.info(msg=f"Camera calibration completed in {timer} with calibration error: {ret} pixels.")
            if ret > WARN_THRESH: logger.warning(msg=f"Calibration error is too high (> {WARN_THRESH}). Consider recalibrating the camera.")

        return cls(
            camera_mat=camera_mat,
            distortion_coeffs=distortion_coeffs,
            params=params_ | {"reprojection_error": ret},
        )
    
    @classmethod
    def trivial_calibration(cls, size: Size2D) -> 'CalibratedCamera':
        ''' Create a trivial camera calibration with no distortion. '''

        w, h = size
        max_ = max(*size)

        return cls(
            camera_mat=np.array([
                [max_,     0, w // 2],
                [   0,  max_, h // 2],
                [   0,     0,      1]
            ]),
            distortion_coeffs=np.zeros((1, 5)),
            params={"reprojection_error": None}
        )

        

    @classmethod
    def from_pickle(cls, path: str, logger: BaseLogger = SilentLogger()) -> 'CalibratedCamera':
        ''' Load camera calibration from a pickle file. '''

        logger.info(msg=f"Loading camera calibration from {path}")

        with open(path, 'rb') as f: return pickle.load(f)

    def __str__(self) -> str:

        # Camera Matrix
        K_str = "Intrisic Camera Matrix\n"
        col_widths = [max(len(f"{row[i]:.6f}") for row in self.camera_mat) for i in range(len(self.camera_mat[0]))]
        for row in self.camera_mat:
            K_str += " | ".join(f"{val:>{col_widths[i]}.6f}" for i, val in enumerate(row)) + "\n"

        # Distortion Coefficients
        dist_str = "Distortion Coefficients\n"
        dist_str += " | ".join(f"{val:.6f}" for val in self.distortion_coeffs[0]) + "\n"

        # Mean Reprojection Error
        error_str = f"Mean Pixel Error: {self.params.get('reprojection_error', None)}\n"

        return f"{K_str}\n{dist_str}\n{error_str}"

    def __repr__(self) -> str: return str(self)

    def undistort(self, img: Frame) -> Frame:
        ''' Undistort an image using the camera calibration coefficients. '''

        # Perform undistortion
        undistorted = cv.undistort(img, self.camera_mat, self.distortion_coeffs)

        # White mask
        if self.white_mask:

            # Create a white probe image
            probe_img = np.full_like(img, 255)
            probe_img_undistorted = cv.undistort(probe_img, self.camera_mat, self.distortion_coeffs)

            # Create a mask for empty pixels
            if len(probe_img.shape) == 3: mask = (probe_img_undistorted == 0).all(axis=2)  # For color images
            else:                         mask = (probe_img_undistorted == 0)              # For grayscale images

            # Fill empty pixels with white
            undistorted[mask] = 255

        return undistorted

    def dump(
        self,
        path   : str,
        logger : BaseLogger = SilentLogger(),
        verbose: bool       = False
    ) -> None:
        ''' Save the camera calibration to a pickle file. '''

        logger_verbose = logger if verbose else SilentLogger()

        ISUtils.check_output(path=PathUtils.get_folder_path(path=path), logger=logger_verbose)

        logger.info(msg=f"Saving camera calibration to {path}")

        with open(path, 'wb') as f:
            pickle.dump(self, f)


class CalibratedVideoStream(VideoStream):

    def __init__(
        self, 
        path        : str, 
        calibration : CalibratedCamera, 
        name        : str        = '',
        logger      : BaseLogger = SilentLogger(), 
        verbose     : bool       = False,
    ):

        super().__init__(path=path, name=name, logger=logger, verbose=verbose)

        self._calibration: CalibratedCamera = calibration

    @property
    def _str_name(self) -> str: return 'CalibratedVideoStream'

    def _process_frame(self, frame: Frame, frame_id: int) -> Views:

        views = super()._process_frame(frame=frame, frame_id=frame_id)

        # Undistort the frame
        frame_undistorted = self._calibration.undistort(views['raw'].copy())

        return views | {'undistorted': frame_undistorted}

class CameraCalibration:
    def __init__(self, camera_matrix, dist_coeffs, rvecs=None, tvecs=None):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.rvecs = rvecs
        self.tvecs = tvecs
        
    def undistort(self, image):
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)
        
    def dump(self, path):
        np.savez(path, 
                 camera_matrix=self.camera_matrix,
                 dist_coeffs=self.dist_coeffs,
                 rvecs=self.rvecs,
                 tvecs=self.tvecs)
    
    @classmethod
    def load(cls, path):
        data = np.load(path)
        return cls(data['camera_matrix'], 
                  data['dist_coeffs'],
                  data['rvecs'],
                  data['tvecs'])