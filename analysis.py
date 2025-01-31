import os
from dotenv import load_dotenv
import cv2 as cv
from src.marker import MarkerDetector
from src.mlic import (
    MLICDynamicCameraVideoStream, 
    MLICStaticCameraVideoStream, 
    MLICCollector, 
    MLIC
)
from src.threshold import (
    AdaptiveThresholding, 
    BaseThresholding, 
    OtsuThresholding, 
    Thresholding, 
    TopHatOtsuThresholding
)
from src.calibration import CalibratedCamera
from src.interpolation import RTIRadialBasisInterpolator, MLICBasisInterpolator, RTIPolynomialTextureMapInterpolator
from utils.io import FileLogger, IOUtils, VideoFile
from utils.misc import Timer

load_dotenv()

OUT_DIR = os.getenv('OUT_DIR', '.')
EXP_NAME = 'coin1'

# Calibration and video paths
CAMERA_1 = 'cam1static'
CAMERA_2 = 'cam2_moving_light'
CALIBRATION_1 = None
CALIBRATION_2 = os.path.join(OUT_DIR, EXP_NAME, 'calibration', f'{CAMERA_2}.pkl')
CAMERA_EXT = 'mp4'
VIDEO_1 = os.path.join(OUT_DIR, EXP_NAME, 'sync', f'{EXP_NAME}_static.{CAMERA_EXT}')
VIDEO_2 = os.path.join(OUT_DIR, EXP_NAME, 'sync', f'{EXP_NAME}_dynamic.{CAMERA_EXT}')

C_1 = CalibratedCamera.from_pickle(CALIBRATION_1) if CALIBRATION_1 else CalibratedCamera.trivial_calibration(size=VideoFile(VIDEO_1).metadata.size)
C_2 = CalibratedCamera.from_pickle(CALIBRATION_2) if CALIBRATION_2 else CalibratedCamera.trivial_calibration(size=VideoFile(VIDEO_2).metadata.size)

# Thresholding types
BINARY_TYPE_1 = 'otsu'
BINARY_TYPE_2 = 'adaptive'

def get_binary_thresholding(type_: str) -> Thresholding:
    match type_:
        case 'threshold': return BaseThresholding(t=65, kernel_size=(27, 27))
        case 'otsu': return OtsuThresholding()
        case 'tophat': return TopHatOtsuThresholding(kernel_size=(191, 191), kernel_shape=cv.MORPH_ELLIPSE)
        case 'adaptive': return AdaptiveThresholding(block_size=117, c=10)
        case _: raise ValueError(f'Invalid binary type: {type_}')

MLIC_SIDE = 256
SKIP_FRAMES = 1
WIN_SCALE = 0.25
WIN_SQUARE = 250
PLOT_HISTORY = True

DETECTOR_PARAMS = {
    'white_thresh': 230,
    'black_thresh': 25,
    'min_area': 200,
    'max_area_prop': 0.5
}

MLIC_DIR = os.path.join(OUT_DIR, EXP_NAME, 'mlic')

INTERPOLATION_ALGO = 'rbf'
INTERPOLATION_SIZE = (48, 48)
PROGRESS = 1000

if __name__ == '__main__':
    # Initialize logger
    logger = FileLogger(file=os.path.join(MLIC_DIR, f'mlic.log'))
    IOUtils.make_dir(path=OUT_DIR)

    # MLIC collection
    logger.info('STARTING MLIC COLLECTION')
    marker_detector = MarkerDetector(**DETECTOR_PARAMS)
    static_thresholding = get_binary_thresholding(BINARY_TYPE_1)
    dynamic_thresholding = get_binary_thresholding(BINARY_TYPE_2)

    mlic_static = MLICStaticCameraVideoStream(
        path=VIDEO_1,
        calibration=C_1,
        thresholding=static_thresholding,
        marker_detector=marker_detector,
        mlic_side=MLIC_SIDE
    )

    mlic_dynamic = MLICDynamicCameraVideoStream(
        path=VIDEO_2,
        calibration=C_2,
        thresholding=dynamic_thresholding,
        marker_detector=marker_detector,
        plot_history=PLOT_HISTORY
    )

    mlic_collector = MLICCollector(
        mlic_static=mlic_static,
        mlic_dynamic=mlic_dynamic,
        logger=logger
    )
    mlic = mlic_collector.collect(skip_frames=SKIP_FRAMES, win_rect_scale=WIN_SCALE, win_square_side=WIN_SQUARE)

    # Save MLIC results
    mlic_path = os.path.join(MLIC_DIR, 'mlic.pkl')
    mlic.dump(mlic_path, logger=logger)

    # Interpolation
    INTERPOLATION_DIR = os.path.join(OUT_DIR, EXP_NAME, 'interpolation', INTERPOLATION_ALGO)
    IOUtils.make_dir(path=INTERPOLATION_DIR)
    logger = FileLogger(file=os.path.join(INTERPOLATION_DIR, f'interpolation.log'))

    match INTERPOLATION_ALGO:
        case 'rbf': rti_interpolation = RTIRadialBasisInterpolator
        case 'ptm': rti_interpolation = RTIPolynomialTextureMapInterpolator
        case _: raise ValueError(f'Invalid interpolation algorithm {INTERPOLATION_ALGO}.')

    mlic = MLIC.from_pickle(path=mlic_path, logger=logger)
    mlic_bi = MLICBasisInterpolator(
        mlic=mlic,
        C_rti_interpolator=rti_interpolation,
        interpolation_size=INTERPOLATION_SIZE,
        logger=logger
    )
    bi_collection = mlic_bi.get_interpolation_collection(progress=PROGRESS)

    # Save interpolation results
    output_filename = f"{EXP_NAME}_{INTERPOLATION_ALGO}.pkl"
    bi_collection.dump(path=os.path.join(INTERPOLATION_DIR, output_filename), logger=logger)
