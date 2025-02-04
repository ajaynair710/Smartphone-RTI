import os
import cv2
import numpy as np
import pickle

# Constants for camera calibration settings
CAMERA = 'cam2_moving_light' 
WINDOW_SIZE = (576, 324)  # Display window size for visualization
VIDEO_PATH = os.path.join('data', CAMERA, 'calibration.mp4')  # Path to input video
CALIBRATION_DIR = 'output'  # Directory to store calibration data
CALIBRATION_FILE = os.path.join(CALIBRATION_DIR, f'{CAMERA}.pkl')  # Output calibration file
CHESSBOARD_SIZE = (9, 6)  # Chessboard pattern size (number of inner corners per row and column)

class ChessboardCameraCalibrator:
    """
    A class to perform camera calibration using a chessboard pattern.
    It extracts chessboard corners from a video and computes the camera matrix and distortion coefficients.
    """
    def __init__(self, path, pattern_size, skip_frames):
        """
        Initializes the calibrator by loading the video and preparing chessboard object points.

        Parameters:
        - path (str): Path to the video file.
        - pattern_size (tuple): Number of inner corners in the chessboard pattern (cols, rows).
        - skip_frames (int): Number of frames to skip between processing to improve efficiency.
        """
        self.video_path = path
        self._chessboard_size = pattern_size
        self.skip_frames = skip_frames
        
        # Open video file
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {path}")
        
        # Extract video metadata
        self.metadata = self._get_video_metadata()
        
        # Lists to store object points and image points
        self.object_points = []  # 3D points in real-world space
        self.image_points = []  # Corresponding 2D points in image plane
        
        # Prepare object points grid (assuming square size of 1 unit)
        self.objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    def calibrate(self, window_size=WINDOW_SIZE):
        """
        Performs camera calibration by detecting chessboard corners in video frames.

        Parameters:
        - window_size (tuple): Window size for displaying frames (optional).

        Returns:
        - CameraCalibration: Object containing camera calibration parameters.

        Raises:
        - ValueError: If no chessboard patterns are detected in the frames.
        """
        frame_idx = 0  # Frame counter
        gray = None  # Placeholder for grayscale frame
        
        while True:
            ret, frame = self.cap.read()  # Read a frame from the video
            if not ret:
                break  # Stop if end of video is reached
            
            if frame_idx % (self.skip_frames + 1) == 0:  # Process only selected frames
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                ret, corners = cv2.findChessboardCorners(gray, self._chessboard_size, None)  # Find chessboard
                
                if ret:
                    # Refine corner detection for better accuracy
                    corners2 = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1),
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    )
                    # Store object points and corresponding image points
                    self.object_points.append(self.objp)
                    self.image_points.append(corners2)
                    
                    # Draw detected chessboard corners on the frame
                    cv2.drawChessboardCorners(frame, self._chessboard_size, corners2, ret)
                
                # Resize frame for better visualization
                if window_size:
                    h, w = frame.shape[:2]
                    scale = min(window_size[0] / w, window_size[1] / h)
                    frame = cv2.resize(frame, None, fx=scale, fy=scale)
                
                # Show the processed frame
                cv2.imshow('Calibration', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if 'q' is pressed
                    break
            
            frame_idx += 1  # Increment frame index
        
        # Cleanup
        cv2.destroyAllWindows()
        self.cap.release()
        
        if gray is None:
            raise ValueError("No frames were processed, or no chessboard patterns were detected.")
        
        # Perform camera calibration if chessboard corners were found
        if self.object_points:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                self.object_points, self.image_points, gray.shape[::-1], None, None
            )
            return SaveCameraCalibration(mtx, dist, rvecs, tvecs)
        else:
            raise ValueError("No calibration patterns detected")

    def _get_video_metadata(self):
        """
        Extracts metadata such as resolution, FPS, and frame count from the video file.

        Returns:
        - dict: Contains video size, FPS, total frame count, and duration.
        """
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        return {
            "size": (width, height),
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
        }

class SaveCameraCalibration:
    """
    A class to storing camera calibration data.
    """
    def __init__(self, camera_matrix, dist_coeffs, rvecs=None, tvecs=None):
        """
        Initializes the camera calibration object.

        Parameters:
        - camera_matrix (np.array): Camera matrix.
        - dist_coeffs (np.array): Distortion coefficients.
        - rvecs (list): Rotation vectors.
        - tvecs (list): Translation vectors.
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.rvecs = rvecs
        self.tvecs = tvecs

    def dump(self, path):
        """
        Saves the calibration data to a file using pickle.

        Parameters:
        - path (str): Path to save the calibration data.
        """
        with open(path, 'wb') as f:
            pickle.dump({
                'camera_matrix': self.camera_matrix,
                'dist_coeffs': self.dist_coeffs,
                'rvecs': self.rvecs,
                'tvecs': self.tvecs,
            }, f)

if __name__ == "__main__":
    print(f"Initializing calibration process for camera '{CAMERA}'.")

    if not os.path.isfile(VIDEO_PATH):
        print(f"Error: Video file not found at {VIDEO_PATH}")
        exit(1)

    print(f"Video file found at {VIDEO_PATH}")
    calibrator = ChessboardCameraCalibrator(VIDEO_PATH, CHESSBOARD_SIZE, skip_frames=40)

    try:
        print("Starting camera calibration...")
        camera_calibration = calibrator.calibrate()
        print("Calibration completed successfully.")
        camera_calibration.dump(CALIBRATION_FILE)
        print(f"Calibration data saved to {CALIBRATION_FILE}")
    except Exception as e:
        print(f"Error during calibration: {e}")
        exit(1)
