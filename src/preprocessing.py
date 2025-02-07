import math
import os
import numpy as np
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from numpy.typing import NDArray
from scipy.signal import correlate
import ffmpeg
import cv2 as cv
import cv2
from src.calibration import CalibratedCamera, CameraCalibration


class VideoSync:
    AUDIO_EXT = 'wav'
    VIDEO_EXT = 'mp4'

    def __init__(self, video1, video2, out_dir):
        exp1 = os.path.splitext(os.path.basename(video1.path))[0]
        exp2 = os.path.splitext(os.path.basename(video2.path))[0]

        if exp1 != exp2:
            raise ValueError(f"Input videos must have the same experiment name, got {exp1} and {exp2}.")

        self._video1 = video1
        self._video2 = video2
        self._exp_name = exp1
        self._out_dir = out_dir

    @property
    def exp_name(self) -> str:
        return self._exp_name

    @property
    def videos(self) -> Tuple:
        return self._video1, self._video2

    @property
    def out_dir(self) -> str:
        return self._out_dir

    def sync(self) -> Tuple:
        audio1, audio2 = self._extract_audio()
        offset = self.calculate_sync_offset(audio1=audio1, audio2=audio2)
        video1_out, video2_out = self.trim_video(self._video1, self._video2, offset, self._out_dir)

        return video1_out, video2_out

    def _extract_audio(self) -> Tuple:
        min_sample_rate = min(self._video1.metadata.get("sample_rate", 44100),  # Default to 44100 Hz if key missing
                              self._video2.metadata.get("sample_rate", 44100))

        audios = []
        for video in [self._video1, self._video2]:
            camera_name = os.path.basename(os.path.dirname(video.path))
            out_path = os.path.join(self._out_dir, f'{camera_name}.{self.AUDIO_EXT}')
            audio = video.extract_audio(out_path=out_path, sample_rate=min_sample_rate)
            audios.append(audio)

        return audios

    @staticmethod
    def calculate_sync_offset(audio1, audio2) -> float:
        if audio1.rate != audio2.rate:
            raise ValueError(f"Sample rates must match, got {audio1.rate} and {audio2.rate}.")

        correlation = correlate(audio1.data, audio2.data, mode='full')
        lag = np.argmax(correlation) - (len(audio2.data) - 1)
        return lag / audio1.rate

    @staticmethod
    def trim_video(video1, video2, sync_time, out_path) -> Tuple:
        if video1.metadata["duration"] < video2.metadata["duration"]:
            video1, video2 = video2, video1

        fps = min(video1.metadata["fps"], video2.metadata["fps"])
        aligned_duration = min(video1.metadata["duration"] - sync_time, video2.metadata["duration"])
        out_video_paths = []

        for video, sync in zip([video1, video2], [sync_time, 0]):
            camera_name = os.path.basename(os.path.dirname(video.path))
            out_video_path = os.path.join(out_path, f'{camera_name}.{VideoSync.VIDEO_EXT}')
            out_video_paths.append(out_video_path)

            cmd = (
                ffmpeg
                .input(video.path, ss=sync)
                .filter('fps', fps=fps)
                .output(out_video_path, t=aligned_duration, vcodec='libx264', acodec='aac', format='mp4')
                .overwrite_output()
            )
            cmd.run(quiet=True)

        return out_video_paths


class ChessboardCameraCalibrator:
    def __init__(self, path, pattern_size=(9, 6), skip_frames=0):
        self.video_path = path
        self._chessboard_size = pattern_size
        self.skip_frames = skip_frames

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {path}")

        self.metadata = self._get_video_metadata()
        self.object_points = []
        self.image_points = []
        self.objp = np.zeros((self._chessboard_size[0] * self._chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self._chessboard_size[0], 0:self._chessboard_size[1]].T.reshape(-1, 2)

    @property
    def chessboard_size(self):
        return self._chessboard_size

    @chessboard_size.setter
    def chessboard_size(self, value):
        self._chessboard_size = value

    def calibrate(self, window_size=None):
        frame_idx = 0
        gray = None

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if frame_idx % (self.skip_frames + 1) == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, self._chessboard_size, None)

                if ret:
                    corners2 = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1),
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    )
                    self.object_points.append(self.objp)
                    self.image_points.append(corners2)
                    cv2.drawChessboardCorners(frame, self._chessboard_size, corners2, ret)

                if window_size:
                    h, w = frame.shape[:2]
                    scale = min(window_size[0] / w, window_size[1] / h)
                    frame = cv2.resize(frame, None, fx=scale, fy=scale)
                cv2.imshow('Calibration', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += 1

        cv2.destroyAllWindows()
        self.cap.release()

        if gray is None:
            raise ValueError("No frames were processed, or no chessboard patterns were detected.")

        if len(self.object_points) > 0:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                self.object_points, self.image_points, gray.shape[::-1], None, None
            )
            return CameraCalibration(mtx, dist, rvecs, tvecs)
        else:
            raise ValueError("No calibration patterns detected")

    def _get_video_metadata(self):
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
        return cls(data['camera_matrix'], data['dist_coeffs'], data.get('rvecs'), data.get('tvecs'))
