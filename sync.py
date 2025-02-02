import os
import cv2
from dotenv import load_dotenv
import numpy as np

load_dotenv()

DATA_DIR = r'C:\Users\ajayn\OneDrive\Documents\GitHub\Smartphone-based-RTI-Workflow\data'
OUT_DIR = r'C:\Users\ajayn\OneDrive\Documents\GitHub\Smartphone-based-RTI-Workflow\data\output'

CAMERA_1, CAMERA_1_EXT, CAMERA_1_WINSIZE = 'cam1static', 'mov', (324, 576)
CAMERA_2, CAMERA_2_EXT, CAMERA_2_WINSIZE = 'cam2_moving_light', 'mp4', (576, 324)

EXP_NAME = 'coin1'

INPUT_1 = os.path.join(DATA_DIR, CAMERA_1, f'{EXP_NAME}.{CAMERA_1_EXT}')
INPUT_2 = os.path.join(DATA_DIR, CAMERA_2, f'{EXP_NAME}.{CAMERA_2_EXT}')

SYNC_DIR = os.path.join(OUT_DIR, EXP_NAME, 'sync')


class VideoFile:
    def __init__(self, path: str):
        self.path = path
        self.capture = cv2.VideoCapture(path)

        if not self.capture.isOpened():
            raise FileNotFoundError(f"Error opening video file: {path}")

        self.frame_rate = self.capture.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_frame(self, frame_idx: int):
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.capture.read()
        return frame if ret else None

    def release(self):
        self.capture.release()

    def __len__(self):
        return self.frame_count


class VideoSync:
    def __init__(self, video1: VideoFile, video2: VideoFile):
        self.video1 = video1
        self.video2 = video2

    def _extract_frames(self, video, max_frames):
        frames = []
        for i in range(0, max_frames, 30):
            frame = video.get_frame(i)
            if frame is not None:
                frames.append(frame)
        return frames

    def _pad_frames(self, frames, target_frame_count):
        while len(frames) < target_frame_count:
            frames.append(np.zeros_like(frames[0]))
        return frames

    def sync(self):
        max_frames = min(self.video1.frame_count, self.video2.frame_count)
        frames1 = self._extract_frames(self.video1, max_frames)
        frames2 = self._extract_frames(self.video2, max_frames)

        max_length = max(len(frames1), len(frames2))
        frames1 = self._pad_frames(frames1, max_length)
        frames2 = self._pad_frames(frames2, max_length)

        return frames1, frames2


def save_synced_videos(frames1, frames2, output_video_path_static, output_video_path_dynamic, frame_rate):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    # Save static camera video
    if frames1:
        height, width = frames1[0].shape[:2]
        video_writer_static = cv2.VideoWriter(output_video_path_static, fourcc, frame_rate, (width, height))
        for frame in frames1:
            video_writer_static.write(frame)
        video_writer_static.release()
        print(f"Static camera video saved to {output_video_path_static}")

    # Save dynamic camera video
    if frames2:
        height, width = frames2[0].shape[:2]
        video_writer_dynamic = cv2.VideoWriter(output_video_path_dynamic, fourcc, frame_rate, (width, height))
        for frame in frames2:
            video_writer_dynamic.write(frame)
        video_writer_dynamic.release()
        print(f"Dynamic camera video saved to {output_video_path_dynamic}")


def main():
    try:
        os.makedirs(SYNC_DIR, exist_ok=True) 
        print(f"Saving synchronization data for experiment '{EXP_NAME}' to {SYNC_DIR}")

        print("Reading videos...")
        video1 = VideoFile(path=INPUT_1)
        video2 = VideoFile(path=INPUT_2)
        print(f"Video 1: {INPUT_1} - {video1.frame_count} frames")
        print(f"Video 2: {INPUT_2} - {video2.frame_count} frames")

        print("Synchronizing videos...")
        video_syncer = VideoSync(video1, video2)
        frames1, frames2 = video_syncer.sync()

        output_video_path_static = os.path.join(SYNC_DIR, f"{EXP_NAME}_static.mp4")
        output_video_path_dynamic = os.path.join(SYNC_DIR, f"{EXP_NAME}_dynamic.mp4")

        print("Saving synchronized videos...")
        save_synced_videos(frames1, frames2, output_video_path_static, output_video_path_dynamic, video1.frame_rate)

    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == '__main__':
    main()
