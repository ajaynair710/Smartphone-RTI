import os
import cv2
from dotenv import load_dotenv

import numpy as np
from src.stream import SynchronizedVideoStream, Stream  # Assuming Stream is imported from src.stream
from utils.io import IOUtils, FileLogger
from utils.misc import Timer

load_dotenv()

DATA_DIR = os.getenv('DATA_DIR', '.')
OUT_DIR  = os.getenv('OUT_DIR',  '.')

CAMERA_1, CAMERA_1_EXT, CAMERA_1_WINSIZE = 'cam1static', 'mov', (324, 576)
CAMERA_2, CAMERA_2_EXT, CAMERA_2_WINSIZE = 'cam2_moving_light', 'mp4', (576, 324)

EXP_NAME     = 'coin1'

INPUT_1  = os.path.join(DATA_DIR, CAMERA_1, f'{EXP_NAME}.{CAMERA_1_EXT}')
INPUT_2  = os.path.join(DATA_DIR, CAMERA_2, f'{EXP_NAME}.{CAMERA_2_EXT}')

SYNC_DIR = os.path.join(OUT_DIR, EXP_NAME, 'sync')


class VideoFile:
    def __init__(self, path: str, logger: FileLogger):
        self.path = path
        self.logger = logger
        self.capture = cv2.VideoCapture(path)

        if not self.capture.isOpened():
            raise FileNotFoundError(f"Error opening video file: {path}")
        
        # Video metadata (frame rate, total frames, resolution)
        self.frame_rate = self.capture.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_frame(self, frame_idx: int):
        """Retrieve a frame by index."""
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.capture.read()
        return frame if ret else None

    def release(self):
        """Release video capture resources."""
        self.capture.release()
    
    def __len__(self):
        return self.frame_count


class ConcreteStream(Stream):
    def __init__(self, name):
        super().__init__(name)
        self.frames = []

    def add_frame(self, frame):
        """Method to add a frame to the stream."""
        self.frames.append(frame)

    def __getitem__(self, idx):
        """Retrieve a frame by index."""
        return self.frames[idx]

    def __len__(self):
        """Return the number of frames."""
        return len(self.frames)

    def _default_window_size(self):
        """Return the default window size (can be customized)."""
        return (640, 480)

    def iter_range(self, start, end, step):
        """Create an iterator for frames in a specific range."""
        for idx in range(start, end, step):
            yield idx, self.frames[idx]


class VideoSync:
    def __init__(self, video1: VideoFile, video2: VideoFile, out_dir: str):
        self.video1 = video1
        self.video2 = video2
        self.out_dir = out_dir
        
    def _extract_frames(self, video, max_frames):
        frames = []
        for i in range(0, max_frames, 30):  # Sample every 30 frames
            frame = video.get_frame(i)
            if frame is not None:
                frames.append(frame)
        return frames
        
    def _pad_frames(self, frames, target_frame_count):
        # Padding with black frames (zeros) if the frame count is less than target_frame_count
        while len(frames) < target_frame_count:
            frames.append(np.zeros_like(frames[0]))  # Add black frame as padding
        return frames

    def sync(self):
        # Ensure both videos have the same number of frames
        max_frames = min(self.video1.frame_count, self.video2.frame_count)  # Take the minimum of both
        frames1 = self._extract_frames(self.video1, max_frames)
        frames2 = self._extract_frames(self.video2, max_frames)

        # Ensure both frames lists are the same length
        max_length = max(len(frames1), len(frames2))
        frames1 = self._pad_frames(frames1, max_length)
        frames2 = self._pad_frames(frames2, max_length)

        return frames1, frames2


def create_stream_from_frames(frames, name):
    """Helper function to create ConcreteStream object from frames."""
    stream = ConcreteStream(name=name)
    for i, frame in enumerate(frames):
        stream.add_frame(frame)
    return stream


def play_and_save_streams(sync_stream, window_sizes, logger, output_video_path_static, output_video_path_dynamic, video1, video2):
    """Function to play and save synchronized video streams separately for static and moving cameras."""
    # Define the codec and create VideoWriter objects
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # You can use other codecs like 'XVID', 'MJPG', etc.
    video_writer_static = None  # For static camera
    video_writer_dynamic = None  # For moving camera

    # Separate streams for static and moving cameras
    static_stream = next(stream for stream in sync_stream.streams if stream.name == CAMERA_1)
    dynamic_stream = next(stream for stream in sync_stream.streams if stream.name == CAMERA_2)

    # Process static camera stream
    for idx, frame in static_stream.iter_range(start=0, end=len(static_stream), step=1):
        # Initialize the VideoWriter for static camera once we get the first frame
        if video_writer_static is None:
            height, width = frame.shape[:2]
            video_writer_static = cv2.VideoWriter(output_video_path_static, fourcc, video1.frame_rate, (width, height))

        # Write the frame to the static camera video file
        video_writer_static.write(frame)

        key = cv2.waitKey(1)  # Wait for a key press
        if key == ord('q'):  # Press 'q' to quit playback
            break

    # Process dynamic (moving light) camera stream
    for idx, frame in dynamic_stream.iter_range(start=0, end=len(dynamic_stream), step=1):
        # Initialize the VideoWriter for moving camera once we get the first frame
        if video_writer_dynamic is None:
            height, width = frame.shape[:2]
            video_writer_dynamic = cv2.VideoWriter(output_video_path_dynamic, fourcc, video2.frame_rate, (width, height))

        # Write the frame to the moving camera video file
        video_writer_dynamic.write(frame)

        key = cv2.waitKey(1)  # Wait for a key press
        if key == ord('q'):  # Press 'q' to quit playback
            break

    # Release the VideoWriter when done
    if video_writer_static:
        video_writer_static.release()
        logger.info(f"Static camera video saved to {output_video_path_static}")

    if video_writer_dynamic:
        video_writer_dynamic.release()
        logger.info(f"Dynamic camera video saved to {output_video_path_dynamic}")

    # Close the stream window after finishing
    cv2.destroyAllWindows()


def main():
    try:
        # Output directory
        IOUtils.make_dir(path=SYNC_DIR)
        logger = FileLogger(file=os.path.join(SYNC_DIR, 'sync.log'))
        logger.info(msg=f'Saving synchronization data for experiment {EXP_NAME} to {SYNC_DIR}.')
        
        # Read videos
        logger.info(msg=f'READING VIDEOS')
        video1 = VideoFile(path=INPUT_1, logger=logger)
        video2 = VideoFile(path=INPUT_2, logger=logger)
        logger.info(msg=f'Video 1: {INPUT_1} - {video1.frame_count} frames')
        logger.info(msg=f'Video 2: {INPUT_2} - {video2.frame_count} frames')
        
        # Synchronization
        logger.info(msg=f"CREATING SYNCHRONIZATION OBJECT")
        video_syncer = VideoSync(
            video1=video1,
            video2=video2,
            out_dir=SYNC_DIR
        )
        logger.info(msg=str(video_syncer))
        
        frames1, frames2 = video_syncer.sync()

        # Create streams for the videos
        stream1 = create_stream_from_frames(frames1, name=CAMERA_1)
        stream2 = create_stream_from_frames(frames2, name=CAMERA_2)

        # Create a synchronized video stream
        sync_stream = SynchronizedVideoStream(
            streams=[stream1, stream2],
            logger=logger,
        )

        # Define the output paths for the synced videos (separate static and moving)
        output_video_path_static = os.path.join(SYNC_DIR, f'{EXP_NAME}_static.mp4')
        output_video_path_dynamic = os.path.join(SYNC_DIR, f'{EXP_NAME}_dynamic.mp4')
        logger.info(msg=f'Saving static camera video to {output_video_path_static}')
        logger.info(msg=f'Saving dynamic camera video to {output_video_path_dynamic}')

        # Playing and saving synced videos separately for static and moving camera
        logger.info(msg=f'PLAYING AND SAVING SYNCED VIDEOS SEPARATELY')
        timer = Timer()

        play_and_save_streams(
            sync_stream,
            window_sizes={CAMERA_1: CAMERA_1_WINSIZE, CAMERA_2: CAMERA_2_WINSIZE},
            logger=logger,
            output_video_path_static=output_video_path_static,
            output_video_path_dynamic=output_video_path_dynamic,
            video1=video1,
            video2=video2
        )
        
        logger.info(msg=f'Streaming completed in {timer}.')
        
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        print(f"Error occurred: {e}")


if __name__ == '__main__':
    main()
