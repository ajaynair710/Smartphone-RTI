from __future__ import annotations

from abc import abstractmethod
import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, Set, Tuple, Type, Any
import cv2
import ffmpeg
from loguru import logger as loguru_logger
from numpy.typing import NDArray
from scipy.io.wavfile import read as wav_read

from utils.misc import Timer
from src.typing import Size2D


# _______________________________ LOGGER _______________________________

class BaseLogger(logging.Logger):
    ''' 
    Abstract class for loggers with a formatting function for info, warning and error channels.
    The formatting is implemented as a function that transform an input string into an output string.
    '''

    def __init__(self, name: str):
        ''' Initialize the logger with a name. '''

        super().__init__(name=name)

        # The default formatter is the identity function
        self._formatter: Callable[[str], str] = lambda x: x

    # --- FORMATTER ---

    @property
    def formatter(self) -> Callable[[str], str]: return self._formatter

    @formatter.setter
    def formatter(self, prefix: Callable[[str], str]): self._formatter = prefix

    def reset_formatter(self): self._formatter = lambda x: x

    # --- LOGGING CHANNELS ---

    ''' The pubic methods are predefined with the formatter function. '''
    def info   (self, msg): [self._info   (self.formatter(msg_)) for msg_ in msg.split("\n")]
    def warning(self, msg): [self._warning(self.formatter(msg_)) for msg_ in msg.split("\n")]
    def error  (self, msg): [self._error  (self.formatter(msg_)) for msg_ in msg.split("\n")]

    ''' The private methods are abstract and must be implemented by the subclasses. '''
    @abstractmethod
    def _info   (self, msg): raise NotImplementedError

    @abstractmethod
    def _warning(self, msg): raise NotImplementedError

    @abstractmethod
    def _error  (self, msg): raise NotImplementedError

    def handle_error(self, msg: str, exception: Type[Exception] = ValueError):
        ''' Log the error message and raise an exception. '''

        self.error(msg)
        raise exception(msg)

class SilentLogger(BaseLogger):
    ''' Logger implemented as a no-op. '''

    def __init__(self): super().__init__(name='SilentLogger')

    def _info   (self, msg): pass
    def _warning(self, msg): pass
    def _error  (self, msg): pass


class PrintLogger(BaseLogger):
    ''' Logger that prints messages to the console. '''

    def __init__(self): super().__init__(name='PrintLogger')

    def _info   (self, msg): print(f"INFO:  {msg}")
    def _warning(self, msg): print(f"WARN:  {msg}")
    def _error  (self, msg): print(f"ERROR: {msg}")


class FileLogger(BaseLogger):
    ''' Logger that writes messages to a .log file. '''

    def __init__(self, file, level=logging.INFO):

        super().__init__(name='FileLogger')

        InputSanitizationUtils.check_extension(path=file, ext='.log')

        loguru_logger.add(file, level=level)
        self._logger = loguru_logger

    def _info   (self, msg): self._logger.info   (msg)
    def _warning(self, msg): self._logger.warning(msg)
    def _error  (self, msg): self._logger.error  (msg)


# _______________________________ CONTAINER UTILS CLASSES _______________________________

class PathUtils:

    @staticmethod
    def get_folder_path(path: str) -> str: return os.path.dirname(path)
    ''' Get the folder path of the file (e.g. /path/to/file.txt -> /path/to). '''

    @staticmethod
    def get_folder_name(path: str) -> str: return os.path.basename(PathUtils.get_folder_path(path))
    ''' Get the name of file containing folder (e.g. /path/to/file.txt -> to). '''

    @staticmethod
    def get_file(path: str) -> str: return os.path.basename(path)
    ''' Get the file from the path (e.g. /path/to/file.txt -> file.txt). '''

    @staticmethod
    def get_folder_and_file(path: str) -> str: return os.path.join(PathUtils.get_folder_name(path), PathUtils.get_file(path))
    ''' Get the folder and file from the path (e.g. /path/to/file.txt -> /to/file.txt). '''

    @staticmethod
    def get_file_name(path: str) -> str:
        ''' Get the file name from the path (e.g. /path/to/file.txt -> file). '''

        file, ext = os.path.splitext(PathUtils.get_file(path))
        return file
    
    @staticmethod
    def get_file_ext(path: str) -> str:
        ''' Get the file extension from the path (e.g. /path/to/file.txt -> txt). '''

        file, ext = os.path.splitext(PathUtils.get_file(path))
        return ext[1:].lower() # Remove the dot
    
class IOUtils:

    @staticmethod
    def make_dir(path: str, logger: BaseLogger = SilentLogger()):
        ''' Create a directory if it does not exist. '''

        if not os.path.exists(path): 
            os.makedirs(path)
            logger.info(msg=f"Directory created at: {path}")
        else:
            logger.info(msg=f"Directory already found at: {path}")
    
    @staticmethod
    def error_handler(msg: str, logger: BaseLogger, exception: Type[Exception] = ValueError):
        ''' Log the error message and raise an exception. '''

        logger.error(msg)
        raise exception(msg)


class InputSanitizationUtils:

    @staticmethod
    def check_input(path: str, logger: BaseLogger = SilentLogger()):
        ''' Check if input file exists. '''

        if not os.path.exists(path):  
            logger.handle_error(
                msg=f"Input file not found: {path}", 
                exception=FileNotFoundError
            )
        else:
            logger.info(msg=f"Input file found at: {path} ")

    @staticmethod
    def check_output(path: str, logger: BaseLogger = SilentLogger()):
        ''' Check if the directory of the output file exists. '''

        out_dir = PathUtils.get_folder_path(path)

        if not os.path.exists(out_dir): 
            logger.handle_error(
                msg=f"Output directory not found: {out_dir}", 
                exception=FileNotFoundError
            )
        else:                           

            logger.info(msg=f"Output directory found at: {out_dir} ")

    @staticmethod
    def check_extension(path: str, ext: str | Set[str], logger: BaseLogger = SilentLogger()):
        ''' Check if any of the extensions in the list match the file extension. '''

        if type(ext) == str: ext = {ext}

        if not any([path.endswith(e) for e in f'.{ext}']):

            logger.handle_error(
                msg=f"Invalid file extension: {path}. Expected one of {ext} extensions.", 
                exception=ValueError
            )


# _______________________________ AUDIO & VIDEO FILES _______________________________

class AudioFile:
    ''' Class to handle audio files. '''

    EXT_AUDIO = {'wav', 'mp3', 'flac', 'ogg', 'm4a', 'wma', 'aac', 'aiff', 'au', 'raw'}

    # --- INITIALIZATION ---

    def __init__(self, path: str, logger: BaseLogger = SilentLogger(), verbose: bool = False):
        ''' Initialize the audio file with its path '''

        self._logger        : BaseLogger = logger
        self._logger_verbose: BaseLogger = logger if verbose else SilentLogger()
        self._is_verbose    : bool = verbose

        self._path: str = path

        self._rate: int
        self._data: NDArray
        self._rate, self._data = self._read_audio()

    def _read_audio(self) -> Tuple[int, NDArray]:
        ''' Read the audio file and return the sample rate and data. '''

        InputSanitizationUtils.check_input    (path=self.path, logger=self._logger_verbose)
        InputSanitizationUtils.check_extension(path=self.path, logger=self._logger_verbose, ext=self.EXT_AUDIO)

        self._logger.info(msg=f"Reading audio file from {self.path} ...")

        timer = Timer()

        try: 
            rate, data = wav_read(filename=self.path)
            self._logger.info(msg=f"Audio read successfully in {timer}.")
        except Exception as e:
            self._logger.error(f"Failed to extract audio: {e}")
            raise e

        # Convert to mono
        if data.ndim > 1: data = data.mean(axis=1)

        return rate, data
    
    # --- MAGIC METHODS ---

    def __str__(self) -> str:

        return f"{self.__class__.__name__}[{self.name}; "\
            f"duration: {int  (self.duration)} sec; "\
            f"sampling rate: {round(self.rate, 2 )} Hz]"

    def __repr__(self) -> str: return str(self)

    def __len__(self) -> int: return len(self.data)
    ''' Number of samples in the audio file. '''

    # --- PROPERTIES ---

    @property
    def path(self) -> str: return self._path

    @property
    def name(self) -> str: return PathUtils.get_folder_and_file(path=self.path)

    @property
    def rate(self) -> int: return self._rate
    ''' Sampling rate of the audio file in Hz. '''

    @property
    def data(self) -> NDArray: return self._data
    ''' Data of the audio file as a one dimensional numpy array. '''

    @property
    def duration(self) -> float: return len(self.data) / self.rate
    ''' Duration of the audio file in seconds. '''


@dataclass

class VideoFile:
    class VideoMetadata:
        def __init__(self, fps: float, width: int, height: int, num_frames: int):
            self.fps = fps
            self.width = width
            self.height = height
            self.num_frames = num_frames

        @property
        def size(self) -> Size2D:
            return (self.width, self.height)
            
        @property
        def frames(self) -> int:
            return self.num_frames


        @staticmethod
        def from_video_path(path: str, logger: Any = None, verbose: bool = False) -> VideoFile.VideoMetadata:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise IOError(f"Error opening video file {path}")
            
            # Get basic video metadata
            fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second (FPS)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Number of frames in the video
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height

            # Return an instance of VideoMetadata with keyword arguments
            return VideoFile.VideoMetadata(fps=fps, width=width, height=height, num_frames=num_frames)


        @staticmethod
        def from_video_path(path: str, logger: BaseLogger, verbose: bool = False):
            # Open the video file with OpenCV
            cap = cv2.VideoCapture(path)
            
            if not cap.isOpened():
                logger.error(f"Failed to open video file: {path}")
                raise FileNotFoundError(f"Video file not found: {path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            return VideoFile.VideoMetadata(fps=fps, width=width, height=height, num_frames=num_frames)


    EXT_VIDEO = {'mov', 'mp4', 'avi', 'flv', 'mkv', 'webm'}

    # --- INITIALIZATION ---

    def __init__(self, path: str, logger: BaseLogger = SilentLogger(), verbose: bool = False):
        self._logger = logger
        self._logger_verbose = logger if verbose else SilentLogger()
        self._is_verbose = verbose
        self._path = path
        self._metadata = self._read_video()

    @property
    def metadata(self) -> VideoMetadata:
        return self._metadata
        
    def _read_video(self) -> VideoMetadata:
        InputSanitizationUtils.check_input(path=self.path, logger=self._logger_verbose)
        InputSanitizationUtils.check_extension(path=self.path, ext=self.EXT_VIDEO, logger=self._logger_verbose)

        self._logger.info(msg=f"Reading video file from {self.path} ...")

        try:
            cap = cv2.VideoCapture(self.path)
            if not cap.isOpened():
                raise IOError(f"Failed to open video: {self.path}")
                
            metadata = self.VideoMetadata(
                fps=cap.get(cv2.CAP_PROP_FPS),
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                num_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            )
            cap.release()
            return metadata
            
        except Exception as e:
            self._logger.error(f"Failed to read video metadata: {e}")
            raise 
    # --- MAGIC METHODS ---
    
    def __str__(self) -> str:

        return f"{self.__class__.__name__}[{self.name}; "\
            f"duration: {round(self.duration, 2)} sec; "\
            f"fps: {round(self.fps, 2)} fps]"

    def __repr__(self) -> str: return str(self)

    def __len__(self) -> int: return self.metadata.num_frames

    # --- PROPERTIES ---

    @property
    def path(self) -> str: return self._path

    @property
    def name(self) -> str: return PathUtils.get_folder_and_file(path=self.path)

    @property
    def fps(self) -> float: return self.metadata.fps
    ''' Frames per second of the video file. '''

    @property
    def width(self) -> int: return self.metadata.width
    ''' Width of the video in pixels. '''

    @property
    def height(self) -> int: return self.metadata.height
    ''' Height of the video in pixels. '''

    @property
    def num_frames(self) -> int: return self.metadata.num_frames
    ''' Number of frames in the video. '''

    @property
    def duration(self) -> float: return self.num_frames / self.fps
    ''' Duration of the video in seconds. '''

    @property
    def metadata(self) -> VideoMetadata: return self._metadata

    @property
    def has_audio(self) -> bool: return self.metadata.has_audio
    ''' Check if the video file has an audio stream. '''

    AUDIO_EXTRACTION_EXT = 'wav'

    @property
    def audio_path(self) -> str: 
        ''' 
        Path to the extracted audio file.
        That is the same as the video file with audio extension.
        '''
        return os.path.join(PathUtils.get_folder_path(self.path), f'{PathUtils.get_file_name(self.path)}.{self.AUDIO_EXTRACTION_EXT}')

    # --- AUDIO EXTRACTION ---

    '''
    NOTE:   When working on audio file, we extract the audio track from the video file in the same directory.
            The audio file is deleted when the video file object is deleted.
    '''

    def extract_audio_track(self) -> AudioFile:
        ''' Extract the audio track from the video file and save it to the same folder with the video file. '''

        if not self.has_audio:
            self._logger.handle_error(f"Cannot extract audio stream in video {self.path}", exception=ValueError)

        audio = self.extract_audio(video=self, out_path=self.audio_path, logger=self._logger, verbose=self._is_verbose)
        self._logger.info(msg='Use method `close_audio` to remove the audio file from the file system.')
        return audio

    @staticmethod
    def extract_audio(
        video: 'VideoFile',
        out_path: str,
        sample_rate: int | None = None,
        logger: BaseLogger = SilentLogger(),
        verbose: bool = False
    ) -> AudioFile:
        '''
        Extract the audio track from the video file and save it to the output path.
        It optionally resamples the audio to the given sample rate.
        '''

        verbose_logger = logger if verbose else SilentLogger()

        InputSanitizationUtils.check_output(path=out_path, logger=verbose_logger)

        logger.info(msg=f"Extracting audio from {video.path} to {out_path} ...")

        try:

            timer = Timer()

            # Add sample rate argument if provided
            sample_rate_args = {'ar': sample_rate} if sample_rate else {}

            cmd = (
                ffmpeg
                .input(video.path)
                .output(out_path, **sample_rate_args)
                .overwrite_output()
            )
            
            cmd.run(overwrite_output=True, quiet=True)

            logger.info(msg=f"Audio extracted successfully in {timer}.")
            return AudioFile(path=out_path, logger=logger)

        except Exception as e:
            logger.error(f"Failed to extract audio: {e}")
            raise e

    def close_audio(self):
        '''
        Close the audio file by removing it from the file system.
        '''
        try:
            if os.path.exists(self.audio_path):
                self._logger.info(msg=f"Attempting to remove audio file {self.audio_path}...")
                os.remove(self.audio_path)
                self._logger.info(msg=f"Audio file {self.audio_path} removed successfully.")
            else:
                self._logger.warning(msg=f"Audio file not found at {self.audio_path}. It may have already been removed.")
        
        except OSError as e:
            self._logger.error(f"Failed to remove audio file {self.audio_path}: {e}")
            raise e
        except Exception as e:
            self._logger.error(f"An unexpected error occurred while removing audio file {self.audio_path}: {e}")
            raise e


    def __del__(self): 
        ''' Remove the audio if it was extracted when the video file object is deleted '''
        self._logger = SilentLogger() # Prevent logging
        self.close_audio()
