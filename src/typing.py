from typing import Dict, Tuple

import cv2

Size2D = Tuple[int, int]
''' 2D size representing width and height. It is used in different contexts, like kernel size, window size, etc. '''

Shape = Tuple[int, ...]
''' Shape of a tensor as sequence of integers. '''

RGBColor = Tuple[int, int, int] | Tuple[int, int, int, int]
''' RGB color as a tuple of 3 in the range [0, 255]. Optionally it can represent a RGBA color as a tuple of 4, where the last value is the alpha channel. '''

Frame = cv2.typing.MatLike
''' Frame as a 2D matrix of pixels. '''

Views = Dict[str, Frame]
''' Represents different possible frame views of a video stream, indexed by a string key. '''

LightDirection = Tuple[float, float]

Pixel = Tuple[int, int]