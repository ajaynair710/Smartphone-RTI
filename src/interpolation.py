from functools import partial
import itertools
import pickle
import os
from typing import Tuple, List, Callable, Type
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from matplotlib.figure import Figure
from scipy.interpolate import Rbf

from src.typing import Shape, Frame
from src.mlic import MLIC
from utils.misc import default
from utils.io import BaseLogger, PathUtils, SilentLogger, InputSanitizationUtils as ISUtils, Timer

class BasisInterpolation:
    
	def __init__(
		self, 
		basis: NDArray,
		min_max_coords : Tuple[Tuple[float, float], ...],
		min_max_values : Tuple[float, float]
	):
		
		self._basis : NDArray = basis


		# Check if basis is withing mix and max values
		# min_val, max_val = min_max_values
		# if not  np.all((min_val <= self._basis) & (self._basis <= max_val)):
		#	raise ValueError(f'The minimum value of the basis is not within the minimum and maximum values of the basis. ')
		
		self._min_max_values: Tuple[float, float] = min_max_values
		
		# if not (self.n_dims == len(min_max_coors)):
		# 	raise ValueError(f'The number of basis dimensions ({self.n_dims}) don\'t match the length of min and max values {len(min_max_coors)}. ')
		
		self._min_max_coords: Tuple[Tuple[float, float], ...] = min_max_coords

	@property
	def basis(self) -> NDArray: return self._basis

	def __str__(self): return f'{self.__class__.__name__}[shape: {"x".join([str(i) for i in self.shape])}]'
	def __repr__(self): return str(self)

	def _map_coordinate       (self, coord: Tuple[float, ...]) -> Tuple[float, ...]: return self.map_coordinate       (coord=coord, min_max_coords=self.min_max_coords, shape=self.shape)
	def _discretize_coordinate(self, coord: Tuple[float, ...]) -> Tuple[int,   ...]: return self.discretize_coordinate(coord=coord, min_max_coords=self.min_max_coords, shape=self.shape)

	def __getitem__(self, coord: Tuple[float, ...]) -> float:

		idx = self.discretize_coordinate(coord=coord, min_max_coords=self.min_max_coords, shape=self.shape)

		return self._basis[idx]
	
	@property
	def shape(self): return self._basis.shape

	@property
	def n_dims(self): return len(self.shape)

	@property
	def min_max_coords(self): return self._min_max_coords

	@property
	def min_max_values(self): return self._min_max_values

	@staticmethod
	def map_coordinate(
		coord          : Tuple[float, ...],
		min_max_coords : Tuple[Tuple[float, float], ...],
		shape          : Shape
	) -> Tuple[float, ...]:

		out_idx = []

		for i, (c, (min_val, max_val), dim) in enumerate(zip(coord, min_max_coords, shape)):

			if not (min_val <= c <= max_val): raise ValueError(f'The {i+1} coordinate {c} is not within the minimum and maximum values of the basis [{min_val}, {max_val}]. ')
			
			step = (max_val - min_val) / dim
			idx = (c - min_val) / step
			out_idx.append(idx)
		
		# out_idx = out_idx[::-1] # Reverse the order to match the numpy indexing
		
		return tuple(out_idx)
	
	@staticmethod
	def discretize_coordinate(
		coord		  : Tuple[float, ...],
		min_max_coords: Tuple[Tuple[float, float], ...],
		shape         : Shape
	) -> Tuple[int, ...]:
		
		mapped_coord = BasisInterpolation.map_coordinate(
			coord=coord, 
			min_max_coords=min_max_coords, 
			shape=shape
		)

		out_idx = []
		for c, dim in zip(mapped_coord, shape):
			idx = int(np.clip(c, 0, dim-1))
			out_idx.append(idx)
			
		return tuple(out_idx)
	
	def plot_interpolation(
		self,
		title            : str = 'Basis Interpolation',
		min_max_colorbar : bool = False,
		points_coord     : Tuple[NDArray, NDArray] | None = None
	) -> Figure:
		
		if self.n_dims != 2: raise ValueError(f'The basis must be 2D to plot. ')

		fig, ax = plt.subplots(figsize=(10, 6))

		ax.set_title(title, fontsize=14)
		ax.set_xlabel('U Light')
		ax.set_ylabel('V Light')

		(min_x, max_x), (min_y, max_y) = self.min_max_coords

		STEP = 5
		dim_x, dim_y = self.shape
		ax.set_xlim(0, dim_x-1); ax.set_xticks(np.linspace(0, dim_x-1, STEP)); ax.set_xticklabels(np.linspace(min_x, max_x, STEP))
		ax.set_ylim(0, dim_y-1); ax.set_yticks(np.linspace(0, dim_y-1, STEP)); ax.set_yticklabels(np.linspace(min_y, max_y, STEP))

		if min_max_colorbar: vmin, vmax = self.min_max_values
		else:                vmin, vmax = self.basis.min(), self.basis.max()

		if points_coord is not None:
			
			values, coords = points_coord

			# Move the coordinates to the pixel space
			coords_ = np.array([self._map_coordinate(c) for c in coords])

			if not min_max_colorbar:
				vmin = min(vmin, values.min())
				vmax = max(vmax, values.max())
			
			ax.scatter(
				*coords_.T,        # Coordinates for scatter points
				c=values,       # Values for coloring
				cmap='viridis', # Use the same colormap
				edgecolor='k',  # Optional: edge for better visibility
				s=25,           # Size of scatter points
				linewidths=0.5,  # Edge width for scatter points
				vmin=vmin, 
				vmax=vmax,
			)
		
		img = ax.imshow(
			self.basis, 
			cmap='viridis', 
			vmin=vmin, 
			vmax=vmax,
		)

		fig.colorbar(img, ax=ax, label='Pixel Luminance')
		
		return fig
	
class BasisInterpolationCollection:
    
	def __init__(self, basis_interpolations: List[BasisInterpolation], out_shape: Shape | None = None):

		# Check all have the same min and max coordinates
		self._min_max_coords = basis_interpolations[0].min_max_coords
		if not all(bi.min_max_coords == self._min_max_coords for bi in basis_interpolations):
			raise ValueError('All basis interpolations must have the same minimum and maximum coordinates. ')
		
		# Check all have the same min and max values
		self._min_max_values = basis_interpolations[0].min_max_values
		if not all(bi.min_max_values == self._min_max_values for bi in basis_interpolations):
			raise ValueError('All basis interpolations must have the same minimum and maximum values. ')
		
		# Stack all basis interpolations
		# NOTE Raises an error if the shapes are not the same
		self._basis_interpolations: NDArray = np.stack([bi.basis for bi in basis_interpolations])

		self._out_shape: Shape | None = out_shape

	def __str__ (self): return f'{self.__class__.__name__}['\
		f'basis: {len(self)}; '\
		f'basis shape: {"x".join([str(i) for i in self.shape])}'\
		f'{f"; out shape: {self._out_shape[0]}x{self._out_shape[1]}" if self._out_shape else ""}]'
	
	def __repr__(self): return str(self)

	def __len__(self): return self._basis_interpolations.shape[0]

	def __getitem__(self, index: int) -> BasisInterpolation: 
		
		return BasisInterpolation(basis=self._basis_interpolations[index], min_max_coords=self._min_max_coords, min_max_values=self._min_max_values)

	def get_interpolation_frame(self, coord: Tuple[float, float]) -> Frame:
		
		idx = BasisInterpolation.discretize_coordinate(coord=coord, min_max_coords=self._min_max_coords, shape=self.shape)

		bi = self._basis_interpolations[:, *idx[::-1]].astype('uint8')

		if self._out_shape is not None: return bi.reshape(self._out_shape)
	
		return bi
	
	@property
	def shape(self) -> Shape:
		n_object, *shape = self._basis_interpolations.shape
		return tuple(shape)

	@classmethod
	def from_pickle(cls, path: str, logger: BaseLogger = SilentLogger()) -> 'BasisInterpolationCollection':
		''' Load camera calibration from a pickle file. '''

		logger.info(msg=f"Loading camera calibration from {path}")
		with open(path, 'rb') as f: return pickle.load(f)

	def dump(
        self,
        path   : str,
        logger : BaseLogger = SilentLogger(),
        verbose: bool       = False
    ) -> None:
		''' Save the camera calibration to a pickle file. '''

		logger_verbose = logger if verbose else SilentLogger()

		ISUtils.check_output(path=PathUtils.get_folder_path(path=path), logger=logger_verbose)

		logger.info(msg=f"Saving basis interpolation collection to {path} ...")

		timer = Timer()
		with open(path, 'wb') as f: pickle.dump(self, f)
		logger.info(msg=f"Completed in {timer}")

class BasisInterpolator(ABC):
    
	def __init__(
		self, 
		coords             : NDArray,
		interpolation_size : Tuple[int, ...],
		range_coordinates  : Tuple[Tuple[float, float], ...],
		range_values       : Tuple[float, float]
	):
	
		self._len, self._dims = coords.shape

		if self._dims != len(range_coordinates) != len(interpolation_size): 
			raise ValueError(
				f'The number of coordinates dimension ({self._dims}), coordinates ranges ({len(range_coordinates)}) '
				f'and interpolation size ({len(interpolation_size)}) must be the same. '
			)

		self._coords             : NDArray                         = coords
		self._interpolation_size : Tuple[int, ...]                 = interpolation_size
		self._range_coords       : Tuple[Tuple[float, float], ...] = range_coordinates
		self._range_values       : Tuple[float, float]             = range_values

		# Precompute interpolation grid
		self._interpolation_grid = np.meshgrid(*[
			np.linspace(min_val, max_val, dim) 
			for (min_val, max_val), dim in zip(self._range_coords, self._interpolation_size)
		])
	
	def __str__(self) -> str : return f'{self.__class__.__name__}['\
		f'points: {len(self)}; '\
		f'dimensions: {self.dims}; '\
		f'interpolation size: {"x".join([str(i) for i in self._interpolation_size])}]'
	
	def __repr__(self): return str(self)

	def __len__(self) -> int: return self._len

	@property
	def dims(self) -> int: return self._dims

	@abstractmethod
	def _fit_interpolation_function_on_values(self, values: NDArray) -> Callable[[NDArray], NDArray]: pass

	def interpolate(self, values: NDArray) -> BasisInterpolation:

		# Get interpolation function
		interpolation_function = self._fit_interpolation_function_on_values(values=values)

		# Interpolate
		basis = interpolation_function(*self._interpolation_grid)

		basis = np.clip(basis, *self._range_values)

		return BasisInterpolation(
			basis=basis, 
			min_max_coords=self._range_coords,
			min_max_values=self._range_values
		)

class RTIBasisInterpolator(BasisInterpolator):

	COORD_RANGE  = (-1,   1)
	VALUES_RANGE = ( 0, 255)

	def __init__(
		self,
		coordinates : NDArray,
		interpolation_size: Tuple[int, int]
	):
		super().__init__(
			coords=coordinates,
			interpolation_size=interpolation_size,
			range_coordinates=(self.COORD_RANGE, self.COORD_RANGE),
			range_values=self.VALUES_RANGE
		)

class RTIRadialBasisInterpolator(RTIBasisInterpolator):
	
	def __init__(self, coordinates: NDArray, interpolation_size: Tuple[int, int]):
		super().__init__(coordinates=coordinates, interpolation_size=interpolation_size)

	def _fit_interpolation_function_on_values(self, values: NDArray):

		return Rbf(*self._coords.T, values, function='linear', smooth=0.1)

class RTIPolynomialTextureMapInterpolator(RTIBasisInterpolator):

	def __init__(self, coordinates: NDArray, interpolation_size: Tuple[int, int]):
		super().__init__(coordinates=coordinates, interpolation_size=interpolation_size)

	def _fit_interpolation_function_on_values(self, values: NDArray):

		# Extract light coordinates (Lx, Ly)
		Lx, Ly = self._coords.T

		# Construct the matrix A for polynomial regression
		A = np.column_stack([
			np.ones_like(Lx),  # a0
			Lx,                # a1
			Ly,                # a2
			Lx**2,             # a3
			Ly**2,             # a4
			Lx * Ly            # a5
		])

		# Solve for the coefficients
		coefficients, _, _, _ = np.linalg.lstsq(A, values, rcond=None)

		# Return an interpolation function
		def interpolation_function(Lx_new: NDArray, Ly_new: NDArray, coefficients: NDArray) -> NDArray:

			return (
				coefficients[0] +
				coefficients[1] * Lx_new +
				coefficients[2] * Ly_new +
				coefficients[3] * Lx_new**2 +
				coefficients[4] * Ly_new**2 +
				coefficients[5] * Lx_new * Ly_new
			)
        
		return partial(interpolation_function, coefficients=coefficients)

class MLICBasisInterpolator:
    
	def __init__(
		self,
		mlic: MLIC,
		C_rti_interpolator: Type[RTIBasisInterpolator],
		interpolation_size: Tuple[int, int],
		logger: BaseLogger = SilentLogger(),
		verbose: bool = False
	):
		
		self._logger         : BaseLogger = logger
		self._logger_verbose : BaseLogger = logger if verbose else SilentLogger()
		self._is_verbose     : bool       = verbose

		self._mlic               : MLIC                 = mlic
		self._interpolation_size : Tuple[int, int]      = interpolation_size
		self._rti_interpolator   : RTIBasisInterpolator = C_rti_interpolator(coordinates=self._mlic.light_directions, interpolation_size=self._interpolation_size)
	
	def __str__(self) -> str:
		
		sx, sy = self._interpolation_size
		mx, my = self._mlic.size
		
		return f'{self.__class__.__name__}['\
            f'interpolation size: {sx}x{sy}; '\
            f'MLIC objects: {len(self._mlic)}; '\
			f'MLIC size: {mx}x{my}; '\
            f'interpolator: {self._rti_interpolator.__class__.__name__}]'
	
	def __repr__(self) -> str: return str(self)

	def get_pixel_interpolation(self, pixel: Tuple[int, int]) -> BasisInterpolation:

		return self._rti_interpolator.interpolate(values=self._mlic.get_pixel_values(pixel=pixel))
	
	def get_interpolation_collection(self, progress: int | None = None, save_dir: str = '') -> BasisInterpolationCollection:

		rows, cols = self._mlic.size
		tot = rows * cols

		interpolated_basis = []

		self._logger.info(msg=f'Starting interpolation for all pixels ({tot}). ')

		if save_dir:
			ISUtils.check_output(path=save_dir, logger=self._logger_verbose)
			self._logger.info(msg=f'Saving plots to {save_dir}. ')

		timer = Timer()

		for i, (px, py) in enumerate(itertools.product(range(rows), range(cols))):

			if progress and i % progress == 0:
				self._logger.info(msg=f'Interpolating pixel {i} of {tot} ({i/tot:.2%}) - Elapsed time: {timer}. ')
			
			bi = self.get_pixel_interpolation(pixel=(px, py))

			interpolated_basis.append(bi)
		
		self._logger.info(msg=f'Interpolation completed in {timer} . ')

		return BasisInterpolationCollection(basis_interpolations=interpolated_basis, out_shape=self._mlic.size)