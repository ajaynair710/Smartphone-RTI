# Smartphone-based RTI (Reflectance Transformation Imaging)

This project implements the Reflectance Transformation Imaging (RTI) technique as described in the paper *"On-the-go Reflectance Transformation Imaging with Ordinary Smartphones"* by M. Pistellato and F. Bergamasco (2022). RTI enables the creation of highly detailed images that showcase the surface texture and features of an object by capturing multiple images under different lighting conditions. The approach here leverages an ordinary smartphone to collect, process, and visualize the results, making RTI accessible on-the-go.

## Project Structure

This repository is organized into multiple directories and scripts, each serving specific functionalities necessary for implementing RTI on smartphones.

### Directory Structure

1. **`utils/`**  
   Contains general utility functions and core I/O tools.
   - `misc.py`: Includes general-purpose utility functions that are used across different modules.
   - `io_.py`: Provides core I/O functions for handling file paths, logging, and processing audio/video files.

2. **`src/`**  
   Contains the core implementation of RTI techniques, including image processing, calibration, and more.
   - `interpolation.py`: Implements interpolation techniques (Radial Basis Functions (RBF) and Polynomial Texture Maps (PTM)) for each pixel in the Multi-Light Image Collection (MLIC) and constructs basis functions fitted to the collected data.
   - `calibration.py`: Defines a data class for storing camera calibration settings, which can be saved and loaded from a file.
   - `preprocessing.py`: Implements preprocessing operations for RTI, including video synchronization, frame rate alignment, and camera calibration.
   - `thresholding.py`: Implements image binarization techniques prior to marker detection.
   - `marker.py`: Contains logic for detecting markers in videos, drawing markers on frames, warping the object in the video, and estimating camera pose.
   - `mlic.py`: Defines the `Multi-Light Image Collection (MLIC)` class for dataset manipulation and file storage, supporting synchronized video streams and dynamic camera pose estimation.
   - `typing.py`: Defines type aliases used throughout other modules for improved code clarity.
   - `stream.py`: Defines classes for video streaming, enabling concurrent visualization of multiple processing steps with togglable options.

### Scripts

1. **`calibration.py`**: A script to calibrate the camera, setting up essential parameters for the RTI process.
2. **`analysis.py`**: A script to perform analysis on the RTI data, including reconstruction and visualization.
3. **`sync.py`**: A script for synchronizing video streams to ensure proper alignment of the multi-light image collection.
4. **`plot_graph.py`**: A script that visualizes and plots graphs of data collected during the RTI process, useful for performance analysis and debugging.
5. **`relighting.py`**: A script to perform the relighting of 3D models or objects based on the RTI data.

## Key Features

- **Real-time video synchronization**: Ensures that images captured by the smartphone camera under various light sources are synchronized.
- **Camera calibration**: Helps in setting up accurate camera parameters to get correct light source information and reconstruct surface details.
- **Marker detection**: Detects and tracks markers in videos, facilitating the alignment of lighting and the estimation of camera positions.
- **Relighting**: Allows users to relight a 3D model or object from different light sources, highlighting the surface features.
- **Interactive visualization**: Visualizes multiple processing steps concurrently, giving users insight into the RTI process in real-time.
- **Image reconstruction**: Uses interpolation techniques like Radial Basis Functions (RBF) and Polynomial Texture Maps (PTM) to reconstruct detailed surface images.

## Installation

To get started with the Smartphone-based RTI project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/smartphone-based-rti.git
cd smartphone-based-rti
pip install -r requirements.txt
```

## Usage

1. **Calibration**: 
   Start by calibrating your smartphone's camera using the provided `calibration.py` script. This step ensures that the camera parameters are correctly set up for accurate RTI processing.

   ```bash
   python calibration.py
   ```

2. **Synchronizing Videos**: 
   After calibration, use the `sync.py` script to synchronize your video streams, ensuring the lighting conditions are aligned.

   ```bash
   python sync.py
   ```

3. **Analysis & Visualization**: 
   After processing, you can analyze the data and visualize the results using `analysis.py` and `plot_graph.py` to gain insights into the RTI process.

   ```bash
   python analysis.py
   python plot_graph.py
   ```

4. **Relighting**: 
   Once the videos are synchronized and analyzed, use `relighting.py` to perform the relighting process and reconstruct the surface details of the object.

   ```bash
   python relighting.py
   ```

## References

- Pistellato, M., & Bergamasco, F. (2022). *On-the-go Reflectance Transformation Imaging with Ordinary Smartphones*.

## Contributing

Contributions to this project are welcome! Feel free to open issues, submit pull requests, or suggest improvements. Please ensure that you follow the existing coding style and add tests where necessary.
