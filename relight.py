import os
import pickle
import numpy as np
import cv2
from dotenv import load_dotenv

from src.interpolation import MLICBasisInterpolator
from utils.io import IOUtils, FileLogger

load_dotenv()

OUT_DIR = os.getenv('OUT_DIR', '.')
EXP_NAME = 'coin1'
INTERPOLATION_ALGO = 'rbf'
INTERPOLATION_FILENAME = f"{EXP_NAME}_{INTERPOLATION_ALGO}.pkl"
INTERPOLATION_PATH = os.path.join(OUT_DIR, EXP_NAME, 'interpolation', INTERPOLATION_ALGO, INTERPOLATION_FILENAME)

def load_interpolator(path, logger):
    """Manually load the interpolator object from a pickle file."""
    try:
        with open(path, 'rb') as f:
            interpolator = pickle.load(f)
            logger.info("Interpolator loaded successfully.")
            return interpolator
    except Exception as e:
        logger.error(f"Failed to load interpolator: {e}")
        raise e


def normalize_vector(vec):
    """Normalize a 3D vector."""
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec


class InteractiveLighting:
    def __init__(self, interpolator, logger, window_size=(500, 500)):
        self.interpolator = interpolator
        self.logger = logger
        self.window_size = window_size
        self.circle_radius = 10
        self.circle_position = (window_size[0] // 2, window_size[1] // 2)
        self.dragging = False

    def on_mouse_event(self, event, x, y, flags, param):
        """Handle mouse events for dragging the circle."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.circle_position = (x, y)

    def calculate_light_direction(self):
        """Calculate light direction based on the circle's position."""
        x, y = self.circle_position
        w, h = self.window_size
        # Normalize coordinates to [-1, 1] range
        light_x = 2 * (x / w) - 1
        light_y = 2 * (y / h) - 1
        light_z = np.sqrt(1 - min(1, light_x ** 2 + light_y ** 2))  # Ensure the vector is on the unit sphere
        return normalize_vector([light_x, light_y, light_z])

    def render_frame(self):
        """Render the current relit frame based on the circle's position."""
        light_direction = self.calculate_light_direction()
        self.logger.info(f"Light direction: {light_direction}")

        # Generate the relit image
        try:
            relit_image = self.interpolator.get_interpolation_frame(light_direction)
        except AttributeError as e:
            self.logger.error(f"Error in rendering frame: {e}")
            return None

        if relit_image is None:
            self.logger.error("Failed to generate relit image.")
            return None

        # Normalize image for display
        return cv2.normalize(relit_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    def draw_arrow(self, frame):
        """Draw an arrow from the center to the circle's position."""
        center = (self.window_size[0] // 2, self.window_size[1] // 2)
        cv2.arrowedLine(frame, center, self.circle_position, (0, 0, 255), 2)

    def run(self):
        """Run the interactive relighting program."""
        cv2.namedWindow("Interactive Relighting")
        cv2.setMouseCallback("Interactive Relighting", self.on_mouse_event)

        while True:
            # Create a blank white background
            frame = np.ones((*self.window_size, 3), dtype=np.uint8) * 255

            # Draw the circle and arrow
            self.draw_arrow(frame)
            cv2.circle(frame, self.circle_position, self.circle_radius, (0, 0, 0), -1)

            # Render the relit frame
            relit_image = self.render_frame()

            # Display the relit image alongside the interactive circle
            if relit_image is not None:
                # Resize relit image to match the interactive window size
                relit_resized = cv2.resize(relit_image, self.window_size)
                combined_frame = np.hstack((frame, cv2.cvtColor(relit_resized, cv2.COLOR_GRAY2BGR)))
                cv2.imshow("Interactive Relighting", combined_frame)
            else:
                cv2.imshow("Interactive Relighting", frame)

            # Quit on pressing 'q'
            key = cv2.waitKey(1)
            if key == ord('q'):
                self.logger.info("Exiting interactive relighting.")
                break

        cv2.destroyAllWindows()


def main():
    # Logger
    logger = FileLogger(file=os.path.join(OUT_DIR, EXP_NAME, 'relighting.log'))
    logger.info(f'Interactive relighting for experiment {EXP_NAME} using {INTERPOLATION_ALGO} interpolation.')

    # Load interpolation data
    logger.info('LOADING INTERPOLATION DATA')
    interpolator = load_interpolator(INTERPOLATION_PATH, logger)
    logger.info('INTERPOLATION DATA LOADED SUCCESSFULLY')

    # Start interactive relighting
    lighting = InteractiveLighting(interpolator, logger)
    lighting.run()


if __name__ == "__main__":
    main()
