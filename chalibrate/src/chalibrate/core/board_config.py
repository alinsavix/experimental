"""ChArUco board configuration."""

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class BoardConfig:
    """Configuration for ChArUco calibration board."""

    squares_x: int  # Number of squares in X direction
    squares_y: int  # Number of squares in Y direction
    square_length: float  # Square side length in mm
    marker_length: float  # ArUco marker side length in mm
    dict_name: str = 'DICT_6X6_250'  # ArUco dictionary name

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.squares_x < 3 or self.squares_y < 3:
            raise ValueError("Board must have at least 3x3 squares")
        if self.square_length <= 0 or self.marker_length <= 0:
            raise ValueError("Lengths must be positive")
        if self.marker_length >= self.square_length:
            raise ValueError("Marker length must be less than square length")

    def create_board(self):
        """Create OpenCV ChArUco board object.

        Returns:
            cv2.aruco.CharucoBoard: The board object for detection
        """
        dictionary = self.get_dictionary()
        board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length,
            self.marker_length,
            dictionary
        )
        return board

    def get_dictionary(self):
        """Get ArUco dictionary from name.

        Returns:
            cv2.aruco.Dictionary: The ArUco dictionary
        """
        dict_attr = getattr(cv2.aruco, self.dict_name, None)
        if dict_attr is None:
            raise ValueError(f"Unknown ArUco dictionary: {self.dict_name}")
        return cv2.aruco.getPredefinedDictionary(dict_attr)

    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            'squares_x': self.squares_x,
            'squares_y': self.squares_y,
            'square_length': self.square_length,
            'marker_length': self.marker_length,
            'dict_name': self.dict_name,
        }
