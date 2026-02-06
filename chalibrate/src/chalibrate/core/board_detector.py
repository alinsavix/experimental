"""Automatic ChArUco board configuration detection."""

from typing import Optional, List, Tuple
import cv2
import numpy as np
from dataclasses import dataclass

from .board_config import BoardConfig


@dataclass
class DetectionCandidate:
    """Candidate board configuration from detection."""
    dict_name: str
    squares_x: int
    squares_y: int
    marker_count: int
    corner_count: int
    confidence: float


class BoardDetector:
    """Automatically detect ChArUco board configuration from images."""

    # Common ArUco dictionaries to try
    DICTIONARIES = [
        'DICT_4X4_50',
        'DICT_4X4_100',
        'DICT_4X4_250',
        'DICT_5X5_50',
        'DICT_5X5_100',
        'DICT_5X5_250',
        'DICT_6X6_50',
        'DICT_6X6_100',
        'DICT_6X6_250',
        'DICT_7X7_50',
        'DICT_7X7_100',
        'DICT_7X7_250',
    ]

    @staticmethod
    def detect_from_images(
        images: List[Tuple[str, np.ndarray]],
        max_images: int = 5
    ) -> Optional[BoardConfig]:
        """Detect board configuration from images.

        Args:
            images: List of (path, image) tuples
            max_images: Maximum number of images to analyze

        Returns:
            Detected BoardConfig or None if detection fails
        """
        if not images:
            return None

        # Sample images if too many
        sample_images = images[:max_images]

        # Try each dictionary and find best match
        best_candidate = None
        best_confidence = 0.0

        for dict_name in BoardDetector.DICTIONARIES:
            candidate = BoardDetector._try_dictionary(sample_images, dict_name)
            if candidate and candidate.confidence > best_confidence:
                best_candidate = candidate
                best_confidence = candidate.confidence

        if not best_candidate:
            return None

        # Estimate physical sizes (use common defaults)
        # Could be improved by analyzing pixel ratios
        square_length = 30.0  # mm (common size)
        marker_length = 20.0  # mm (common ratio is 0.67)

        try:
            return BoardConfig(
                squares_x=best_candidate.squares_x,
                squares_y=best_candidate.squares_y,
                square_length=square_length,
                marker_length=marker_length,
                dict_name=best_candidate.dict_name,
            )
        except ValueError:
            return None

    @staticmethod
    def _try_dictionary(
        images: List[Tuple[str, np.ndarray]],
        dict_name: str
    ) -> Optional[DetectionCandidate]:
        """Try detecting board with a specific dictionary.

        Args:
            images: Sample images
            dict_name: ArUco dictionary name

        Returns:
            DetectionCandidate or None if no detection
        """
        try:
            dict_attr = getattr(cv2.aruco, dict_name)
            dictionary = cv2.aruco.getPredefinedDictionary(dict_attr)
            detector_params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
        except Exception:
            return None

        # Collect marker IDs from all images
        all_marker_ids = set()
        total_corners = 0
        detection_count = 0
        successful_interpolations = 0

        for _, image in images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect markers
            marker_corners, marker_ids, _ = detector.detectMarkers(gray)

            if marker_ids is not None and len(marker_ids) > 0:
                detection_count += 1
                all_marker_ids.update(marker_ids.flatten())

        # No detections
        if not all_marker_ids:
            return None

        # Infer board dimensions from marker IDs
        max_marker_id = max(all_marker_ids)
        possible_sizes = BoardDetector._infer_board_sizes(max_marker_id)

        if not possible_sizes:
            return None

        # Pick the most likely size (most square-like)
        best_size = min(possible_sizes, key=lambda s: abs(s[0] - s[1]))
        squares_x, squares_y = best_size

        # NOW validate by trying to interpolate corners with this board configuration
        # This is the key validation step - the dictionary must work with the board
        for _, image in images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            marker_corners, marker_ids, _ = detector.detectMarkers(gray)

            if marker_ids is not None and len(marker_ids) > 0:
                try:
                    board = cv2.aruco.CharucoBoard(
                        (squares_x, squares_y),
                        1.0,  # Dummy size for validation
                        0.67,  # Dummy size for validation
                        dictionary
                    )
                    result = cv2.aruco.interpolateCornersCharuco(
                        marker_corners, marker_ids, gray, board
                    )
                    if result[0] is not None and result[0] > 0:
                        total_corners += result[0]
                        successful_interpolations += 1
                except Exception:
                    continue

        # Calculate confidence based on:
        # - Number of images with successful corner interpolation (bonus if it works!)
        # - Number of unique markers found
        # - Number of images with detections
        expected_markers = (squares_x - 1) * (squares_y - 1)
        marker_ratio = len(all_marker_ids) / expected_markers
        image_ratio = detection_count / len(images)

        # Base confidence from marker detection
        base_confidence = (marker_ratio * 0.6 + image_ratio * 0.4) * 100

        # Bonus for successful corner interpolation (proves dictionary works)
        if total_corners > 0:
            interpolation_bonus = min(successful_interpolations / len(images), 1.0) * 30
            confidence = base_confidence + interpolation_bonus
        else:
            confidence = base_confidence

        return DetectionCandidate(
            dict_name=dict_name,
            squares_x=squares_x,
            squares_y=squares_y,
            marker_count=len(all_marker_ids),
            corner_count=total_corners,
            confidence=min(confidence, 100.0)
        )

    @staticmethod
    def _infer_board_sizes(max_marker_id: int) -> List[Tuple[int, int]]:
        """Infer possible board sizes from maximum marker ID.

        In a ChArUco board with dimensions NxM squares,
        there are (N-1)*(M-1) markers numbered 0 to (N-1)*(M-1)-1.

        Args:
            max_marker_id: Maximum marker ID detected

        Returns:
            List of (squares_x, squares_y) tuples
        """
        num_markers = max_marker_id + 1
        possible_sizes = []

        # Try different factorizations - support up to 25x25 boards
        for n_minus_1 in range(2, 25):
            if num_markers % n_minus_1 == 0:
                m_minus_1 = num_markers // n_minus_1
                if 2 <= m_minus_1 <= 25:
                    # Squares are one more than markers in each dimension
                    squares_x = n_minus_1 + 1
                    squares_y = m_minus_1 + 1
                    possible_sizes.append((squares_x, squares_y))

        return possible_sizes

    @staticmethod
    def estimate_physical_size(
        images: List[Tuple[str, np.ndarray]],
        board_config: BoardConfig
    ) -> Tuple[float, float]:
        """Estimate physical square and marker sizes from images.

        This analyzes the pixel dimensions of detected features to estimate
        physical sizes based on common ratios.

        Args:
            images: List of (path, image) tuples
            board_config: Detected board configuration

        Returns:
            (square_length, marker_length) in mm
        """
        # This is a simplified estimation
        # In practice, without a reference, we use common defaults
        # A more sophisticated approach could:
        # 1. Measure pixel distances between markers
        # 2. Use multiple images at different scales
        # 3. Ask user to provide one reference measurement

        # Common sizes:
        # - Small boards: 20mm squares, 15mm markers
        # - Medium boards: 30mm squares, 20mm markers
        # - Large boards: 40mm squares, 30mm markers

        # Default to medium size
        return (30.0, 20.0)


def auto_detect_board(
    images: List[Tuple[str, np.ndarray]],
    max_images: int = 5
) -> Tuple[Optional[BoardConfig], str]:
    """Auto-detect board configuration from images.

    Args:
        images: List of (path, image) tuples
        max_images: Maximum images to analyze

    Returns:
        (BoardConfig or None, status message)
    """
    if not images:
        return None, "No images provided"

    detector = BoardDetector()

    # First, check if any ArUco markers are detected at all
    has_markers = False
    for dict_name in BoardDetector.DICTIONARIES[:4]:  # Try first few
        try:
            dict_attr = getattr(cv2.aruco, dict_name)
            dictionary = cv2.aruco.getPredefinedDictionary(dict_attr)
            detector_obj = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

            img = images[0][1]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            marker_corners, marker_ids, _ = detector_obj.detectMarkers(gray)

            if marker_ids is not None and len(marker_ids) > 0:
                has_markers = True
                break
        except:
            continue

    if not has_markers:
        return None, (
            "No ArUco markers detected in images.\n\n"
            "Please ensure:\n"
            "- Images contain a ChArUco calibration board\n"
            "- Board is clearly visible and well-lit\n"
            "- Images are in focus"
        )

    config = detector.detect_from_images(images, max_images)

    if config:
        message = (
            f"Auto-detected board: {config.squares_x}x{config.squares_y} squares\n"
            f"Dictionary: {config.dict_name}\n"
            f"Estimated sizes: {config.square_length}mm squares, "
            f"{config.marker_length}mm markers\n\n"
            f"⚠️ Note: Physical sizes are estimated defaults.\n"
            f"Measure your actual board and update if needed.\n\n"
            f"If calibration fails, your board might be:\n"
            f"- Pure ArUco grid (not ChArUco with chessboard)\n"
            f"- Different dimensions than detected\n"
            f"- Non-standard configuration"
        )
        return config, message
    else:
        message = (
            "ArUco markers detected, but could not determine board configuration.\n\n"
            "Possible causes:\n"
            "- Images contain ArUco grid (not ChArUco board)\n"
            "- Non-standard board configuration\n"
            "- Insufficient marker visibility\n\n"
            "Please configure board parameters manually."
        )
        return None, message
