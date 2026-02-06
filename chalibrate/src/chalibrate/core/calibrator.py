"""Camera calibration using ChArUco boards."""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
import numpy as np
import cv2

from .board_config import BoardConfig


@dataclass
class ImageDetection:
    """Detection results for a single image."""

    image_path: str
    image: np.ndarray
    charuco_corners: Optional[np.ndarray]  # Detected corner positions
    charuco_ids: Optional[np.ndarray]  # Corner IDs
    marker_corners: Optional[List]  # ArUco marker corners for visualization
    marker_ids: Optional[np.ndarray]  # ArUco marker IDs
    reprojection_error: float = 0.0  # Per-image error (set after calibration)
    excluded: bool = False  # Whether to exclude from calibration

    @property
    def has_detection(self) -> bool:
        """Check if board was successfully detected."""
        return self.charuco_corners is not None and len(self.charuco_corners) > 0

    @property
    def num_corners(self) -> int:
        """Number of detected ChArUco corners."""
        return len(self.charuco_corners) if self.has_detection else 0


@dataclass
class CalibrationResult:
    """Results from camera calibration."""

    camera_matrix: np.ndarray  # 3x3 camera intrinsic matrix
    dist_coeffs: np.ndarray  # Distortion coefficients [K1, K2, P1, P2, K3, K4]
    rms_error: float  # Overall RMS reprojection error
    image_size: Tuple[int, int]  # (width, height)
    detections: List[ImageDetection]  # All image detections with per-image errors

    @property
    def fx(self) -> float:
        """Focal length in X direction."""
        return self.camera_matrix[0, 0]

    @property
    def fy(self) -> float:
        """Focal length in Y direction."""
        return self.camera_matrix[1, 1]

    @property
    def cx(self) -> float:
        """Principal point X coordinate."""
        return self.camera_matrix[0, 2]

    @property
    def cy(self) -> float:
        """Principal point Y coordinate."""
        return self.camera_matrix[1, 2]

    @property
    def k1(self) -> float:
        """Radial distortion coefficient K1."""
        return self.dist_coeffs[0, 0]

    @property
    def k2(self) -> float:
        """Radial distortion coefficient K2."""
        return self.dist_coeffs[0, 1]

    @property
    def p1(self) -> float:
        """Tangential distortion coefficient P1."""
        return self.dist_coeffs[0, 2]

    @property
    def p2(self) -> float:
        """Tangential distortion coefficient P2."""
        return self.dist_coeffs[0, 3]

    @property
    def k3(self) -> float:
        """Radial distortion coefficient K3."""
        return self.dist_coeffs[0, 4] if len(self.dist_coeffs[0]) > 4 else 0.0

    @property
    def k4(self) -> float:
        """Radial distortion coefficient K4."""
        return self.dist_coeffs[0, 5] if len(self.dist_coeffs[0]) > 5 else 0.0

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'camera_matrix': self.camera_matrix.tolist(),
            'distortion_coefficients': self.dist_coeffs.tolist()[0],
            'rms_error': float(self.rms_error),
            'image_size': list(self.image_size),
            'focal_length': {
                'fx': float(self.fx),
                'fy': float(self.fy),
            },
            'principal_point': {
                'cx': float(self.cx),
                'cy': float(self.cy),
            },
            'distortion': {
                'K1': float(self.k1),
                'K2': float(self.k2),
                'P1': float(self.p1),
                'P2': float(self.p2),
                'K3': float(self.k3),
                'K4': float(self.k4),
            }
        }


class Calibrator:
    """Camera calibration using ChArUco boards."""

    def __init__(self, board_config: BoardConfig):
        """Initialize calibrator.

        Args:
            board_config: ChArUco board configuration
        """
        self.board_config = board_config
        self.board = board_config.create_board()
        self.dictionary = board_config.get_dictionary()
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.detector_params)

    def detect_boards(
        self,
        images: List[Tuple[str, np.ndarray]],
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[ImageDetection]:
        """Detect ChArUco boards in images.

        Args:
            images: List of (path, image) tuples
            progress_callback: Optional callback(current, total, image_path)

        Returns:
            List of ImageDetection objects
        """
        detections = []

        for i, (path, image) in enumerate(images):
            if progress_callback:
                progress_callback(i + 1, len(images), path)

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect ArUco markers
            marker_corners, marker_ids, _ = self.detector.detectMarkers(gray)

            charuco_corners = None
            charuco_ids = None

            # Interpolate ChArUco corners if markers found
            if marker_ids is not None and len(marker_ids) > 0:
                result = cv2.aruco.interpolateCornersCharuco(
                    marker_corners, marker_ids, gray, self.board
                )
                if result[0] is not None and result[0] > 0:
                    charuco_corners, charuco_ids = result[1], result[2]

            detection = ImageDetection(
                image_path=path,
                image=image,
                charuco_corners=charuco_corners,
                charuco_ids=charuco_ids,
                marker_corners=marker_corners,
                marker_ids=marker_ids,
            )
            detections.append(detection)

        return detections

    def calibrate(
        self,
        detections: List[ImageDetection],
        image_size: Tuple[int, int]
    ) -> CalibrationResult:
        """Perform camera calibration.

        Args:
            detections: List of image detections
            image_size: Image size (width, height)

        Returns:
            CalibrationResult object

        Raises:
            ValueError: If insufficient valid detections
        """
        # Filter to non-excluded detections with valid corners
        # Also filter out detections with too few corners (need at least 4 for calibration)
        valid_detections = [
            d for d in detections
            if d.has_detection and not d.excluded and len(d.charuco_corners) >= 4
        ]

        if len(valid_detections) < 3:
            raise ValueError(
                f"Need at least 3 valid detections with 4+ corners, got {len(valid_detections)}"
            )

        # Prepare data for calibration
        all_corners = []
        all_ids = []

        for detection in valid_detections:
            all_corners.append(detection.charuco_corners)
            all_ids.append(detection.charuco_ids)

        # Perform calibration
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            all_corners,
            all_ids,
            self.board,
            image_size,
            None,
            None
        )

        # Calculate per-image reprojection errors
        for i, detection in enumerate(valid_detections):
            if detection.has_detection:
                # Project points
                obj_points = self.board.getChessboardCorners()[detection.charuco_ids]
                img_points, _ = cv2.projectPoints(
                    obj_points,
                    rvecs[i],
                    tvecs[i],
                    camera_matrix,
                    dist_coeffs
                )

                # Calculate error
                error = cv2.norm(
                    detection.charuco_corners,
                    img_points,
                    cv2.NORM_L2
                ) / len(img_points)
                detection.reprojection_error = error

        return CalibrationResult(
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            rms_error=ret,
            image_size=image_size,
            detections=detections,
        )

    def draw_detection(self, detection: ImageDetection) -> np.ndarray:
        """Draw detected ChArUco corners on image.

        Args:
            detection: Image detection to visualize

        Returns:
            Image with detection overlay
        """
        output = detection.image.copy()

        # Draw ArUco markers
        if detection.marker_corners is not None and detection.marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(output, detection.marker_corners, detection.marker_ids)

        # Draw ChArUco corners
        if detection.has_detection:
            for i, corner in enumerate(detection.charuco_corners):
                corner_id = detection.charuco_ids[i][0]
                x, y = int(corner[0][0]), int(corner[0][1])

                # Draw circle at corner
                cv2.circle(output, (x, y), 8, (0, 255, 0), -1)

                # Draw corner ID
                cv2.putText(
                    output, str(corner_id), (x + 10, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2
                )

        return output

    def undistort_image(
        self,
        image: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        crop_to_valid: bool = True
    ) -> np.ndarray:
        """Undistort an image using calibration results.

        Args:
            image: Input image
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Distortion coefficients
            crop_to_valid: Whether to crop to valid region

        Returns:
            Undistorted image
        """
        h, w = image.shape[:2]

        if crop_to_valid:
            # Get optimal new camera matrix
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                camera_matrix, dist_coeffs, (w, h), 1, (w, h)
            )

            # Undistort
            undistorted = cv2.undistort(
                image, camera_matrix, dist_coeffs, None, new_camera_matrix
            )

            # Crop to valid region
            if roi != (0, 0, 0, 0):
                x, y, w, h = roi
                undistorted = undistorted[y:y+h, x:x+w]
        else:
            # Simple undistort without cropping
            undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)

        return undistorted
