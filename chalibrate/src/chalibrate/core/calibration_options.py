"""Configuration options for camera calibration accuracy improvements."""

from dataclasses import dataclass, field
from typing import Tuple, Optional
import cv2


@dataclass
class CalibrationOptions:
    """Configuration for camera calibration accuracy features.

    This class provides control over various accuracy-enhancing features:
    - Subpixel corner refinement
    - Image quality filtering
    - Advanced calibration flags
    - Iterative outlier removal
    - ArUco detector parameters
    - Image preprocessing
    """

    # === Subpixel Corner Refinement ===
    enable_subpixel: bool = True
    """Enable subpixel corner refinement for detected ChArUco corners."""

    subpixel_window_size: Tuple[int, int] = (5, 5)
    """Window size for subpixel refinement (width, height)."""

    subpixel_max_iterations: int = 30
    """Maximum iterations for subpixel refinement."""

    subpixel_epsilon: float = 0.001
    """Convergence epsilon for subpixel refinement."""

    # === Image Quality Filtering ===
    enable_quality_filter: bool = False
    """Enable automatic image quality filtering. Disabled by default - enable if you have known quality issues."""

    blur_threshold: float = 30.0
    """Minimum Laplacian variance (higher = sharper). Values < 20 typically indicate significant blur."""

    brightness_min: float = 30.0
    """Minimum acceptable mean brightness (0-255)."""

    brightness_max: float = 225.0
    """Maximum acceptable mean brightness (0-255)."""

    # === Calibration Flags ===
    use_rational_model: bool = False
    """Use rational distortion model (K4, K5, K6 coefficients)."""

    use_thin_prism: bool = False
    """Use thin prism distortion model (S1, S2, S3, S4 coefficients)."""

    fix_principal_point: bool = False
    """Fix principal point at image center."""

    fix_aspect_ratio: bool = False
    """Fix aspect ratio (fx/fy) to 1.0."""

    zero_tangent_dist: bool = False
    """Set tangential distortion coefficients (P1, P2) to zero."""

    fix_k1: bool = False
    """Fix K1 radial distortion coefficient."""

    fix_k2: bool = False
    """Fix K2 radial distortion coefficient."""

    fix_k3: bool = False
    """Fix K3 radial distortion coefficient."""

    # === Iterative Refinement ===
    enable_iterative_refinement: bool = False
    """Enable iterative outlier removal and re-calibration."""

    max_refinement_iterations: int = 3
    """Maximum number of refinement iterations."""

    outlier_percentile: float = 95.0
    """Percentile threshold for outlier detection (0-100)."""

    outlier_multiplier: float = 2.0
    """Multiplier for median-based outlier detection."""

    min_images_after_outliers: int = 5
    """Minimum images to retain after outlier removal."""

    # === ArUco Detector Parameters ===
    adaptive_thresh_win_size_min: int = 3
    """Minimum window size for adaptive thresholding."""

    adaptive_thresh_win_size_max: int = 23
    """Maximum window size for adaptive thresholding."""

    adaptive_thresh_win_size_step: int = 10
    """Step size for adaptive thresholding window."""

    adaptive_thresh_constant: float = 7.0
    """Constant subtracted from mean in adaptive thresholding."""

    corner_refinement_method: int = cv2.aruco.CORNER_REFINE_SUBPIX
    """ArUco corner refinement method (NONE, SUBPIX, CONTOUR, APRILTAG)."""

    # === Preprocessing ===
    enable_clahe: bool = False
    """Enable CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing."""

    clahe_clip_limit: float = 2.0
    """CLAHE clip limit."""

    clahe_tile_size: Tuple[int, int] = (8, 8)
    """CLAHE tile grid size."""

    def get_calibration_flags(self) -> int:
        """Build OpenCV calibration flags bitmask.

        Returns:
            Integer bitmask for cv2.aruco.calibrateCameraCharuco() flags parameter
        """
        flags = 0

        if self.use_rational_model:
            flags |= cv2.CALIB_RATIONAL_MODEL

        if self.use_thin_prism:
            flags |= cv2.CALIB_THIN_PRISM_MODEL

        if self.fix_principal_point:
            flags |= cv2.CALIB_FIX_PRINCIPAL_POINT

        if self.fix_aspect_ratio:
            flags |= cv2.CALIB_FIX_ASPECT_RATIO

        if self.zero_tangent_dist:
            flags |= cv2.CALIB_ZERO_TANGENT_DIST

        if self.fix_k1:
            flags |= cv2.CALIB_FIX_K1

        if self.fix_k2:
            flags |= cv2.CALIB_FIX_K2

        if self.fix_k3:
            flags |= cv2.CALIB_FIX_K3

        return flags

    def get_subpixel_criteria(self) -> Tuple[int, int, float]:
        """Get cv2.TermCriteria for subpixel refinement.

        Returns:
            Tuple of (type, max_iterations, epsilon) for cv2.TermCriteria
        """
        return (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            self.subpixel_max_iterations,
            self.subpixel_epsilon
        )

    @classmethod
    def from_preset(cls, preset: str) -> 'CalibrationOptions':
        """Create CalibrationOptions from a preset configuration.

        Args:
            preset: Preset name ('default', 'high_accuracy', 'fast', 'fisheye', 'webcam')

        Returns:
            CalibrationOptions instance

        Raises:
            ValueError: If preset name is unknown
        """
        preset = preset.lower()

        if preset == 'default':
            return cls()

        elif preset == 'high_accuracy':
            return cls(
                enable_subpixel=True,
                enable_quality_filter=True,
                enable_iterative_refinement=True,
                use_rational_model=True,
                max_refinement_iterations=5,
                outlier_percentile=90.0,
                blur_threshold=50.0,
            )

        elif preset == 'fast':
            return cls(
                enable_subpixel=False,
                enable_quality_filter=False,
                enable_iterative_refinement=False,
            )

        elif preset == 'fisheye':
            return cls(
                enable_subpixel=True,
                enable_quality_filter=True,
                use_rational_model=True,
                use_thin_prism=True,
                adaptive_thresh_win_size_min=5,
                adaptive_thresh_win_size_max=50,
            )

        elif preset == 'webcam':
            return cls(
                enable_subpixel=True,
                enable_quality_filter=True,
                enable_iterative_refinement=True,
                enable_clahe=True,
                blur_threshold=20.0,  # Webcams often have lower quality
                brightness_min=40.0,
                brightness_max=215.0,
            )

        else:
            raise ValueError(
                f"Unknown preset '{preset}'. Valid options: "
                "default, high_accuracy, fast, fisheye, webcam"
            )

    def to_dict(self) -> dict:
        """Convert options to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            'subpixel': {
                'enabled': self.enable_subpixel,
                'window_size': self.subpixel_window_size,
                'max_iterations': self.subpixel_max_iterations,
                'epsilon': self.subpixel_epsilon,
            },
            'quality_filter': {
                'enabled': self.enable_quality_filter,
                'blur_threshold': self.blur_threshold,
                'brightness_min': self.brightness_min,
                'brightness_max': self.brightness_max,
            },
            'calibration_flags': {
                'rational_model': self.use_rational_model,
                'thin_prism': self.use_thin_prism,
                'fix_principal_point': self.fix_principal_point,
                'fix_aspect_ratio': self.fix_aspect_ratio,
                'zero_tangent_dist': self.zero_tangent_dist,
                'fix_k1': self.fix_k1,
                'fix_k2': self.fix_k2,
                'fix_k3': self.fix_k3,
            },
            'iterative_refinement': {
                'enabled': self.enable_iterative_refinement,
                'max_iterations': self.max_refinement_iterations,
                'outlier_percentile': self.outlier_percentile,
                'outlier_multiplier': self.outlier_multiplier,
                'min_images': self.min_images_after_outliers,
            },
            'preprocessing': {
                'clahe_enabled': self.enable_clahe,
                'clahe_clip_limit': self.clahe_clip_limit,
                'clahe_tile_size': self.clahe_tile_size,
            },
        }
