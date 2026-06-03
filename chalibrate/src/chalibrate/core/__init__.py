"""Core calibration functionality."""

from .board_config import BoardConfig
from .board_detector import BoardDetector, auto_detect_board
from .calibrator import CalibrationResult, Calibrator, ImageDetection
from .metrics import ImageQuality, QualityMetrics

__all__ = [
    'BoardConfig',
    'Calibrator',
    'CalibrationResult',
    'ImageDetection',
    'QualityMetrics',
    'ImageQuality',
    'BoardDetector',
    'auto_detect_board',
]
