"""Quality metrics for calibration assessment."""

from enum import Enum
from typing import List
from .calibrator import CalibrationResult, ImageDetection


class ImageQuality(Enum):
    """Image quality classification based on reprojection error."""

    EXCELLENT = ("Excellent", 0.0, 0.3, (0, 200, 0))  # Green - BGR format
    GOOD = ("Good", 0.3, 0.5, (0, 150, 0))  # Dark green
    ACCEPTABLE = ("Acceptable", 0.5, 1.0, (0, 200, 200))  # Yellow (was cyan by mistake)
    POOR = ("Poor", 1.0, 2.0, (0, 140, 255))  # Orange (was backwards)
    BAD = ("Bad", 2.0, float('inf'), (0, 0, 200))  # Red
    NOT_CALIBRATED = ("Not calibrated", 0.0, 0.0, (128, 128, 128))  # Gray

    def __init__(self, label: str, min_error: float, max_error: float, color: tuple):
        self.label = label
        self.min_error = min_error
        self.max_error = max_error
        self.color = color  # BGR color for OpenCV

    @classmethod
    def from_error(cls, error: float) -> 'ImageQuality':
        """Classify quality from reprojection error.

        Args:
            error: Reprojection error in pixels

        Returns:
            ImageQuality enum value
        """
        if error < 0:
            return cls.NOT_CALIBRATED

        for quality in cls:
            if quality == cls.NOT_CALIBRATED:
                continue
            if quality.min_error <= error < quality.max_error:
                return quality

        return cls.BAD


class QualityMetrics:
    """Calculate and analyze calibration quality metrics."""

    @staticmethod
    def classify_images(detections: List[ImageDetection]) -> dict:
        """Classify all images by quality.

        Args:
            detections: List of image detections with errors

        Returns:
            Dictionary mapping ImageQuality to list of detections
        """
        classification = {quality: [] for quality in ImageQuality}

        for detection in detections:
            if detection.reprojection_error > 0:
                quality = ImageQuality.from_error(detection.reprojection_error)
            else:
                quality = ImageQuality.NOT_CALIBRATED

            classification[quality].append(detection)

        return classification

    @staticmethod
    def get_statistics(result: CalibrationResult) -> dict:
        """Get calibration quality statistics.

        Args:
            result: Calibration result

        Returns:
            Dictionary with statistics
        """
        classification = QualityMetrics.classify_images(result.detections)

        # Count valid (non-excluded) detections
        valid_detections = [d for d in result.detections if not d.excluded]
        excluded_count = len(result.detections) - len(valid_detections)

        # Count calibrated images
        calibrated = [d for d in valid_detections if d.reprojection_error > 0]

        stats = {
            'total_images': len(result.detections),
            'valid_detections': len(valid_detections),
            'excluded_images': excluded_count,
            'calibrated_images': len(calibrated),
            'rms_error': result.rms_error,
            'quality_counts': {
                'excellent': len(classification[ImageQuality.EXCELLENT]),
                'good': len(classification[ImageQuality.GOOD]),
                'acceptable': len(classification[ImageQuality.ACCEPTABLE]),
                'poor': len(classification[ImageQuality.POOR]),
                'bad': len(classification[ImageQuality.BAD]),
            }
        }

        # Calculate percentages
        if len(calibrated) > 0:
            # Create a list of keys to avoid modifying dict during iteration
            quality_keys = list(stats['quality_counts'].keys())
            for key in quality_keys:
                count = stats['quality_counts'][key]
                stats['quality_counts'][f'{key}_percent'] = (count / len(calibrated)) * 100

        return stats
