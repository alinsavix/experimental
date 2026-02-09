"""Image quality analysis for calibration."""

from dataclasses import dataclass, field
from typing import List
import numpy as np
import cv2


@dataclass
class QualityReport:
    """Image quality analysis report."""

    blur_score: float
    """Laplacian variance (higher = sharper)."""

    brightness_mean: float
    """Mean brightness (0-255)."""

    is_blurry: bool
    """Whether image is too blurry."""

    is_too_dark: bool
    """Whether image is too dark."""

    is_too_bright: bool
    """Whether image is too bright."""

    issues: List[str] = field(default_factory=list)
    """List of quality issues detected."""

    @property
    def passes(self) -> bool:
        """Whether image passes quality checks."""
        return not (self.is_blurry or self.is_too_dark or self.is_too_bright)

    @property
    def status_text(self) -> str:
        """Human-readable status summary."""
        if self.passes:
            return "Good quality"
        return ", ".join(self.issues)


class ImageQualityAnalyzer:
    """Analyzes image quality for calibration suitability."""

    def __init__(
        self,
        blur_threshold: float = 30.0,
        brightness_min: float = 30.0,
        brightness_max: float = 225.0
    ):
        """Initialize analyzer.

        Args:
            blur_threshold: Minimum Laplacian variance (higher = sharper, 30.0 is a good default)
            brightness_min: Minimum acceptable mean brightness (0-255)
            brightness_max: Maximum acceptable mean brightness (0-255)
        """
        self.blur_threshold = blur_threshold
        self.brightness_min = brightness_min
        self.brightness_max = brightness_max

    def analyze_blur(self, image: np.ndarray) -> float:
        """Detect blur using Laplacian variance.

        The Laplacian operator highlights edges and rapid intensity changes.
        Sharp images have high variance in Laplacian, blurry images have low variance.

        Args:
            image: Grayscale image (single channel)

        Returns:
            Laplacian variance (higher values = sharper image)
        """
        # Compute Laplacian
        laplacian = cv2.Laplacian(image, cv2.CV_64F)

        # Return variance
        return float(laplacian.var())

    def analyze_brightness(self, image: np.ndarray) -> float:
        """Analyze image brightness.

        Args:
            image: Grayscale image (single channel)

        Returns:
            Mean brightness (0-255)
        """
        return float(image.mean())

    def analyze_image(self, image: np.ndarray) -> QualityReport:
        """Perform comprehensive quality analysis.

        Args:
            image: BGR or grayscale image

        Returns:
            QualityReport with analysis results
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Analyze blur
        blur_score = self.analyze_blur(gray)
        is_blurry = blur_score < self.blur_threshold

        # Analyze brightness
        brightness_mean = self.analyze_brightness(gray)
        is_too_dark = brightness_mean < self.brightness_min
        is_too_bright = brightness_mean > self.brightness_max

        # Build issues list
        issues = []
        if is_blurry:
            issues.append(f"Blurry (score: {blur_score:.1f}, threshold: {self.blur_threshold})")
        if is_too_dark:
            issues.append(f"Too dark (brightness: {brightness_mean:.1f}, min: {self.brightness_min})")
        if is_too_bright:
            issues.append(f"Too bright (brightness: {brightness_mean:.1f}, max: {self.brightness_max})")

        return QualityReport(
            blur_score=blur_score,
            brightness_mean=brightness_mean,
            is_blurry=is_blurry,
            is_too_dark=is_too_dark,
            is_too_bright=is_too_bright,
            issues=issues,
        )
