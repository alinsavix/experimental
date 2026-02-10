"""Spatial coverage analysis for calibration quality assessment."""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import cv2

from .calibrator import ImageDetection


@dataclass
class CoverageReport:
    """Report on spatial coverage of detected corners."""

    coverage_map: np.ndarray
    """2D array showing corner density per grid cell (normalized 0-1)."""

    coverage_percentage: float
    """Percentage of grid cells with at least one corner (0-100)."""

    empty_regions: List[str]
    """List of regions with no coverage (e.g., 'top-right', 'bottom-left')."""

    min_corners_per_region: int
    """Minimum number of corners in any non-empty region."""

    max_corners_per_region: int
    """Maximum number of corners in any region."""

    total_corners: int
    """Total number of corners across all images."""

    num_images_used: int
    """Number of images used for calibration."""

    is_adequate: bool
    """Whether coverage is adequate for good calibration."""

    warnings: List[str]
    """List of coverage warnings."""

    @property
    def quality_score(self) -> float:
        """Overall coverage quality score (0-100)."""
        # Combine multiple factors
        score = 0.0

        # Coverage percentage (0-40 points)
        score += min(self.coverage_percentage * 0.4, 40)

        # Image count (0-20 points: <5=0, 5-10=10, 10-20=15, 20+=20)
        if self.num_images_used >= 20:
            score += 20
        elif self.num_images_used >= 10:
            score += 15
        elif self.num_images_used >= 5:
            score += 10

        # Distribution uniformity (0-20 points)
        if self.max_corners_per_region > 0:
            uniformity = self.min_corners_per_region / self.max_corners_per_region
            score += uniformity * 20

        # Empty regions penalty (0-20 points)
        if len(self.empty_regions) == 0:
            score += 20
        elif len(self.empty_regions) <= 2:
            score += 10

        return min(score, 100.0)

    @property
    def quality_label(self) -> str:
        """Human-readable quality label."""
        score = self.quality_score
        if score >= 80:
            return "Excellent"
        elif score >= 60:
            return "Good"
        elif score >= 40:
            return "Fair"
        else:
            return "Poor"


class CoverageAnalyzer:
    """Analyzes spatial coverage of detected corners across the image frame."""

    def __init__(self, grid_size: Tuple[int, int] = (10, 10)):
        """Initialize coverage analyzer.

        Args:
            grid_size: Grid dimensions for coverage analysis (rows, cols)
        """
        self.grid_size = grid_size

    def analyze(
        self,
        detections: List[ImageDetection],
        image_size: Tuple[int, int]
    ) -> CoverageReport:
        """Analyze spatial coverage of corners.

        Args:
            detections: List of image detections
            image_size: Image size (width, height)

        Returns:
            CoverageReport with analysis results
        """
        width, height = image_size
        grid_rows, grid_cols = self.grid_size

        # Initialize coverage map
        coverage_map = np.zeros((grid_rows, grid_cols), dtype=np.int32)

        # Get valid detections (not excluded, has corners)
        valid_detections = [
            d for d in detections
            if d.has_detection and not d.excluded and d.charuco_corners is not None
        ]

        total_corners = 0

        # Count corners in each grid cell
        for detection in valid_detections:
            for corner in detection.charuco_corners:
                x, y = corner[0]

                # Calculate grid cell
                col = int((x / width) * grid_cols)
                row = int((y / height) * grid_rows)

                # Clamp to valid range
                col = max(0, min(col, grid_cols - 1))
                row = max(0, min(row, grid_rows - 1))

                coverage_map[row, col] += 1
                total_corners += 1

        # Analyze coverage
        covered_cells = np.sum(coverage_map > 0)
        total_cells = grid_rows * grid_cols
        coverage_percentage = (covered_cells / total_cells) * 100

        # Find empty regions (divide into 9 regions: top-left, top-center, etc.)
        empty_regions = []
        region_names = [
            ['top-left', 'top-center', 'top-right'],
            ['middle-left', 'center', 'middle-right'],
            ['bottom-left', 'bottom-center', 'bottom-right']
        ]

        region_rows = grid_rows // 3
        region_cols = grid_cols // 3

        for region_row in range(3):
            for region_col in range(3):
                r_start = region_row * region_rows
                r_end = (region_row + 1) * region_rows if region_row < 2 else grid_rows
                c_start = region_col * region_cols
                c_end = (region_col + 1) * region_cols if region_col < 2 else grid_cols

                region_corners = np.sum(coverage_map[r_start:r_end, c_start:c_end])
                if region_corners == 0:
                    empty_regions.append(region_names[region_row][region_col])

        # Calculate min/max corners per region
        non_zero_cells = coverage_map[coverage_map > 0]
        min_corners = int(np.min(non_zero_cells)) if len(non_zero_cells) > 0 else 0
        max_corners = int(np.max(coverage_map))

        # Normalize coverage map for visualization (0-1)
        normalized_map = coverage_map.astype(np.float32)
        if max_corners > 0:
            normalized_map = normalized_map / max_corners

        # Generate warnings
        warnings = []

        if len(valid_detections) < 5:
            warnings.append(f"Too few images: {len(valid_detections)} (recommend 10+)")
        elif len(valid_detections) < 10:
            warnings.append(f"Limited images: {len(valid_detections)} (recommend 10+)")

        if coverage_percentage < 50:
            warnings.append(f"Low frame coverage: {coverage_percentage:.0f}% (recommend 70+%)")

        if len(empty_regions) > 0:
            warnings.append(f"Empty regions: {', '.join(empty_regions)}")

        if max_corners > 0 and min_corners / max_corners < 0.3:
            warnings.append(f"Uneven distribution: some areas have {max_corners}x more corners than others")

        # Determine if adequate
        is_adequate = (
            len(valid_detections) >= 10 and
            coverage_percentage >= 60 and
            len(empty_regions) <= 2
        )

        return CoverageReport(
            coverage_map=normalized_map,
            coverage_percentage=coverage_percentage,
            empty_regions=empty_regions,
            min_corners_per_region=min_corners,
            max_corners_per_region=max_corners,
            total_corners=total_corners,
            num_images_used=len(valid_detections),
            is_adequate=is_adequate,
            warnings=warnings,
        )

    def create_heatmap_visualization(
        self,
        coverage_map: np.ndarray,
        image_size: Tuple[int, int]
    ) -> np.ndarray:
        """Create a heat map visualization of coverage.

        Args:
            coverage_map: Normalized coverage map (0-1)
            image_size: Target image size (width, height)

        Returns:
            Heat map image (BGR)
        """
        # Resize coverage map to image size
        resized = cv2.resize(
            coverage_map,
            image_size,
            interpolation=cv2.INTER_LINEAR
        )

        # Apply color map (green = well covered, red = no coverage)
        # Invert so red = 0 (no coverage), green = 1 (high coverage)
        heatmap = cv2.applyColorMap(
            (resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )

        return heatmap

    def create_overlay_visualization(
        self,
        coverage_map: np.ndarray,
        base_image: np.ndarray,
        alpha: float = 0.4
    ) -> np.ndarray:
        """Create heat map overlay on base image.

        Args:
            coverage_map: Normalized coverage map (0-1)
            base_image: Base image to overlay on
            alpha: Overlay transparency (0-1)

        Returns:
            Image with heat map overlay (BGR)
        """
        h, w = base_image.shape[:2]

        # Create heatmap at image size
        heatmap = self.create_heatmap_visualization(coverage_map, (w, h))

        # Blend with base image
        overlay = cv2.addWeighted(base_image, 1 - alpha, heatmap, alpha, 0)

        # Add grid lines to show regions
        grid_rows, grid_cols = self.grid_size
        cell_height = h // grid_rows
        cell_width = w // grid_cols

        for i in range(1, grid_rows):
            y = i * cell_height
            cv2.line(overlay, (0, y), (w, y), (255, 255, 255), 1)

        for i in range(1, grid_cols):
            x = i * cell_width
            cv2.line(overlay, (x, 0), (x, h), (255, 255, 255), 1)

        # Add legend
        legend_y = 30
        cv2.rectangle(overlay, (10, 10), (300, legend_y + 100), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 10), (300, legend_y + 100), (255, 255, 255), 2)

        cv2.putText(overlay, "Corner Coverage Heat Map", (20, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        legend_y += 25
        cv2.putText(overlay, "Red = No corners detected", (20, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        legend_y += 20
        cv2.putText(overlay, "Yellow = Some corners", (20, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        legend_y += 20
        cv2.putText(overlay, "Green = Well covered", (20, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return overlay
