"""Coverage heat map visualization dialog."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage, QPainter
import numpy as np

from ..core import CalibrationResult
from ..core.coverage_analyzer import CoverageAnalyzer


class CoverageHeatmapDialog(QDialog):
    """Dialog for displaying spatial coverage heat map."""

    def __init__(self, result: CalibrationResult, parent=None):
        """Initialize coverage heat map dialog.

        Args:
            result: Calibration result with coverage report
            parent: Parent widget
        """
        super().__init__(parent)
        self.result = result
        self.coverage_report = result.coverage_report

        self.setWindowTitle("Spatial Coverage Heat Map")
        self.setMinimumSize(900, 700)

        self._setup_ui()

    def _setup_ui(self):
        """Setup UI components."""
        layout = QVBoxLayout()

        # Title and explanation
        title_label = QLabel("<h2>Corner Coverage Analysis</h2>")
        layout.addWidget(title_label)

        desc_label = QLabel(
            "This heat map shows which parts of the camera frame have been sampled by detected corners. "
            "Good calibration requires corners distributed across the <b>entire frame</b>, especially edges and corners."
        )
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        # Coverage score
        if self.coverage_report:
            score = self.coverage_report.quality_score
            quality = self.coverage_report.quality_label

            if score >= 80:
                color = "green"
            elif score >= 60:
                color = "orange"
            else:
                color = "red"

            score_html = f'<div style="background-color: #f0f0f0; padding: 10px; border-left: 4px solid {color};">' \
                        f'<b>Coverage Quality: <span style="color: {color}; font-size: 16px;">{quality} ({score:.0f}/100)</span></b><br>' \
                        f'Frame coverage: {self.coverage_report.coverage_percentage:.0f}% | ' \
                        f'Images used: {self.coverage_report.num_images_used} | ' \
                        f'Total corners: {self.coverage_report.total_corners}' \
                        f'</div>'
            score_label = QLabel(score_html)
            layout.addWidget(score_label)

            # Warnings
            if self.coverage_report.warnings:
                warnings_html = '<div style="background-color: #fff3cd; padding: 10px; border-left: 4px solid orange;">' \
                               '<b>⚠ Coverage Issues:</b><ul>'
                for warning in self.coverage_report.warnings:
                    warnings_html += f'<li>{warning}</li>'
                warnings_html += '</ul></div>'

                # Add recommendations
                warnings_html += '<br><b>Recommendations:</b><ul>'
                if self.coverage_report.num_images_used < 10:
                    warnings_html += '<li>Capture more images (10-20 recommended)</li>'
                if self.coverage_report.empty_regions:
                    warnings_html += f'<li>Capture images with board in: {", ".join(self.coverage_report.empty_regions)}</li>'
                if self.coverage_report.coverage_percentage < 70:
                    warnings_html += '<li>Move board to different positions and angles</li>'
                    warnings_html += '<li>Include images with board near frame edges</li>'
                warnings_html += '</ul>'

                warnings_label = QLabel(warnings_html)
                warnings_label.setWordWrap(True)
                layout.addWidget(warnings_label)

        # Heat map visualization
        heatmap_image = self._create_heatmap_visualization()

        view = QGraphicsView()
        scene = QGraphicsScene()
        view.setScene(scene)

        # Convert to QPixmap
        if not heatmap_image.flags['C_CONTIGUOUS']:
            heatmap_image = np.ascontiguousarray(heatmap_image)

        h, w, ch = heatmap_image.shape
        bytes_per_line = ch * w
        image_bytes = heatmap_image.tobytes()
        qt_image = QImage(image_bytes, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        qt_image._image_bytes = image_bytes

        pixmap = QPixmap.fromImage(qt_image)
        pixmap_item = QGraphicsPixmapItem(pixmap)
        scene.addItem(pixmap_item)
        scene.setSceneRect(pixmap_item.boundingRect())

        view.fitInView(pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        layout.addWidget(view)

        # Close button
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _create_heatmap_visualization(self) -> np.ndarray:
        """Create heat map visualization.

        Returns:
            Heat map image with overlay
        """
        # Get a representative image (first valid detection)
        base_image = None
        for detection in self.result.detections:
            if detection.has_detection and not detection.excluded:
                base_image = detection.image
                break

        if base_image is None:
            # Use first image if no valid detections
            base_image = self.result.detections[0].image if self.result.detections else np.zeros((480, 640, 3), dtype=np.uint8)

        # Create overlay
        analyzer = CoverageAnalyzer()
        overlay = analyzer.create_overlay_visualization(
            self.coverage_report.coverage_map,
            base_image,
            alpha=0.5
        )

        return overlay
