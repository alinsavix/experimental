"""Scrollable image grid widget."""

from typing import List, Optional
from PyQt6.QtWidgets import (
    QWidget, QScrollArea, QGridLayout, QLabel, QMenu,
    QVBoxLayout, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QAction
import cv2
import numpy as np

from ..core import ImageDetection, ImageQuality
from ..utils import ImageLoader


class ImageThumbnail(QFrame):
    """Thumbnail widget for a single image."""

    clicked = pyqtSignal(int)  # Emits image index when clicked
    contextMenuRequested = pyqtSignal(int)  # Emits index for context menu

    def __init__(self, index: int, image: np.ndarray, path: str, parent=None):
        """Initialize thumbnail.

        Args:
            index: Image index
            image: Full-size image (BGR)
            path: Image file path
            parent: Parent widget
        """
        super().__init__(parent)
        self.index = index
        self.image = image
        self.path = path
        self.quality = ImageQuality.NOT_CALIBRATED
        self.excluded = False
        self.reprojection_error = -1.0

        self.setFrameStyle(QFrame.Shape.Box)
        self.setLineWidth(3)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # Create layout
        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)

        # Thumbnail image
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._update_thumbnail()
        layout.addWidget(self.image_label)

        # Filename label (always shown)
        import os
        filename = os.path.basename(self.path)
        self.filename_label = QLabel(filename[:25])
        self.filename_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.filename_label.setStyleSheet("font-size: 9px; color: #666;")
        self.filename_label.setWordWrap(True)
        layout.addWidget(self.filename_label)

        # Status/error label (changes based on state)
        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("font-size: 10px;")
        self._update_status_label()
        layout.addWidget(self.status_label)

        self.setLayout(layout)
        self._update_border()

    def _update_thumbnail(self):
        """Update thumbnail image display."""
        # Create thumbnail
        thumbnail = ImageLoader.create_thumbnail(self.image, size=200)

        # Apply excluded overlay if needed
        if self.excluded:
            # Semi-transparent gray overlay
            overlay = thumbnail.copy()
            gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            thumbnail = cv2.addWeighted(thumbnail, 0.3, gray, 0.7, 0)

            # Draw red X
            h, w = thumbnail.shape[:2]
            cv2.line(thumbnail, (10, 10), (w-10, h-10), (0, 0, 255), 3)
            cv2.line(thumbnail, (w-10, 10), (10, h-10), (0, 0, 255), 3)

        # Convert to QPixmap
        # Ensure contiguous memory layout
        if not thumbnail.flags['C_CONTIGUOUS']:
            thumbnail = np.ascontiguousarray(thumbnail)

        h, w, ch = thumbnail.shape
        bytes_per_line = ch * w

        # Convert to bytes for PyQt6
        image_bytes = thumbnail.tobytes()
        qt_image = QImage(image_bytes, w, h, bytes_per_line, QImage.Format.Format_BGR888)

        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap)

    def _update_status_label(self):
        """Update status label text (reprojection error or status)."""
        if self.excluded:
            self.status_label.setText("EXCLUDED")
            self.status_label.setStyleSheet("font-size: 10px; color: red; font-weight: bold;")
        elif self.reprojection_error > 0:
            self.status_label.setText(f"{self.reprojection_error:.3f} px")
            self.status_label.setStyleSheet("font-size: 11px; font-weight: bold;")
        elif self.reprojection_error == -1.0:
            # Failed detection after calibration attempted
            self.status_label.setText("NO DETECTION")
            self.status_label.setStyleSheet("font-size: 10px; color: red; font-weight: bold;")
        else:
            # Not yet calibrated
            self.status_label.setText("Not calibrated")
            self.status_label.setStyleSheet("font-size: 9px; color: #999;")

    def _update_border(self):
        """Update border color based on quality."""
        if self.excluded:
            color = QColor(128, 128, 128)  # Gray
        else:
            r, g, b = self.quality.color
            color = QColor(b, g, r)  # BGR to RGB

        self.setStyleSheet(f"QFrame {{ border: 3px solid rgb({color.red()}, {color.green()}, {color.blue()}); }}")

    def update_quality(self, error: float, quality: ImageQuality):
        """Update quality indicators.

        Args:
            error: Reprojection error
            quality: Quality classification
        """
        self.reprojection_error = error
        self.quality = quality
        self._update_status_label()
        self._update_border()

    def mark_as_failed(self):
        """Mark this image as having failed detection (no corners or too few)."""
        self.reprojection_error = -1.0
        self.quality = ImageQuality.NOT_CALIBRATED
        self._update_status_label()
        # Use red border for failed detections
        self.setStyleSheet("QFrame { border: 3px solid rgb(200, 0, 0); background-color: rgba(200, 0, 0, 30); }")

    def set_excluded(self, excluded: bool):
        """Set exclusion status.

        Args:
            excluded: Whether image is excluded
        """
        self.excluded = excluded
        self._update_thumbnail()
        self._update_status_label()
        self._update_border()

    def mousePressEvent(self, event):
        """Handle mouse press."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.index)
        elif event.button() == Qt.MouseButton.RightButton:
            self.contextMenuRequested.emit(self.index)

    def sizeHint(self):
        """Suggested size for thumbnail."""
        return QSize(210, 260)  # Increased height for two labels


class ImageGrid(QWidget):
    """Scrollable grid of image thumbnails."""

    imageClicked = pyqtSignal(int)  # Emits image index
    exclusionChanged = pyqtSignal()  # Emits when exclusion status changes

    def __init__(self, parent=None):
        """Initialize image grid."""
        super().__init__(parent)
        self.thumbnails: List[ImageThumbnail] = []
        self.detections: List[ImageDetection] = []
        self._setup_ui()

    def _setup_ui(self):
        """Setup UI components."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Grid container
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout()
        self.grid_layout.setSpacing(10)
        self.grid_container.setLayout(self.grid_layout)

        self.scroll_area.setWidget(self.grid_container)
        layout.addWidget(self.scroll_area)

        self.setLayout(layout)

    def load_images(self, images: List[tuple]):
        """Load images into grid.

        Args:
            images: List of (path, image) tuples
        """
        # Clear existing
        self.clear()

        # Create thumbnails
        columns = 4
        for i, (path, image) in enumerate(images):
            thumbnail = ImageThumbnail(i, image, path)
            thumbnail.clicked.connect(self.imageClicked.emit)
            thumbnail.contextMenuRequested.connect(self._show_context_menu)

            row = i // columns
            col = i % columns
            self.grid_layout.addWidget(thumbnail, row, col)

            self.thumbnails.append(thumbnail)

    def set_detections(self, detections: List[ImageDetection]):
        """Set detection results for quality display.

        Args:
            detections: List of ImageDetection objects
        """
        self.detections = detections

        for i, detection in enumerate(detections):
            if i < len(self.thumbnails):
                if detection.reprojection_error > 0:
                    quality = ImageQuality.from_error(detection.reprojection_error)
                    self.thumbnails[i].update_quality(detection.reprojection_error, quality)
                elif not detection.has_detection or (detection.charuco_corners is not None and len(detection.charuco_corners) < 4):
                    # Mark as failed if no detection or too few corners
                    self.thumbnails[i].mark_as_failed()

                if detection.excluded:
                    self.thumbnails[i].set_excluded(True)

    def _show_context_menu(self, index: int):
        """Show context menu for image.

        Args:
            index: Image index
        """
        if index >= len(self.detections):
            return

        detection = self.detections[index]
        thumbnail = self.thumbnails[index]

        menu = QMenu(self)

        # Exclude/Include action
        if detection.excluded:
            exclude_action = QAction("Include in calibration", self)
            exclude_action.triggered.connect(lambda: self._toggle_exclusion(index, False))
        else:
            exclude_action = QAction("Exclude from calibration", self)
            exclude_action.triggered.connect(lambda: self._toggle_exclusion(index, True))
        menu.addAction(exclude_action)

        # Show detection action
        if detection.has_detection:
            view_action = QAction("Show detection", self)
            view_action.triggered.connect(lambda: self.imageClicked.emit(index))
            menu.addAction(view_action)

        menu.exec(thumbnail.mapToGlobal(thumbnail.rect().center()))

    def _toggle_exclusion(self, index: int, excluded: bool):
        """Toggle image exclusion status.

        Args:
            index: Image index
            excluded: New exclusion status
        """
        if index < len(self.detections):
            self.detections[index].excluded = excluded
            self.thumbnails[index].set_excluded(excluded)
            self.exclusionChanged.emit()

    def get_excluded_count(self) -> int:
        """Get number of excluded images.

        Returns:
            Count of excluded images
        """
        return sum(1 for d in self.detections if d.excluded)

    def clear(self):
        """Clear all thumbnails."""
        for thumbnail in self.thumbnails:
            self.grid_layout.removeWidget(thumbnail)
            thumbnail.deleteLater()

        self.thumbnails.clear()
        self.detections.clear()
