"""Undistortion preview dialog."""

from typing import List
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QScrollArea, QWidget, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage
import cv2
import numpy as np
from pathlib import Path

from ..core import CalibrationResult, ImageDetection


class UndistortionDialog(QDialog):
    """Dialog for previewing undistorted images."""

    def __init__(self, result: CalibrationResult, parent=None):
        """Initialize undistortion dialog.

        Args:
            result: Calibration result
            parent: Parent widget
        """
        super().__init__(parent)
        self.result = result
        self.current_index = 0

        self.setWindowTitle("Undistortion Preview")
        self.setMinimumSize(1000, 600)

        self._setup_ui()
        self._update_display()

    def _setup_ui(self):
        """Setup UI components."""
        layout = QVBoxLayout()

        # Image selector
        selector_layout = QHBoxLayout()

        selector_layout.addWidget(QLabel("Select Image:"))

        self.image_combo = QComboBox()
        for i, detection in enumerate(self.result.detections):
            filename = Path(detection.image_path).name
            self.image_combo.addItem(f"{i+1}. {filename}", i)

        self.image_combo.currentIndexChanged.connect(self._on_image_changed)
        selector_layout.addWidget(self.image_combo)

        selector_layout.addStretch()

        layout.addLayout(selector_layout)

        # Split view container
        split_layout = QHBoxLayout()

        # Original image
        original_group = QWidget()
        original_layout = QVBoxLayout()
        original_layout.addWidget(QLabel("<b>Original</b>"))

        self.original_scroll = QScrollArea()
        self.original_scroll.setWidgetResizable(False)
        self.original_label = QLabel()
        self.original_scroll.setWidget(self.original_label)

        original_layout.addWidget(self.original_scroll)
        original_group.setLayout(original_layout)
        split_layout.addWidget(original_group)

        # Undistorted image
        undistorted_group = QWidget()
        undistorted_layout = QVBoxLayout()
        undistorted_layout.addWidget(QLabel("<b>Undistorted</b>"))

        self.undistorted_scroll = QScrollArea()
        self.undistorted_scroll.setWidgetResizable(False)
        self.undistorted_label = QLabel()
        self.undistorted_scroll.setWidget(self.undistorted_label)

        undistorted_layout.addWidget(self.undistorted_scroll)
        undistorted_group.setLayout(undistorted_layout)
        split_layout.addWidget(undistorted_group)

        layout.addLayout(split_layout)

        # Action buttons
        button_layout = QHBoxLayout()

        export_all_btn = QPushButton("Export All Undistorted...")
        export_all_btn.clicked.connect(self._export_all_undistorted)
        button_layout.addWidget(export_all_btn)

        button_layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _on_image_changed(self, index: int):
        """Handle image selection change.

        Args:
            index: Combo box index
        """
        self.current_index = self.image_combo.itemData(index)
        self._update_display()

    def _update_display(self):
        """Update image display."""
        if self.current_index >= len(self.result.detections):
            return

        detection = self.result.detections[self.current_index]

        # Display original
        original_pixmap = self._image_to_pixmap(detection.image)
        self.original_label.setPixmap(original_pixmap)
        self.original_label.setScaledContents(False)
        self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Display undistorted
        undistorted = self._undistort_image(detection.image)
        undistorted_pixmap = self._image_to_pixmap(undistorted)
        self.undistorted_label.setPixmap(undistorted_pixmap)
        self.undistorted_label.setScaledContents(False)
        self.undistorted_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def _image_to_pixmap(self, image: np.ndarray) -> QPixmap:
        """Convert OpenCV image to QPixmap.

        Args:
            image: Image in BGR format

        Returns:
            QPixmap
        """
        # Ensure contiguous memory layout
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)

        h, w, ch = image.shape
        bytes_per_line = ch * w

        # Convert to bytes for PyQt6
        image_bytes = image.tobytes()
        qt_image = QImage(image_bytes, w, h, bytes_per_line, QImage.Format.Format_BGR888)

        return QPixmap.fromImage(qt_image)

    def _undistort_image(self, image: np.ndarray) -> np.ndarray:
        """Undistort an image.

        Args:
            image: Original image

        Returns:
            Undistorted image
        """
        h, w = image.shape[:2]

        # Get optimal new camera matrix
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.result.camera_matrix,
            self.result.dist_coeffs,
            (w, h),
            1,
            (w, h)
        )

        # Undistort
        undistorted = cv2.undistort(
            image,
            self.result.camera_matrix,
            self.result.dist_coeffs,
            None,
            new_camera_matrix
        )

        # Crop to valid region
        if roi != (0, 0, 0, 0):
            x, y, w, h = roi
            undistorted = undistorted[y:y+h, x:x+w]

        return undistorted

    def _export_all_undistorted(self):
        """Export all undistorted images to directory."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory for Undistorted Images",
            "",
            QFileDialog.Option.ShowDirsOnly
        )

        if not directory:
            return

        try:
            output_dir = Path(directory)
            success_count = 0

            for i, detection in enumerate(self.result.detections):
                # Generate output filename
                input_path = Path(detection.image_path)
                output_filename = f"{input_path.stem}_undistorted{input_path.suffix}"
                output_path = output_dir / output_filename

                # Undistort and save
                undistorted = self._undistort_image(detection.image)
                cv2.imwrite(str(output_path), undistorted)
                success_count += 1

            QMessageBox.information(
                self,
                "Export Complete",
                f"Successfully exported {success_count} undistorted images to:\n{directory}"
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Failed to export undistorted images:\n{str(e)}"
            )
