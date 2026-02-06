"""Full-size image viewer with detection overlay."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTabWidget,
    QWidget, QScrollArea, QPushButton, QSlider
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage, QWheelEvent
import cv2
import numpy as np

from ..core import ImageDetection, CalibrationResult, Calibrator


class ZoomableImageLabel(QLabel):
    """Label that displays an image and supports zooming."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_pixmap = None
        self.zoom_factor = 1.0
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def set_image(self, pixmap: QPixmap):
        """Set the image to display.

        Args:
            pixmap: Original image pixmap
        """
        self.original_pixmap = pixmap
        self.update_display()

    def set_zoom(self, zoom_factor: float):
        """Set the zoom factor.

        Args:
            zoom_factor: Zoom level (1.0 = 100%, 2.0 = 200%, etc.)
        """
        self.zoom_factor = max(0.1, min(zoom_factor, 10.0))
        self.update_display()

    def update_display(self):
        """Update the displayed image based on current zoom."""
        if self.original_pixmap is None:
            return

        if abs(self.zoom_factor - 1.0) < 0.01:
            # No zoom, show original
            self.setPixmap(self.original_pixmap)
        else:
            # Scale the pixmap
            scaled_size = self.original_pixmap.size() * self.zoom_factor
            scaled_pixmap = self.original_pixmap.scaled(
                scaled_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.setPixmap(scaled_pixmap)


class ImageViewer(QDialog):
    """Dialog for viewing full-size images with detection overlay."""

    def __init__(
        self,
        detection: ImageDetection,
        result: CalibrationResult,
        board_config,
        parent=None
    ):
        """Initialize image viewer.

        Args:
            detection: Image detection to display
            result: Full calibration result (for undistortion)
            board_config: Board configuration used for calibration
            parent: Parent widget
        """
        super().__init__(parent)
        self.detection = detection
        self.result = result
        self.board_config = board_config

        self.setWindowTitle(f"Image Viewer - {detection.image_path}")
        self.setMinimumSize(800, 600)

        self._setup_ui()

    def _setup_ui(self):
        """Setup UI components."""
        layout = QVBoxLayout()

        # Info label
        info_text = f"<b>File:</b> {self.detection.image_path}<br>"
        if self.detection.has_detection:
            info_text += f"<b>Corners detected:</b> {self.detection.num_corners}<br>"
        if self.detection.reprojection_error > 0:
            info_text += f"<b>Reprojection error:</b> {self.detection.reprojection_error:.4f} pixels"

        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Zoom controls
        zoom_layout = QHBoxLayout()

        zoom_out_btn = QPushButton("−")
        zoom_out_btn.setFixedWidth(40)
        zoom_out_btn.setToolTip("Zoom out (Ctrl+−)")
        zoom_out_btn.clicked.connect(lambda: self._change_zoom(-0.1))
        zoom_layout.addWidget(zoom_out_btn)

        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setMinimum(10)  # 10% = 0.1x
        self.zoom_slider.setMaximum(1000)  # 1000% = 10x
        self.zoom_slider.setValue(100)  # 100% = 1.0x
        self.zoom_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.zoom_slider.setTickInterval(100)
        self.zoom_slider.valueChanged.connect(self._on_zoom_slider_changed)
        zoom_layout.addWidget(self.zoom_slider)

        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setFixedWidth(40)
        zoom_in_btn.setToolTip("Zoom in (Ctrl++)")
        zoom_in_btn.clicked.connect(lambda: self._change_zoom(0.1))
        zoom_layout.addWidget(zoom_in_btn)

        fit_btn = QPushButton("Fit")
        fit_btn.setToolTip("Fit to window")
        fit_btn.clicked.connect(self._fit_to_window)
        zoom_layout.addWidget(fit_btn)

        actual_btn = QPushButton("100%")
        actual_btn.setToolTip("Actual size")
        actual_btn.clicked.connect(lambda: self._set_zoom(1.0))
        zoom_layout.addWidget(actual_btn)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(60)
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        zoom_layout.addWidget(self.zoom_label)

        layout.addLayout(zoom_layout)

        # Tab widget for different views
        self.tabs = QTabWidget()

        # Original image tab
        original_widget = self._create_image_widget(self.detection.image)
        self.tabs.addTab(original_widget, "Original")

        # Detection overlay tab
        if self.detection.has_detection:
            detection_image = self._draw_detection()
            detection_widget = self._create_image_widget(detection_image)
            self.tabs.addTab(detection_widget, "Detection")

        # Undistorted image tab
        if self.result:
            undistorted = self._create_undistorted_image()
            undistorted_widget = self._create_image_widget(undistorted)
            self.tabs.addTab(undistorted_widget, "Undistorted")

        # Reprojection error visualization tab
        if self.detection.has_detection and self.result and self.detection.reprojection_error > 0:
            error_viz = self._create_error_visualization()
            error_widget = self._create_image_widget(error_viz)
            self.tabs.addTab(error_widget, "Reprojection Error")

        layout.addWidget(self.tabs)

        # Close button
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Store references to image labels for zoom control
        self.image_labels = []
        for i in range(self.tabs.count()):
            tab_widget = self.tabs.widget(i)
            scroll_area = tab_widget.findChild(QScrollArea)
            if scroll_area:
                label = scroll_area.widget()
                if isinstance(label, ZoomableImageLabel):
                    self.image_labels.append(label)

    def _change_zoom(self, delta: float):
        """Change zoom by a relative amount.

        Args:
            delta: Change in zoom factor
        """
        current_zoom = self.zoom_slider.value() / 100.0
        new_zoom = current_zoom + delta
        self._set_zoom(new_zoom)

    def _set_zoom(self, zoom: float):
        """Set absolute zoom level.

        Args:
            zoom: Zoom factor (1.0 = 100%)
        """
        zoom = max(0.1, min(zoom, 10.0))
        self.zoom_slider.setValue(int(zoom * 100))

    def _on_zoom_slider_changed(self, value: int):
        """Handle zoom slider changes.

        Args:
            value: Slider value (10-1000, representing 10%-1000%)
        """
        zoom_factor = value / 100.0
        self.zoom_label.setText(f"{value}%")

        # Apply zoom to all image labels
        for label in self.image_labels:
            label.set_zoom(zoom_factor)

    def _fit_to_window(self):
        """Fit image to the current window size."""
        # Get the current tab's scroll area
        current_widget = self.tabs.currentWidget()
        scroll_area = current_widget.findChild(QScrollArea)
        if not scroll_area:
            return

        label = scroll_area.widget()
        if not isinstance(label, ZoomableImageLabel) or label.original_pixmap is None:
            return

        # Calculate zoom to fit
        viewport_size = scroll_area.viewport().size()
        pixmap_size = label.original_pixmap.size()

        width_ratio = viewport_size.width() / pixmap_size.width()
        height_ratio = viewport_size.height() / pixmap_size.height()
        fit_zoom = min(width_ratio, height_ratio) * 0.95  # 95% to leave some margin

        self._set_zoom(fit_zoom)

    def _create_image_widget(self, image: np.ndarray) -> QWidget:
        """Create scrollable widget for displaying image with zoom support.

        Args:
            image: Image to display (BGR format)

        Returns:
            Widget containing zoomable image
        """
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(False)

        # Use zoomable label
        image_label = ZoomableImageLabel()

        # Convert to QPixmap
        # Ensure the image is contiguous in memory
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)

        h, w, ch = image.shape
        bytes_per_line = ch * w

        # Convert numpy array to bytes for PyQt6
        image_bytes = image.tobytes()
        qt_image = QImage(image_bytes, w, h, bytes_per_line, QImage.Format.Format_BGR888)

        # Keep a reference to prevent garbage collection
        qt_image._image_bytes = image_bytes

        pixmap = QPixmap.fromImage(qt_image)

        image_label.set_image(pixmap)
        scroll_area.setWidget(image_label)

        # Enable mouse wheel zoom
        scroll_area.wheelEvent = lambda event: self._on_wheel_event(event, scroll_area)

        layout.addWidget(scroll_area)
        widget.setLayout(layout)

        return widget

    def _on_wheel_event(self, event: QWheelEvent, scroll_area: QScrollArea):
        """Handle mouse wheel events for zooming.

        Args:
            event: Wheel event
            scroll_area: The scroll area being scrolled
        """
        # Check if Ctrl is pressed
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Zoom with Ctrl+Wheel
            delta = event.angleDelta().y()
            if delta > 0:
                self._change_zoom(0.1)
            else:
                self._change_zoom(-0.1)
            event.accept()
        else:
            # Normal scroll
            QScrollArea.wheelEvent(scroll_area, event)

    def _draw_detection(self) -> np.ndarray:
        """Draw detection overlay on image.

        Returns:
            Image with detection overlay
        """
        output = self.detection.image.copy()

        # Draw ArUco markers
        if self.detection.marker_corners is not None and self.detection.marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(output, self.detection.marker_corners, self.detection.marker_ids)

        # Draw ChArUco corners
        if self.detection.has_detection:
            for i, corner in enumerate(self.detection.charuco_corners):
                corner_id = self.detection.charuco_ids[i][0]
                x, y = int(corner[0][0]), int(corner[0][1])

                # Draw circle at corner
                cv2.circle(output, (x, y), 8, (0, 255, 0), -1)

                # Draw corner ID
                cv2.putText(
                    output, str(corner_id), (x + 10, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2
                )

        return output

    def _create_undistorted_image(self) -> np.ndarray:
        """Create undistorted version of image.

        Returns:
            Undistorted image
        """
        h, w = self.detection.image.shape[:2]

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
            self.detection.image,
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

    def _create_error_visualization(self) -> np.ndarray:
        """Create visualization of reprojection errors.

        Returns:
            Image with reprojection error visualization
        """
        output = self.detection.image.copy()

        if not self.detection.has_detection or not self.result or not self.board_config:
            return output

        try:
            # Use the actual board configuration
            board = self.board_config.create_board()

            # Get object points for detected corners
            all_obj_points = board.getChessboardCorners()

            board_obj_points = []
            image_points = []

            if self.detection.charuco_ids is not None:
                for i, corner_id in enumerate(self.detection.charuco_ids):
                    corner_id_int = int(corner_id[0])
                    if corner_id_int < len(all_obj_points):
                        board_obj_points.append(all_obj_points[corner_id_int])
                        image_points.append(self.detection.charuco_corners[i])

                if len(board_obj_points) >= 4:
                    # Solve for pose
                    obj_pts = np.array(board_obj_points, dtype=np.float32)
                    img_pts = np.array(image_points, dtype=np.float32)

                    success, rvec, tvec = cv2.solvePnP(
                        obj_pts,
                        img_pts,
                        self.result.camera_matrix,
                        self.result.dist_coeffs
                    )

                    if success:
                        # Project the points back
                        projected_points, _ = cv2.projectPoints(
                            obj_pts,
                            rvec,
                            tvec,
                            self.result.camera_matrix,
                            self.result.dist_coeffs
                        )

                        # Draw the errors as arrows
                        max_error = 0
                        errors = []
                        for i, (detected, projected) in enumerate(zip(img_pts, projected_points)):
                            detected_pt = tuple(detected[0].astype(int))
                            projected_pt = tuple(projected[0].astype(int))

                            error = np.linalg.norm(detected[0] - projected[0])
                            errors.append(error)
                            max_error = max(max_error, error)

                        # Scale line thickness based on image resolution
                        line_thickness = max(2, int(output.shape[1] / 2000))

                        # Draw errors with color mapping
                        if max_error > 0:
                            for i, (detected, projected) in enumerate(zip(img_pts, projected_points)):
                                detected_pt = tuple(detected[0].astype(int))
                                projected_pt = tuple(projected[0].astype(int))

                                error = errors[i]
                                # Color from green (low error) to red (high error)
                                ratio = error / max_error
                                color = (0, int(255 * (1 - ratio)), int(255 * ratio))  # BGR

                                # Draw arrow from projected to detected
                                # Only draw if error is visible (> 0.5 pixels)
                                if error > 0.5:
                                    cv2.arrowedLine(output, projected_pt, detected_pt, color,
                                                   line_thickness, tipLength=0.2)

                                # Draw detected point (green)
                                cv2.circle(output, detected_pt, line_thickness * 3, (0, 255, 0), -1)

                                # Draw projected point (magenta) - slightly smaller
                                cv2.circle(output, projected_pt, line_thickness * 2, (255, 0, 255), -1)

                                # Show error value for significant errors (> 1.0 pixels)
                                if error > 1.0:
                                    cv2.putText(output, f"{error:.1f}",
                                              (detected_pt[0] + 15, detected_pt[1] + 15),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                        # Add legend
                        legend_y = 30
                        cv2.putText(output, "Reprojection Error Visualization", (10, legend_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                        legend_y += 35
                        cv2.putText(output, f"Max error: {max_error:.3f} px", (10, legend_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        legend_y += 30
                        cv2.putText(output, f"Mean error: {np.mean(errors):.3f} px", (10, legend_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        legend_y += 35
                        cv2.circle(output, (20, legend_y), 6, (0, 255, 0), -1)
                        cv2.putText(output, "Detected corner", (35, legend_y + 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        legend_y += 25
                        cv2.circle(output, (20, legend_y), 4, (255, 0, 255), -1)
                        cv2.putText(output, "Reprojected corner", (35, legend_y + 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        legend_y += 25
                        cv2.arrowedLine(output, (20, legend_y), (50, legend_y), (0, 128, 255), 2, tipLength=0.3)
                        cv2.putText(output, "Error vector (green=low, red=high)", (60, legend_y + 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        except Exception as e:
            # If we can't compute the visualization, just show the error
            cv2.putText(output, f"Could not compute error visualization: {str(e)}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return output
