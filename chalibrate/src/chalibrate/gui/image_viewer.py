"""Full-size image viewer with detection overlay."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTabWidget,
    QWidget, QPushButton, QSlider, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage, QWheelEvent, QTransform, QPainter
import cv2
import numpy as np

from ..core import ImageDetection, CalibrationResult, Calibrator


class ZoomableImageView(QGraphicsView):
    """High-performance image view with smooth zooming and panning."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = None
        self.zoom_factor = 1.0

        # Enable smooth scrolling and rendering
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Optimize for performance
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def set_image(self, pixmap: QPixmap):
        """Set the image to display.

        Args:
            pixmap: Image pixmap
        """
        self.scene.clear()
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)
        self.scene.setSceneRect(self.pixmap_item.boundingRect())
        self.zoom_factor = 1.0
        self.resetTransform()

    def set_zoom(self, zoom_factor: float):
        """Set the zoom factor.

        Args:
            zoom_factor: Zoom level (1.0 = 100%, 2.0 = 200%, etc.)
        """
        if self.pixmap_item is None:
            return

        zoom_factor = max(0.1, min(zoom_factor, 10.0))

        # Calculate scale change
        scale_change = zoom_factor / self.zoom_factor
        self.zoom_factor = zoom_factor

        # Apply scale
        self.scale(scale_change, scale_change)

    def fit_in_view(self):
        """Fit image to view."""
        if self.pixmap_item is None:
            return

        self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        # Calculate actual zoom factor after fit
        transform = self.transform()
        self.zoom_factor = transform.m11()  # Get scale factor from transform matrix

    def reset_zoom(self):
        """Reset to 100% zoom."""
        self.resetTransform()
        self.zoom_factor = 1.0

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming.

        Args:
            event: Wheel event
        """
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Zoom with Ctrl+Wheel
            delta = event.angleDelta().y()
            if delta > 0:
                scale_factor = 1.1
            else:
                scale_factor = 0.9

            new_zoom = self.zoom_factor * scale_factor
            new_zoom = max(0.1, min(new_zoom, 10.0))

            if new_zoom != self.zoom_factor:
                scale_change = new_zoom / self.zoom_factor
                self.scale(scale_change, scale_change)
                self.zoom_factor = new_zoom

            event.accept()
        else:
            # Normal scroll
            super().wheelEvent(event)


class ImageViewer(QDialog):
    """Dialog for viewing full-size images with detection overlay."""

    def __init__(
        self,
        detection: ImageDetection,
        result: CalibrationResult,
        board_config,
        calibration_options=None,
        parent=None
    ):
        """Initialize image viewer.

        Args:
            detection: Image detection to display
            result: Full calibration result (for undistortion)
            board_config: Board configuration used for calibration
            calibration_options: Calibration options (for CLAHE preview)
            parent: Parent widget
        """
        super().__init__(parent)
        self.detection = detection
        self.result = result
        self.board_config = board_config
        self.calibration_options = calibration_options
        self.last_tab_index = 0  # Track previous tab for position sync

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
        # Only update zoom when slider is released (not during drag)
        self.zoom_slider.sliderReleased.connect(self._on_zoom_slider_released)
        # Update label while dragging (lightweight)
        self.zoom_slider.valueChanged.connect(self._on_zoom_slider_value_changed)
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

        # CLAHE preprocessed tab (if enabled)
        if self.calibration_options and self.calibration_options.enable_clahe:
            clahe_image = self._create_clahe_image()
            clahe_widget = self._create_image_widget(clahe_image)
            self.tabs.addTab(clahe_widget, "CLAHE Processed")

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

        # Coverage heat map tab
        if self.result and self.result.coverage_report:
            coverage_viz = self._create_coverage_visualization()
            coverage_widget = self._create_image_widget(coverage_viz)
            self.tabs.addTab(coverage_widget, "Coverage Heat Map")

        # Connect tab change to synchronize zoom/pan
        self.tabs.currentChanged.connect(self._on_tab_changed)

        layout.addWidget(self.tabs)

        # Close button
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Store references to image views for zoom control
        self.image_views = []
        for i in range(self.tabs.count()):
            view = self.tabs.widget(i)
            if isinstance(view, ZoomableImageView):
                self.image_views.append(view)

        # Default to "fit to window" for first view
        if self.image_views:
            self.image_views[0].fit_in_view()
            # Update slider to match fitted zoom
            self.zoom_slider.setValue(int(self.image_views[0].zoom_factor * 100))
            self.zoom_label.setText(f"{int(self.image_views[0].zoom_factor * 100)}%")

    def _on_tab_changed(self, index: int):
        """Handle tab change - sync zoom and pan position from previous tab.

        Args:
            index: New tab index
        """
        if index < 0 or not self.image_views:
            return

        # Get the previous tab's view (the one we're switching FROM)
        previous_widget = self.tabs.widget(self.last_tab_index)
        source_view = previous_widget if isinstance(previous_widget, ZoomableImageView) else None

        # Get the new tab's view (the one we're switching TO)
        target_view = self.tabs.widget(index)

        if not isinstance(target_view, ZoomableImageView):
            return

        # If we have a source view, copy its state
        if source_view:
            # Get the center point in scene coordinates from the source
            source_center = source_view.mapToScene(source_view.viewport().rect().center())

            # Sync zoom level
            if abs(target_view.zoom_factor - source_view.zoom_factor) > 0.01:
                target_view.set_zoom(source_view.zoom_factor)

            # Sync scroll position
            target_view.centerOn(source_center)

        # Update last tab index for next switch
        self.last_tab_index = index

    def _change_zoom(self, delta: float):
        """Change zoom by a relative amount.

        Args:
            delta: Change in zoom factor
        """
        # Get current zoom from the active view
        current_view = self.tabs.currentWidget()
        if isinstance(current_view, ZoomableImageView):
            current_zoom = current_view.zoom_factor
        else:
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
        self.zoom_label.setText(f"{int(zoom * 100)}%")

        # Get current view's center point before zooming
        current_view = self.tabs.currentWidget()
        if isinstance(current_view, ZoomableImageView):
            center_point = current_view.mapToScene(current_view.viewport().rect().center())
        else:
            center_point = None

        # Apply zoom immediately when set programmatically (not from drag)
        for view in self.image_views:
            view.set_zoom(zoom)
            # Sync center point
            if center_point:
                view.centerOn(center_point)

    def _on_zoom_slider_value_changed(self, value: int):
        """Update zoom label while slider is being dragged (lightweight).

        Args:
            value: Slider value (10-1000, representing 10%-1000%)
        """
        self.zoom_label.setText(f"{value}%")

    def _on_zoom_slider_released(self):
        """Apply zoom when slider is released (prevents lag during drag)."""
        value = self.zoom_slider.value()
        zoom_factor = value / 100.0

        # Get current view's center point before zooming
        current_view = self.tabs.currentWidget()
        if isinstance(current_view, ZoomableImageView):
            center_point = current_view.mapToScene(current_view.viewport().rect().center())
        else:
            center_point = None

        # Apply zoom to all image views
        for view in self.image_views:
            view.set_zoom(zoom_factor)
            # Sync center point
            if center_point:
                view.centerOn(center_point)

    def _fit_to_window(self):
        """Fit image to the current window size."""
        # Fit all views (they should have similar aspect ratios)
        for view in self.image_views:
            view.fit_in_view()

        # Update slider to match first view
        if self.image_views:
            zoom = self.image_views[0].zoom_factor
            self.zoom_slider.setValue(int(zoom * 100))
            self.zoom_label.setText(f"{int(zoom * 100)}%")

    def _create_image_widget(self, image: np.ndarray) -> QWidget:
        """Create widget for displaying image with zoom support.

        Args:
            image: Image to display (BGR format)

        Returns:
            Widget containing zoomable image view
        """
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

        # Create graphics view
        view = ZoomableImageView()
        view.set_image(pixmap)

        return view

    def _create_clahe_image(self) -> np.ndarray:
        """Create CLAHE-processed version of image.

        Returns:
            CLAHE-processed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(self.detection.image, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=self.calibration_options.clahe_clip_limit,
            tileGridSize=self.calibration_options.clahe_tile_size
        )
        clahe_gray = clahe.apply(gray)

        # Convert back to BGR for display
        clahe_bgr = cv2.cvtColor(clahe_gray, cv2.COLOR_GRAY2BGR)

        return clahe_bgr

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

    def _create_coverage_visualization(self) -> np.ndarray:
        """Create coverage heat map overlay on this image.

        Returns:
            Image with coverage heat map overlay
        """
        from ..core.coverage_analyzer import CoverageAnalyzer

        analyzer = CoverageAnalyzer()
        overlay = analyzer.create_overlay_visualization(
            self.result.coverage_report.coverage_map,
            self.detection.image,
            alpha=0.5
        )

        return overlay
