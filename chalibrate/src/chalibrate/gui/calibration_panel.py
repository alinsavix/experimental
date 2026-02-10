"""Calibration results and control panel."""

from typing import Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QGroupBox, QGridLayout, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt
import json
import numpy as np

from ..core import CalibrationResult, QualityMetrics


class CalibrationPanel(QWidget):
    """Panel displaying calibration progress and results."""

    def __init__(self, parent=None):
        """Initialize calibration panel."""
        super().__init__(parent)
        self.result: Optional[CalibrationResult] = None
        self._setup_ui()

    def _setup_ui(self):
        """Setup UI components."""
        layout = QVBoxLayout()

        # Progress section
        self.progress_group = QGroupBox("Calibration Progress")
        progress_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        progress_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready to calibrate")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self.status_label)

        # Advanced options button
        self.advanced_options_btn = QPushButton("⚙️ Advanced Options...")
        self.advanced_options_btn.setToolTip("Configure accuracy features like subpixel refinement and quality filtering")
        progress_layout.addWidget(self.advanced_options_btn)

        self.progress_group.setLayout(progress_layout)
        layout.addWidget(self.progress_group)

        # Results section
        self.results_group = QGroupBox("Calibration Results")
        results_layout = QVBoxLayout()

        # Camera matrix
        camera_group = QGroupBox("Camera Intrinsics")
        camera_layout = QGridLayout()

        self.fx_label = QLabel("--")
        self.fy_label = QLabel("--")
        self.cx_label = QLabel("--")
        self.cy_label = QLabel("--")

        camera_layout.addWidget(QLabel("Focal Length X (fx):"), 0, 0)
        camera_layout.addWidget(self.fx_label, 0, 1)
        camera_layout.addWidget(QLabel("pixels"), 0, 2)

        camera_layout.addWidget(QLabel("Focal Length Y (fy):"), 1, 0)
        camera_layout.addWidget(self.fy_label, 1, 1)
        camera_layout.addWidget(QLabel("pixels"), 1, 2)

        camera_layout.addWidget(QLabel("<b>Principal Point X (cx):</b>"), 2, 0)
        principal_x_label = QLabel()
        principal_x_label.setObjectName("cx_value")
        self.cx_label = principal_x_label
        camera_layout.addWidget(self.cx_label, 2, 1)
        camera_layout.addWidget(QLabel("pixels"), 2, 2)

        camera_layout.addWidget(QLabel("<b>Principal Point Y (cy):</b>"), 3, 0)
        principal_y_label = QLabel()
        principal_y_label.setObjectName("cy_value")
        self.cy_label = principal_y_label
        camera_layout.addWidget(self.cy_label, 3, 1)
        camera_layout.addWidget(QLabel("pixels"), 3, 2)

        camera_group.setLayout(camera_layout)
        results_layout.addWidget(camera_group)

        # Distortion coefficients
        distortion_group = QGroupBox("Distortion Coefficients")
        distortion_layout = QGridLayout()

        self.k1_label = QLabel("--")
        self.k2_label = QLabel("--")
        self.k3_label = QLabel("--")
        self.k4_label = QLabel("--")
        self.p1_label = QLabel("--")
        self.p2_label = QLabel("--")

        distortion_layout.addWidget(QLabel("K1 (radial):"), 0, 0)
        distortion_layout.addWidget(self.k1_label, 0, 1)

        distortion_layout.addWidget(QLabel("K2 (radial):"), 1, 0)
        distortion_layout.addWidget(self.k2_label, 1, 1)

        distortion_layout.addWidget(QLabel("K3 (radial):"), 2, 0)
        distortion_layout.addWidget(self.k3_label, 2, 1)

        distortion_layout.addWidget(QLabel("K4 (radial):"), 3, 0)
        distortion_layout.addWidget(self.k4_label, 3, 1)

        distortion_layout.addWidget(QLabel("P1 (tangential):"), 0, 2)
        distortion_layout.addWidget(self.p1_label, 0, 3)

        distortion_layout.addWidget(QLabel("P2 (tangential):"), 1, 2)
        distortion_layout.addWidget(self.p2_label, 1, 3)

        distortion_group.setLayout(distortion_layout)
        results_layout.addWidget(distortion_group)

        # Quality metrics
        quality_group = QGroupBox("Quality Metrics")
        quality_layout = QVBoxLayout()

        self.rms_label = QLabel("RMS Error: --")
        self.rms_label.setStyleSheet("font-weight: bold;")
        quality_layout.addWidget(self.rms_label)

        self.stats_label = QLabel()
        self.stats_label.setWordWrap(True)
        quality_layout.addWidget(self.stats_label)

        quality_group.setLayout(quality_layout)
        results_layout.addWidget(quality_group)

        # Coverage analysis
        self.coverage_group = QGroupBox("Spatial Coverage")
        coverage_layout = QVBoxLayout()

        self.coverage_score_label = QLabel()
        self.coverage_score_label.setStyleSheet("font-weight: bold;")
        coverage_layout.addWidget(self.coverage_score_label)

        self.coverage_warnings_label = QLabel()
        self.coverage_warnings_label.setWordWrap(True)
        coverage_layout.addWidget(self.coverage_warnings_label)

        self.view_heatmap_btn = QPushButton("📊 View Coverage Heat Map")
        self.view_heatmap_btn.setToolTip("Show which parts of the frame have corner coverage")
        self.view_heatmap_btn.clicked.connect(self._show_coverage_heatmap)
        self.view_heatmap_btn.setEnabled(False)
        coverage_layout.addWidget(self.view_heatmap_btn)

        self.coverage_group.setLayout(coverage_layout)
        results_layout.addWidget(self.coverage_group)
        self.coverage_group.setVisible(False)

        # Action buttons (two rows)
        button_container = QVBoxLayout()
        button_container.setSpacing(5)

        # First row: Export and re-calibrate
        button_row1 = QHBoxLayout()

        self.export_json_btn = QPushButton("Export JSON")
        self.export_json_btn.clicked.connect(self._export_json)
        self.export_json_btn.setEnabled(False)
        button_row1.addWidget(self.export_json_btn)

        self.export_npz_btn = QPushButton("Export NumPy")
        self.export_npz_btn.clicked.connect(self._export_npz)
        self.export_npz_btn.setEnabled(False)
        button_row1.addWidget(self.export_npz_btn)

        button_container.addLayout(button_row1)

        # Second row: Re-calibrate and undistortion
        button_row2 = QHBoxLayout()

        self.recalibrate_btn = QPushButton("Re-calibrate")
        self.recalibrate_btn.setToolTip("Re-run calibration with current settings")
        self.recalibrate_btn.setEnabled(False)
        button_row2.addWidget(self.recalibrate_btn)

        self.undistortion_btn = QPushButton("Undistortion Preview")
        self.undistortion_btn.setToolTip("Show before/after undistortion comparison")
        self.undistortion_btn.clicked.connect(self._show_undistortion_preview)
        self.undistortion_btn.setEnabled(False)
        button_row2.addWidget(self.undistortion_btn)

        button_container.addLayout(button_row2)

        results_layout.addLayout(button_container)

        self.results_group.setLayout(results_layout)
        layout.addWidget(self.results_group)

        # Style principal point labels
        self.setStyleSheet("""
            #cx_value, #cy_value {
                font-weight: bold;
                font-size: 13px;
            }
        """)

        layout.addStretch()
        self.setLayout(layout)

        # Initially hide results
        self.results_group.setVisible(False)

    def set_progress(self, percent: int, message: str):
        """Update progress display.

        Args:
            percent: Progress percentage (0-100)
            message: Status message
        """
        self.progress_bar.setValue(percent)
        self.status_label.setText(message)

    def set_result(self, result: CalibrationResult):
        """Display calibration results.

        Args:
            result: Calibration result
        """
        self.result = result

        # Update camera parameters
        self.fx_label.setText(f"{result.fx:.2f}")
        self.fy_label.setText(f"{result.fy:.2f}")
        self.cx_label.setText(f"{result.cx:.2f}")
        self.cy_label.setText(f"{result.cy:.2f}")

        # Update distortion coefficients
        self.k1_label.setText(f"{result.k1:.6f}")
        self.k2_label.setText(f"{result.k2:.6f}")
        self.k3_label.setText(f"{result.k3:.6f}")
        self.k4_label.setText(f"{result.k4:.6f}")
        self.p1_label.setText(f"{result.p1:.6f}")
        self.p2_label.setText(f"{result.p2:.6f}")

        # Update quality metrics
        self.rms_label.setText(f"RMS Reprojection Error: {result.rms_error:.4f} pixels")

        stats = QualityMetrics.get_statistics(result)
        stats_text = f"""
        <b>Image Statistics:</b><br>
        Total images: {stats['total_images']}<br>
        Calibrated: {stats['calibrated_images']}<br>
        Excluded: {stats['excluded_images']}<br>
        <br>
        <b>Quality Distribution:</b><br>
        Excellent (&lt;0.3px): {stats['quality_counts']['excellent']} ({stats['quality_counts'].get('excellent_percent', 0):.1f}%)<br>
        Good (0.3-0.5px): {stats['quality_counts']['good']} ({stats['quality_counts'].get('good_percent', 0):.1f}%)<br>
        Acceptable (0.5-1.0px): {stats['quality_counts']['acceptable']} ({stats['quality_counts'].get('acceptable_percent', 0):.1f}%)<br>
        Poor (1.0-2.0px): {stats['quality_counts']['poor']} ({stats['quality_counts'].get('poor_percent', 0):.1f}%)<br>
        Bad (&gt;2.0px): {stats['quality_counts']['bad']} ({stats['quality_counts'].get('bad_percent', 0):.1f}%)
        """
        self.stats_label.setText(stats_text)

        # Update coverage analysis
        if result.coverage_report:
            cov = result.coverage_report

            # Coverage score with color coding
            score = cov.quality_score
            if score >= 80:
                color = "green"
            elif score >= 60:
                color = "orange"
            else:
                color = "red"

            score_text = f'<span style="color: {color}; font-size: 14px;">' \
                        f'{cov.quality_label} ({score:.0f}/100)</span><br>' \
                        f'Coverage: {cov.coverage_percentage:.0f}% of frame | ' \
                        f'{cov.num_images_used} images | ' \
                        f'{cov.total_corners} corners'
            self.coverage_score_label.setText(score_text)

            # Warnings
            if cov.warnings:
                warnings_html = '<span style="color: #CC6600;">⚠ Coverage Issues:</span><br>'
                warnings_html += '<br>'.join(f'• {w}' for w in cov.warnings)
                self.coverage_warnings_label.setText(warnings_html)
            elif not cov.is_adequate:
                self.coverage_warnings_label.setText('<span style="color: #CC6600;">⚠ Coverage is marginal</span>')
            else:
                self.coverage_warnings_label.setText('<span style="color: green;">✓ Good coverage</span>')

            self.coverage_group.setVisible(True)
            self.view_heatmap_btn.setEnabled(True)
        else:
            self.coverage_group.setVisible(False)

        # Show results and enable buttons
        self.results_group.setVisible(True)
        self.export_json_btn.setEnabled(True)
        self.export_npz_btn.setEnabled(True)
        self.undistortion_btn.setEnabled(True)

    def set_recalibrate_enabled(self, enabled: bool):
        """Enable/disable re-calibrate button.

        Args:
            enabled: Whether re-calibration is available
        """
        self.recalibrate_btn.setEnabled(enabled and self.result is not None)

    def _export_json(self):
        """Export calibration results to JSON."""
        if not self.result:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Calibration (JSON)",
            "calibration.json",
            "JSON Files (*.json)"
        )

        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(self.result.to_dict(), f, indent=2)

                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Calibration saved to:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Export Failed",
                    f"Failed to save calibration:\n{str(e)}"
                )

    def _export_npz(self):
        """Export calibration results to NumPy format."""
        if not self.result:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Calibration (NumPy)",
            "calibration.npz",
            "NumPy Files (*.npz)"
        )

        if file_path:
            try:
                np.savez(
                    file_path,
                    camera_matrix=self.result.camera_matrix,
                    dist_coeffs=self.result.dist_coeffs,
                    rms_error=self.result.rms_error,
                    image_size=np.array(self.result.image_size)
                )

                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Calibration saved to:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Export Failed",
                    f"Failed to save calibration:\n{str(e)}"
                )

    def _show_undistortion_preview(self):
        """Show undistortion preview dialog."""
        if not self.result:
            return

        from .undistortion_dialog import UndistortionDialog
        dialog = UndistortionDialog(self.result, self)
        dialog.exec()

    def _show_coverage_heatmap(self):
        """Show coverage heat map dialog."""
        if not self.result or not self.result.coverage_report:
            return

        from .coverage_heatmap_dialog import CoverageHeatmapDialog
        dialog = CoverageHeatmapDialog(self.result, self)
        dialog.exec()

    def reset(self):
        """Reset panel to initial state."""
        self.result = None
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready to calibrate")
        self.results_group.setVisible(False)
        self.coverage_group.setVisible(False)
        self.export_json_btn.setEnabled(False)
        self.export_npz_btn.setEnabled(False)
        self.recalibrate_btn.setEnabled(False)
        self.undistortion_btn.setEnabled(False)
        self.view_heatmap_btn.setEnabled(False)
