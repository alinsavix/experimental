"""Advanced calibration options dialog."""

from typing import Optional
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QTabWidget,
    QLabel, QSpinBox, QDoubleSpinBox, QPushButton, QCheckBox,
    QComboBox, QDialogButtonBox, QGroupBox, QWidget
)
from PyQt6.QtCore import Qt

from ..core.calibration_options import CalibrationOptions


class AdvancedOptionsDialog(QDialog):
    """Dialog for configuring advanced calibration options."""

    def __init__(self, parent=None, initial_options: Optional[CalibrationOptions] = None):
        """Initialize advanced options dialog.

        Args:
            parent: Parent widget
            initial_options: Optional initial CalibrationOptions to populate fields
        """
        super().__init__(parent)
        self.setWindowTitle("Advanced Calibration Options")
        self.setModal(True)
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)

        self.options = initial_options if initial_options else CalibrationOptions()
        self._setup_ui()

    def _setup_ui(self):
        """Setup dialog UI."""
        layout = QVBoxLayout()

        # Preset selector at top
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(['Custom', 'default', 'high_accuracy', 'fast', 'fisheye', 'webcam'])
        self.preset_combo.setCurrentIndex(0)
        self.preset_combo.currentTextChanged.connect(self._load_preset)
        preset_layout.addWidget(self.preset_combo)
        preset_layout.addStretch()
        layout.addLayout(preset_layout)

        # Tabbed interface
        tabs = QTabWidget()
        tabs.addTab(self._create_basic_tab(), "Basic")
        tabs.addTab(self._create_calibration_tab(), "Calibration Flags")
        tabs.addTab(self._create_detection_tab(), "Detection")
        tabs.addTab(self._create_preprocessing_tab(), "Preprocessing")
        tabs.addTab(self._create_refinement_tab(), "Iterative Refinement")
        layout.addWidget(tabs)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._save_and_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)
        self._populate_fields()

    def _create_basic_tab(self) -> QWidget:
        """Create basic options tab."""
        widget = QWidget()
        layout = QVBoxLayout()

        # Subpixel refinement group
        subpixel_group = QGroupBox("Subpixel Corner Refinement")
        subpixel_layout = QVBoxLayout()

        self.enable_subpixel_check = QCheckBox("Enable subpixel refinement")
        self.enable_subpixel_check.setToolTip(
            "Refine corner positions to subpixel accuracy. "
            "Provides 30-50% accuracy improvement with minimal performance cost."
        )
        subpixel_layout.addWidget(self.enable_subpixel_check)

        subpixel_form = QFormLayout()
        self.subpixel_window_spin = QSpinBox()
        self.subpixel_window_spin.setMinimum(3)
        self.subpixel_window_spin.setMaximum(15)
        self.subpixel_window_spin.setValue(5)
        self.subpixel_window_spin.setToolTip("Window size for corner refinement (larger = more stable but slower)")
        subpixel_form.addRow("Window Size:", self.subpixel_window_spin)
        subpixel_layout.addLayout(subpixel_form)

        subpixel_group.setLayout(subpixel_layout)
        layout.addWidget(subpixel_group)

        # Quality filtering group
        quality_group = QGroupBox("Image Quality Filtering")
        quality_layout = QVBoxLayout()

        self.enable_quality_check = QCheckBox("Enable automatic quality filtering")
        self.enable_quality_check.setToolTip(
            "Automatically exclude blurry, too dark, or too bright images. "
            "Prevents poor quality images from corrupting calibration."
        )
        quality_layout.addWidget(self.enable_quality_check)

        quality_form = QFormLayout()

        self.blur_threshold_spin = QDoubleSpinBox()
        self.blur_threshold_spin.setMinimum(10.0)
        self.blur_threshold_spin.setMaximum(500.0)
        self.blur_threshold_spin.setValue(30.0)
        self.blur_threshold_spin.setToolTip("Minimum Laplacian variance (higher = sharper required). 30 is typical, <15 is very blurry.")
        quality_form.addRow("Blur Threshold:", self.blur_threshold_spin)

        self.brightness_min_spin = QDoubleSpinBox()
        self.brightness_min_spin.setMinimum(0.0)
        self.brightness_min_spin.setMaximum(255.0)
        self.brightness_min_spin.setValue(30.0)
        self.brightness_min_spin.setToolTip("Minimum acceptable brightness (0-255)")
        quality_form.addRow("Min Brightness:", self.brightness_min_spin)

        self.brightness_max_spin = QDoubleSpinBox()
        self.brightness_max_spin.setMinimum(0.0)
        self.brightness_max_spin.setMaximum(255.0)
        self.brightness_max_spin.setValue(225.0)
        self.brightness_max_spin.setToolTip("Maximum acceptable brightness (0-255)")
        quality_form.addRow("Max Brightness:", self.brightness_max_spin)

        quality_layout.addLayout(quality_form)
        quality_group.setLayout(quality_layout)
        layout.addWidget(quality_group)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def _create_calibration_tab(self) -> QWidget:
        """Create calibration flags tab."""
        widget = QWidget()
        layout = QVBoxLayout()

        desc = QLabel(
            "Advanced distortion model options. Most cameras work best with defaults. "
            "Enable rational model for wide-angle lenses, thin prism for high precision."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Distortion model
        model_group = QGroupBox("Distortion Model")
        model_layout = QVBoxLayout()

        self.rational_model_check = QCheckBox("Use rational distortion model (K4, K5, K6)")
        self.rational_model_check.setToolTip("Adds higher-order radial distortion terms. Good for wide-angle lenses.")
        model_layout.addWidget(self.rational_model_check)

        self.thin_prism_check = QCheckBox("Use thin prism model (S1-S4)")
        self.thin_prism_check.setToolTip("Adds thin prism distortion terms. Rarely needed.")
        model_layout.addWidget(self.thin_prism_check)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Constraints
        constraints_group = QGroupBox("Calibration Constraints")
        constraints_layout = QVBoxLayout()

        self.fix_principal_point_check = QCheckBox("Fix principal point at image center")
        self.fix_principal_point_check.setToolTip("Constrains principal point to image center. Use for symmetric lenses.")
        constraints_layout.addWidget(self.fix_principal_point_check)

        self.fix_aspect_ratio_check = QCheckBox("Fix aspect ratio (fx/fy = 1)")
        self.fix_aspect_ratio_check.setToolTip("Forces square pixels. Use for known symmetric sensors.")
        constraints_layout.addWidget(self.fix_aspect_ratio_check)

        self.zero_tangent_check = QCheckBox("Zero tangential distortion (P1, P2 = 0)")
        self.zero_tangent_check.setToolTip("Assumes no tangential distortion. Use for high-quality lenses.")
        constraints_layout.addWidget(self.zero_tangent_check)

        constraints_group.setLayout(constraints_layout)
        layout.addWidget(constraints_group)

        # Individual coefficient fixes
        coeff_group = QGroupBox("Fix Individual Coefficients")
        coeff_layout = QVBoxLayout()

        self.fix_k1_check = QCheckBox("Fix K1 (primary radial distortion)")
        coeff_layout.addWidget(self.fix_k1_check)

        self.fix_k2_check = QCheckBox("Fix K2 (secondary radial distortion)")
        coeff_layout.addWidget(self.fix_k2_check)

        self.fix_k3_check = QCheckBox("Fix K3 (tertiary radial distortion)")
        coeff_layout.addWidget(self.fix_k3_check)

        coeff_group.setLayout(coeff_layout)
        layout.addWidget(coeff_group)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def _create_detection_tab(self) -> QWidget:
        """Create ArUco detection parameters tab."""
        widget = QWidget()
        layout = QVBoxLayout()

        desc = QLabel(
            "ArUco marker detection parameters. Defaults work well for most cases. "
            "Adjust for challenging lighting or small markers."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        form = QFormLayout()

        self.adapt_win_min_spin = QSpinBox()
        self.adapt_win_min_spin.setMinimum(3)
        self.adapt_win_min_spin.setMaximum(50)
        self.adapt_win_min_spin.setValue(3)
        self.adapt_win_min_spin.setToolTip("Minimum window size for adaptive thresholding")
        form.addRow("Adaptive Thresh Win Min:", self.adapt_win_min_spin)

        self.adapt_win_max_spin = QSpinBox()
        self.adapt_win_max_spin.setMinimum(3)
        self.adapt_win_max_spin.setMaximum(100)
        self.adapt_win_max_spin.setValue(23)
        self.adapt_win_max_spin.setToolTip("Maximum window size for adaptive thresholding")
        form.addRow("Adaptive Thresh Win Max:", self.adapt_win_max_spin)

        self.adapt_win_step_spin = QSpinBox()
        self.adapt_win_step_spin.setMinimum(1)
        self.adapt_win_step_spin.setMaximum(50)
        self.adapt_win_step_spin.setValue(10)
        self.adapt_win_step_spin.setToolTip("Step size for adaptive thresholding window")
        form.addRow("Adaptive Thresh Win Step:", self.adapt_win_step_spin)

        self.adapt_constant_spin = QDoubleSpinBox()
        self.adapt_constant_spin.setMinimum(0.0)
        self.adapt_constant_spin.setMaximum(20.0)
        self.adapt_constant_spin.setValue(7.0)
        self.adapt_constant_spin.setToolTip("Constant subtracted from mean in adaptive thresholding")
        form.addRow("Adaptive Thresh Constant:", self.adapt_constant_spin)

        layout.addLayout(form)
        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def _create_preprocessing_tab(self) -> QWidget:
        """Create preprocessing options tab."""
        widget = QWidget()
        layout = QVBoxLayout()

        desc = QLabel(
            "Image preprocessing can help with difficult lighting conditions. "
            "CLAHE (Contrast Limited Adaptive Histogram Equalization) improves contrast."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        clahe_group = QGroupBox("CLAHE Preprocessing")
        clahe_layout = QVBoxLayout()

        self.enable_clahe_check = QCheckBox("Enable CLAHE preprocessing")
        self.enable_clahe_check.setToolTip(
            "Apply CLAHE to improve contrast. Useful for uneven lighting or low-contrast images."
        )
        clahe_layout.addWidget(self.enable_clahe_check)

        clahe_form = QFormLayout()

        self.clahe_clip_spin = QDoubleSpinBox()
        self.clahe_clip_spin.setMinimum(1.0)
        self.clahe_clip_spin.setMaximum(10.0)
        self.clahe_clip_spin.setValue(2.0)
        self.clahe_clip_spin.setToolTip("Threshold for contrast limiting (higher = more contrast)")
        clahe_form.addRow("Clip Limit:", self.clahe_clip_spin)

        self.clahe_tile_spin = QSpinBox()
        self.clahe_tile_spin.setMinimum(4)
        self.clahe_tile_spin.setMaximum(16)
        self.clahe_tile_spin.setValue(8)
        self.clahe_tile_spin.setToolTip("Tile grid size for histogram equalization")
        clahe_form.addRow("Tile Size:", self.clahe_tile_spin)

        clahe_layout.addLayout(clahe_form)
        clahe_group.setLayout(clahe_layout)
        layout.addWidget(clahe_group)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def _create_refinement_tab(self) -> QWidget:
        """Create iterative refinement tab."""
        widget = QWidget()
        layout = QVBoxLayout()

        desc = QLabel(
            "Iterative refinement automatically detects and removes outlier images "
            "based on reprojection error, then re-calibrates. Can improve accuracy by 10-20% "
            "but increases processing time."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        refine_group = QGroupBox("Outlier Removal")
        refine_layout = QVBoxLayout()

        self.enable_refinement_check = QCheckBox("Enable iterative refinement")
        self.enable_refinement_check.setToolTip(
            "Iteratively remove high-error images and re-calibrate. "
            "Can improve accuracy but takes longer."
        )
        refine_layout.addWidget(self.enable_refinement_check)

        refine_form = QFormLayout()

        self.max_iterations_spin = QSpinBox()
        self.max_iterations_spin.setMinimum(1)
        self.max_iterations_spin.setMaximum(10)
        self.max_iterations_spin.setValue(3)
        self.max_iterations_spin.setToolTip("Maximum number of refinement iterations")
        refine_form.addRow("Max Iterations:", self.max_iterations_spin)

        self.outlier_percentile_spin = QDoubleSpinBox()
        self.outlier_percentile_spin.setMinimum(50.0)
        self.outlier_percentile_spin.setMaximum(99.0)
        self.outlier_percentile_spin.setValue(95.0)
        self.outlier_percentile_spin.setToolTip("Error percentile threshold (images above this are outliers)")
        refine_form.addRow("Outlier Percentile:", self.outlier_percentile_spin)

        self.min_images_spin = QSpinBox()
        self.min_images_spin.setMinimum(3)
        self.min_images_spin.setMaximum(50)
        self.min_images_spin.setValue(5)
        self.min_images_spin.setToolTip("Minimum images to keep (stops removing outliers if below this)")
        refine_form.addRow("Min Images After Outliers:", self.min_images_spin)

        refine_layout.addLayout(refine_form)
        refine_group.setLayout(refine_layout)
        layout.addWidget(refine_group)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def _populate_fields(self):
        """Populate fields from current options."""
        # Basic
        self.enable_subpixel_check.setChecked(self.options.enable_subpixel)
        self.subpixel_window_spin.setValue(self.options.subpixel_window_size[0])
        self.enable_quality_check.setChecked(self.options.enable_quality_filter)
        self.blur_threshold_spin.setValue(self.options.blur_threshold)
        self.brightness_min_spin.setValue(self.options.brightness_min)
        self.brightness_max_spin.setValue(self.options.brightness_max)

        # Calibration flags
        self.rational_model_check.setChecked(self.options.use_rational_model)
        self.thin_prism_check.setChecked(self.options.use_thin_prism)
        self.fix_principal_point_check.setChecked(self.options.fix_principal_point)
        self.fix_aspect_ratio_check.setChecked(self.options.fix_aspect_ratio)
        self.zero_tangent_check.setChecked(self.options.zero_tangent_dist)
        self.fix_k1_check.setChecked(self.options.fix_k1)
        self.fix_k2_check.setChecked(self.options.fix_k2)
        self.fix_k3_check.setChecked(self.options.fix_k3)

        # Detection
        self.adapt_win_min_spin.setValue(self.options.adaptive_thresh_win_size_min)
        self.adapt_win_max_spin.setValue(self.options.adaptive_thresh_win_size_max)
        self.adapt_win_step_spin.setValue(self.options.adaptive_thresh_win_size_step)
        self.adapt_constant_spin.setValue(self.options.adaptive_thresh_constant)

        # Preprocessing
        self.enable_clahe_check.setChecked(self.options.enable_clahe)
        self.clahe_clip_spin.setValue(self.options.clahe_clip_limit)
        self.clahe_tile_spin.setValue(self.options.clahe_tile_size[0])

        # Refinement
        self.enable_refinement_check.setChecked(self.options.enable_iterative_refinement)
        self.max_iterations_spin.setValue(self.options.max_refinement_iterations)
        self.outlier_percentile_spin.setValue(self.options.outlier_percentile)
        self.min_images_spin.setValue(self.options.min_images_after_outliers)

    def _load_preset(self, preset_name: str):
        """Load a preset configuration."""
        if preset_name == 'Custom':
            return

        try:
            self.options = CalibrationOptions.from_preset(preset_name)
            self._populate_fields()
        except ValueError:
            pass

    def _save_and_accept(self):
        """Save fields to options and accept dialog."""
        # Basic
        self.options.enable_subpixel = self.enable_subpixel_check.isChecked()
        self.options.subpixel_window_size = (
            self.subpixel_window_spin.value(),
            self.subpixel_window_spin.value()
        )
        self.options.enable_quality_filter = self.enable_quality_check.isChecked()
        self.options.blur_threshold = self.blur_threshold_spin.value()
        self.options.brightness_min = self.brightness_min_spin.value()
        self.options.brightness_max = self.brightness_max_spin.value()

        # Calibration flags
        self.options.use_rational_model = self.rational_model_check.isChecked()
        self.options.use_thin_prism = self.thin_prism_check.isChecked()
        self.options.fix_principal_point = self.fix_principal_point_check.isChecked()
        self.options.fix_aspect_ratio = self.fix_aspect_ratio_check.isChecked()
        self.options.zero_tangent_dist = self.zero_tangent_check.isChecked()
        self.options.fix_k1 = self.fix_k1_check.isChecked()
        self.options.fix_k2 = self.fix_k2_check.isChecked()
        self.options.fix_k3 = self.fix_k3_check.isChecked()

        # Detection
        self.options.adaptive_thresh_win_size_min = self.adapt_win_min_spin.value()
        self.options.adaptive_thresh_win_size_max = self.adapt_win_max_spin.value()
        self.options.adaptive_thresh_win_size_step = self.adapt_win_step_spin.value()
        self.options.adaptive_thresh_constant = self.adapt_constant_spin.value()

        # Preprocessing
        self.options.enable_clahe = self.enable_clahe_check.isChecked()
        self.options.clahe_clip_limit = self.clahe_clip_spin.value()
        self.options.clahe_tile_size = (
            self.clahe_tile_spin.value(),
            self.clahe_tile_spin.value()
        )

        # Refinement
        self.options.enable_iterative_refinement = self.enable_refinement_check.isChecked()
        self.options.max_refinement_iterations = self.max_iterations_spin.value()
        self.options.outlier_percentile = self.outlier_percentile_spin.value()
        self.options.min_images_after_outliers = self.min_images_spin.value()

        self.accept()

    def get_options(self) -> CalibrationOptions:
        """Get the configured CalibrationOptions.

        Returns:
            CalibrationOptions instance
        """
        return self.options
