"""Board configuration dialog."""

from typing import Optional, List, Tuple
import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QSpinBox, QDoubleSpinBox, QPushButton,
    QComboBox, QDialogButtonBox, QMessageBox
)
from PyQt6.QtCore import Qt

from ..core import BoardConfig, auto_detect_board
from ..core.calibration_options import CalibrationOptions


class ConfigDialog(QDialog):
    """Dialog for configuring ChArUco board parameters."""

    def __init__(self, parent=None, initial_config=None, images=None, initial_options=None):
        """Initialize configuration dialog.

        Args:
            parent: Parent widget
            initial_config: Optional initial BoardConfig to populate fields
            images: Optional images for auto-detection
            initial_options: Optional initial CalibrationOptions
        """
        super().__init__(parent)
        self.setWindowTitle("ChArUco Board Configuration")
        self.setModal(True)
        self.setMinimumWidth(400)

        self.config = None
        self.images = images
        self.calibration_options = initial_options if initial_options else CalibrationOptions()
        self._setup_ui(initial_config)

    def _setup_ui(self, initial_config):
        """Setup dialog UI.

        Args:
            initial_config: Optional initial configuration
        """
        layout = QVBoxLayout()

        # Description
        desc_label = QLabel(
            "Configure the ChArUco calibration board parameters.\n"
            "These must match your physical calibration board."
        )
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        # Form layout for parameters
        form_layout = QFormLayout()

        # Squares X
        self.squares_x_spin = QSpinBox()
        self.squares_x_spin.setMinimum(3)
        self.squares_x_spin.setMaximum(20)
        self.squares_x_spin.setValue(5 if not initial_config else initial_config.squares_x)
        form_layout.addRow("Squares in X:", self.squares_x_spin)

        # Squares Y
        self.squares_y_spin = QSpinBox()
        self.squares_y_spin.setMinimum(3)
        self.squares_y_spin.setMaximum(20)
        self.squares_y_spin.setValue(7 if not initial_config else initial_config.squares_y)
        form_layout.addRow("Squares in Y:", self.squares_y_spin)

        # Square length
        self.square_length_spin = QDoubleSpinBox()
        self.square_length_spin.setMinimum(1.0)
        self.square_length_spin.setMaximum(1000.0)
        self.square_length_spin.setSuffix(" mm")
        self.square_length_spin.setValue(30.0 if not initial_config else initial_config.square_length)
        form_layout.addRow("Square Length:", self.square_length_spin)

        # Marker length
        self.marker_length_spin = QDoubleSpinBox()
        self.marker_length_spin.setMinimum(1.0)
        self.marker_length_spin.setMaximum(1000.0)
        self.marker_length_spin.setSuffix(" mm")
        self.marker_length_spin.setValue(20.0 if not initial_config else initial_config.marker_length)
        form_layout.addRow("Marker Length:", self.marker_length_spin)

        # ArUco dictionary
        self.dict_combo = QComboBox()
        dict_options = [
            'DICT_4X4_50', 'DICT_4X4_100', 'DICT_4X4_250', 'DICT_4X4_1000',
            'DICT_5X5_50', 'DICT_5X5_100', 'DICT_5X5_250', 'DICT_5X5_1000',
            'DICT_6X6_50', 'DICT_6X6_100', 'DICT_6X6_250', 'DICT_6X6_1000',
            'DICT_7X7_50', 'DICT_7X7_100', 'DICT_7X7_250', 'DICT_7X7_1000',
        ]
        self.dict_combo.addItems(dict_options)
        default_idx = dict_options.index('DICT_6X6_250')
        if initial_config:
            try:
                default_idx = dict_options.index(initial_config.dict_name)
            except ValueError:
                pass
        self.dict_combo.setCurrentIndex(default_idx)
        form_layout.addRow("ArUco Dictionary:", self.dict_combo)

        layout.addLayout(form_layout)

        # Validation note
        note_label = QLabel(
            "Note: Marker length must be less than square length."
        )
        note_label.setStyleSheet("color: #666;")
        layout.addWidget(note_label)

        # Auto-detect and advanced options buttons
        buttons_layout = QHBoxLayout()

        if self.images:
            auto_detect_btn = QPushButton("🔍 Auto-Detect from Images")
            auto_detect_btn.setToolTip("Automatically detect board configuration from loaded images")
            auto_detect_btn.clicked.connect(self._auto_detect)
            buttons_layout.addWidget(auto_detect_btn)

        advanced_btn = QPushButton("⚙️ Advanced Options...")
        advanced_btn.setToolTip("Configure advanced calibration options for improved accuracy")
        advanced_btn.clicked.connect(self._show_advanced_options)
        buttons_layout.addWidget(advanced_btn)

        buttons_layout.addStretch()
        layout.addLayout(buttons_layout)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._validate_and_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def _validate_and_accept(self):
        """Validate input and accept dialog."""
        try:
            # Create config to validate
            self.config = BoardConfig(
                squares_x=self.squares_x_spin.value(),
                squares_y=self.squares_y_spin.value(),
                square_length=self.square_length_spin.value(),
                marker_length=self.marker_length_spin.value(),
                dict_name=self.dict_combo.currentText(),
            )
            self.accept()

        except ValueError as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Invalid Configuration", str(e))

    def _auto_detect(self):
        """Auto-detect board configuration from images."""
        if not self.images:
            QMessageBox.warning(
                self,
                "No Images",
                "No images available for auto-detection. Please load images first."
            )
            return

        # Show progress message
        from PyQt6.QtWidgets import QApplication
        progress = QMessageBox(self)
        progress.setWindowTitle("Auto-Detecting Board")
        progress.setText("Analyzing images to detect board configuration...")
        progress.setStandardButtons(QMessageBox.StandardButton.NoButton)
        progress.show()

        # Force the dialog to display immediately
        QApplication.processEvents()
        progress.repaint()
        QApplication.processEvents()

        try:
            # Run detection
            detected_config, message = auto_detect_board(self.images, max_images=5)

            progress.close()
            QApplication.processEvents()  # Ensure progress closes before next dialog

            if detected_config:
                # Update fields with detected values
                self.squares_x_spin.setValue(detected_config.squares_x)
                self.squares_y_spin.setValue(detected_config.squares_y)
                self.square_length_spin.setValue(detected_config.square_length)
                self.marker_length_spin.setValue(detected_config.marker_length)

                # Set dictionary
                try:
                    idx = self.dict_combo.findText(detected_config.dict_name)
                    if idx >= 0:
                        self.dict_combo.setCurrentIndex(idx)
                except Exception:
                    pass

                QMessageBox.information(
                    self,
                    "Auto-Detection Successful",
                    message
                )
            else:
                QMessageBox.warning(
                    self,
                    "Auto-Detection Failed",
                    message
                )

        except Exception as e:
            progress.close()
            QMessageBox.critical(
                self,
                "Auto-Detection Error",
                f"An error occurred during auto-detection:\n{str(e)}"
            )

    def _show_advanced_options(self):
        """Show advanced calibration options dialog."""
        from .advanced_options_dialog import AdvancedOptionsDialog

        dialog = AdvancedOptionsDialog(self, self.calibration_options)
        if dialog.exec():
            self.calibration_options = dialog.get_options()

    def get_config(self):
        """Get the configured BoardConfig.

        Returns:
            BoardConfig or None if dialog was cancelled
        """
        return self.config

    def get_config_and_options(self):
        """Get the configured BoardConfig and CalibrationOptions.

        Returns:
            Tuple of (BoardConfig, CalibrationOptions) or (None, None) if cancelled
        """
        return (self.config, self.calibration_options)
