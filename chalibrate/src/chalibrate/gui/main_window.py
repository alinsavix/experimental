"""Main application window."""

from typing import Optional, List, Tuple
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QMenuBar, QMenu, QFileDialog, QMessageBox, QStatusBar,
    QToolBar
)
from PyQt6.QtCore import Qt, QThreadPool
from PyQt6.QtGui import QAction
import numpy as np

from ..core import BoardConfig, CalibrationResult
from ..core.calibration_options import CalibrationOptions
from ..utils import ImageLoader
from .config_dialog import ConfigDialog
from .image_grid import ImageGrid
from .calibration_panel import CalibrationPanel
from .workers import CalibrationWorker


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(
        self,
        preload_config: Optional[BoardConfig] = None,
        preload_images: Optional[List[Tuple[str, np.ndarray]]] = None
    ):
        """Initialize main window.

        Args:
            preload_config: Optional pre-loaded board configuration
            preload_images: Optional pre-loaded images
        """
        super().__init__()
        self.setWindowTitle("ChArUco Camera Calibration")
        self.setMinimumSize(1200, 800)

        self.board_config: Optional[BoardConfig] = preload_config
        self.calibration_options = CalibrationOptions()
        self.images: List[Tuple[str, np.ndarray]] = preload_images or []
        self.current_result: Optional[CalibrationResult] = None
        self.thread_pool = QThreadPool.globalInstance()

        self._setup_ui()
        self._create_menus()
        self._create_toolbar()

        # If images pre-loaded, display them
        if self.images:
            self.image_grid.load_images(self.images)
            self.statusBar().showMessage(f"Loaded {len(self.images)} images")
            self._update_actions()
        else:
            # Show config dialog if no config provided
            if not self.board_config:
                self._configure_board()

    def _setup_ui(self):
        """Setup UI components."""
        # Central widget with splitter
        central_widget = QWidget()
        layout = QHBoxLayout()

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Image grid
        self.image_grid = ImageGrid()
        self.image_grid.imageClicked.connect(self._show_image_viewer)
        self.image_grid.exclusionChanged.connect(self._on_exclusion_changed)
        splitter.addWidget(self.image_grid)

        # Right: Calibration panel
        self.calibration_panel = CalibrationPanel()
        self.calibration_panel.recalibrate_btn.clicked.connect(self._recalibrate)
        splitter.addWidget(self.calibration_panel)

        # Set initial splitter sizes (70% images, 30% panel)
        splitter.setSizes([700, 300])

        layout.addWidget(splitter)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Status bar
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")

    def _create_menus(self):
        """Create menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        self.load_images_action = QAction("&Load Images...", self)
        self.load_images_action.setShortcut("Ctrl+O")
        self.load_images_action.triggered.connect(self._load_images)
        file_menu.addAction(self.load_images_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Calibration menu
        calib_menu = menubar.addMenu("&Calibration")

        self.config_action = QAction("&Configure Board...", self)
        self.config_action.triggered.connect(self._configure_board)
        calib_menu.addAction(self.config_action)

        self.advanced_options_action = QAction("&Advanced Options...", self)
        self.advanced_options_action.triggered.connect(self._show_advanced_options)
        calib_menu.addAction(self.advanced_options_action)

        calib_menu.addSeparator()

        self.calibrate_action = QAction("&Run Calibration", self)
        self.calibrate_action.setShortcut("Ctrl+R")
        self.calibrate_action.setEnabled(False)
        self.calibrate_action.triggered.connect(self._start_calibration)
        calib_menu.addAction(self.calibrate_action)

    def _create_toolbar(self):
        """Create toolbar."""
        toolbar = QToolBar()
        self.addToolBar(toolbar)

        toolbar.addAction(self.load_images_action)
        toolbar.addSeparator()
        toolbar.addAction(self.config_action)
        toolbar.addAction(self.calibrate_action)

    def _update_actions(self):
        """Update action enabled states."""
        has_images = len(self.images) > 0
        has_config = self.board_config is not None
        can_calibrate = has_images and has_config

        self.calibrate_action.setEnabled(can_calibrate)

    def _configure_board(self):
        """Show board configuration dialog."""
        dialog = ConfigDialog(
            self,
            self.board_config,
            self.images if self.images else None,
            self.calibration_options
        )
        if dialog.exec():
            self.board_config, self.calibration_options = dialog.get_config_and_options()
            self.statusBar().showMessage(
                f"Board configured: {self.board_config.squares_x}x{self.board_config.squares_y}"
            )
            self._update_actions()

    def _show_advanced_options(self):
        """Show advanced calibration options dialog."""
        from .advanced_options_dialog import AdvancedOptionsDialog

        dialog = AdvancedOptionsDialog(self, self.calibration_options)
        if dialog.exec():
            self.calibration_options = dialog.get_options()
            self.statusBar().showMessage("Advanced calibration options updated")

    def _load_images(self):
        """Load images from directory."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Image Directory",
            "",
            QFileDialog.Option.ShowDirsOnly
        )

        if directory:
            try:
                self.images = ImageLoader.load_directory(directory)
                self.image_grid.load_images(self.images)
                self.statusBar().showMessage(f"Loaded {len(self.images)} images from {directory}")
                self._update_actions()

                # Reset calibration panel
                self.calibration_panel.reset()
                self.current_result = None

            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error Loading Images",
                    f"Failed to load images:\n{str(e)}"
                )

    def _start_calibration(self):
        """Start calibration in background thread."""
        if not self.board_config or not self.images:
            return

        # Disable calibration during processing
        self.calibrate_action.setEnabled(False)
        self.calibration_panel.reset()

        # Create and start worker
        worker = CalibrationWorker(self.board_config, self.images, self.calibration_options)
        worker.signals.progress.connect(self._on_calibration_progress)
        worker.signals.detections_ready.connect(self._on_detections_ready)
        worker.signals.finished.connect(self._on_calibration_finished)
        worker.signals.error.connect(self._on_calibration_error)

        self.thread_pool.start(worker)
        self.statusBar().showMessage("Calibration started...")

    def _recalibrate(self):
        """Re-run calibration without excluded images."""
        if not self.board_config or not self.images:
            return

        # Disable calibration during processing
        self.calibrate_action.setEnabled(False)
        self.calibration_panel.set_recalibrate_enabled(False)

        # Create and start worker (uses existing detections)
        worker = CalibrationWorker(self.board_config, self.images, recalibrate=True)
        worker.signals.progress.connect(self._on_calibration_progress)
        worker.signals.finished.connect(self._on_calibration_finished)
        worker.signals.error.connect(self._on_calibration_error)

        self.thread_pool.start(worker)
        self.statusBar().showMessage("Re-calibrating without excluded images...")

    def _on_calibration_progress(self, percent: int, message: str):
        """Handle calibration progress update.

        Args:
            percent: Progress percentage
            message: Status message
        """
        self.calibration_panel.set_progress(percent, message)
        self.statusBar().showMessage(message)

    def _on_detections_ready(self, detections: list):
        """Handle detection completion (before calibration).

        Updates image grid with quality information.

        Args:
            detections: List of ImageDetection objects with quality reports
        """
        # Update image grid with quality info
        self.image_grid.set_detections(detections)

        # Count quality metrics
        total = len(detections)
        detected = sum(1 for d in detections if d.has_detection)
        auto_excluded = sum(1 for d in detections if d.auto_excluded)

        status = f"Detected {detected}/{total} boards"
        if auto_excluded > 0:
            status += f" ({auto_excluded} auto-excluded)"

        self.statusBar().showMessage(status)

    def _on_calibration_finished(self, result: CalibrationResult):
        """Handle calibration completion.

        Args:
            result: Calibration result
        """
        self.current_result = result

        # Update UI
        self.calibration_panel.set_result(result)
        self.image_grid.set_detections(result.detections)

        # Update status
        self.statusBar().showMessage(
            f"Calibration complete - RMS error: {result.rms_error:.4f} pixels"
        )

        # Re-enable calibration
        self.calibrate_action.setEnabled(True)

        # Enable re-calibrate if there are excluded images
        excluded_count = self.image_grid.get_excluded_count()
        self.calibration_panel.set_recalibrate_enabled(excluded_count > 0)

    def _on_calibration_error(self, error_message: str):
        """Handle calibration error.

        Args:
            error_message: Error message
        """
        QMessageBox.critical(
            self,
            "Calibration Error",
            f"Calibration failed:\n\n{error_message}"
        )

        self.statusBar().showMessage("Calibration failed")
        self.calibrate_action.setEnabled(True)

    def _on_exclusion_changed(self):
        """Handle image exclusion status change."""
        if self.current_result:
            excluded_count = self.image_grid.get_excluded_count()
            self.calibration_panel.set_recalibrate_enabled(excluded_count > 0)

            if excluded_count > 0:
                self.statusBar().showMessage(
                    f"{excluded_count} image(s) excluded - click Re-calibrate to update results"
                )
            else:
                self.statusBar().showMessage("All images included in calibration")

    def _show_image_viewer(self, index: int):
        """Show full-size image viewer.

        Args:
            index: Image index
        """
        if not self.current_result or index >= len(self.current_result.detections):
            return

        # Import here to avoid circular dependency
        from .image_viewer import ImageViewer

        detection = self.current_result.detections[index]
        viewer = ImageViewer(detection, self.current_result, self.board_config, self)
        viewer.exec()
