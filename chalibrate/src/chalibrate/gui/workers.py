"""Background worker threads for calibration."""

from typing import List, Tuple
from PyQt6.QtCore import QObject, QRunnable, pyqtSignal, pyqtSlot
import numpy as np

from ..core import BoardConfig, Calibrator, CalibrationResult
from ..utils import ImageLoader


class WorkerSignals(QObject):
    """Signals for calibration worker."""

    progress = pyqtSignal(int, str)  # (percentage, status_message)
    finished = pyqtSignal(object)  # CalibrationResult
    error = pyqtSignal(str)  # Error message


class CalibrationWorker(QRunnable):
    """Worker thread for running calibration in background."""

    def __init__(
        self,
        board_config: BoardConfig,
        images: List[Tuple[str, np.ndarray]],
        recalibrate: bool = False
    ):
        """Initialize calibration worker.

        Args:
            board_config: ChArUco board configuration
            images: List of (path, image) tuples
            recalibrate: If True, this is a re-calibration (use existing detections)
        """
        super().__init__()
        self.board_config = board_config
        self.images = images
        self.recalibrate = recalibrate
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        """Run calibration in background thread."""
        try:
            # Create calibrator
            calibrator = Calibrator(self.board_config)

            # Phase 1: Detect boards (0-70%)
            self.signals.progress.emit(0, "Detecting ChArUco boards...")

            def detection_progress(current, total, path):
                percent = int((current / total) * 70)
                import os
                filename = os.path.basename(path)
                self.signals.progress.emit(percent, f"Detecting: {filename} ({current}/{total})")

            detections = calibrator.detect_boards(self.images, detection_progress)

            # Check if we have enough detections
            valid_count = sum(1 for d in detections if d.has_detection)
            if valid_count < 3:
                self.signals.error.emit(
                    f"Insufficient detections: Found {valid_count} boards, need at least 3"
                )
                return

            # Phase 2: Calibration (70-90%)
            self.signals.progress.emit(70, "Computing camera calibration...")

            image_size = ImageLoader.get_image_size(self.images)
            result = calibrator.calibrate(detections, image_size)

            # Phase 3: Metrics (90-100%)
            self.signals.progress.emit(90, "Computing quality metrics...")

            # Metrics already computed in calibrate()

            # Done
            self.signals.progress.emit(100, "Calibration complete")
            self.signals.finished.emit(result)

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.signals.error.emit(error_msg)
