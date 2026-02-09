"""Background worker threads for calibration."""

from typing import List, Tuple
from PyQt6.QtCore import QObject, QRunnable, pyqtSignal, pyqtSlot
import numpy as np

from ..core import BoardConfig, Calibrator, CalibrationResult
from ..core.calibration_options import CalibrationOptions
from ..utils import ImageLoader


class WorkerSignals(QObject):
    """Signals for calibration worker."""

    progress = pyqtSignal(int, str)  # (percentage, status_message)
    finished = pyqtSignal(object)  # CalibrationResult
    error = pyqtSignal(str)  # Error message
    detections_ready = pyqtSignal(list)  # List of ImageDetection (after detection, before calibration)


class CalibrationWorker(QRunnable):
    """Worker thread for running calibration in background."""

    def __init__(
        self,
        board_config: BoardConfig,
        images: List[Tuple[str, np.ndarray]],
        calibration_options: CalibrationOptions = None,
        recalibrate: bool = False
    ):
        """Initialize calibration worker.

        Args:
            board_config: ChArUco board configuration
            images: List of (path, image) tuples
            calibration_options: Calibration accuracy options
            recalibrate: If True, this is a re-calibration (use existing detections)
        """
        super().__init__()
        self.board_config = board_config
        self.images = images
        self.calibration_options = calibration_options if calibration_options else CalibrationOptions()
        self.recalibrate = recalibrate
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        """Run calibration in background thread."""
        try:
            # Create calibrator
            calibrator = Calibrator(self.board_config, self.calibration_options)

            # Phase 1: Detect boards (0-70%)
            self.signals.progress.emit(0, "Detecting ChArUco boards...")

            def detection_progress(current, total, path):
                percent = int((current / total) * 70)
                import os
                filename = os.path.basename(path)
                self.signals.progress.emit(percent, f"Detecting: {filename} ({current}/{total})")

            detections = calibrator.detect_boards(self.images, detection_progress)

            # Emit detections with quality info (before calibration)
            self.signals.detections_ready.emit(detections)

            # Check if we have enough detections
            valid_count = sum(1 for d in detections if d.has_detection and not d.excluded)
            if valid_count < 3:
                # Provide detailed error message
                total_detected = sum(1 for d in detections if d.has_detection)
                auto_excluded_count = sum(1 for d in detections if d.auto_excluded)

                error_msg = f"Insufficient detections: Found {valid_count} valid boards, need at least 3\n\n"
                error_msg += f"Details:\n"
                error_msg += f"  - Total images: {len(detections)}\n"
                error_msg += f"  - Boards detected: {total_detected}\n"
                error_msg += f"  - Auto-excluded (quality): {auto_excluded_count}\n"

                if auto_excluded_count > 0:
                    error_msg += "\nQuality filtering excluded many images. Try:\n"
                    error_msg += "  - Disable quality filtering in Advanced Options\n"
                    error_msg += "  - Adjust blur threshold (lower = less strict)\n"
                    error_msg += "  - Adjust brightness thresholds\n\n"

                    # Show some example issues
                    error_msg += "Sample exclusion reasons:\n"
                    count = 0
                    for d in detections:
                        if d.auto_excluded and count < 3:
                            import os
                            filename = os.path.basename(d.image_path)
                            error_msg += f"  - {filename}: {d.exclusion_reason}\n"
                            count += 1

                self.signals.error.emit(error_msg)
                return

            # Phase 2: Calibration (70-90%)
            self.signals.progress.emit(70, "Computing camera calibration...")

            image_size = ImageLoader.get_image_size(self.images)

            # Use iterative refinement if enabled
            if self.calibration_options.enable_iterative_refinement:
                result = calibrator.calibrate_with_refinement(detections, image_size)
            else:
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
