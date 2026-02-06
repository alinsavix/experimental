"""CLI output formatting."""

from typing import List
from ..core import CalibrationResult, QualityMetrics, ImageQuality


class CLIFormatter:
    """Format calibration results for command-line output."""

    @staticmethod
    def print_progress(current: int, total: int, message: str = ""):
        """Print progress bar to stdout.

        Args:
            current: Current progress
            total: Total items
            message: Optional message to display
        """
        percent = int((current / total) * 100)
        bar_length = 40
        filled = int((current / total) * bar_length)
        bar = '=' * filled + ' ' * (bar_length - filled)

        print(f"\r[{bar}] {percent}% - {message}", end='', flush=True)

        if current == total:
            print()  # New line when complete

    @staticmethod
    def print_results(result: CalibrationResult, quiet: bool = False):
        """Print calibration results to stdout.

        Args:
            result: Calibration results
            quiet: If True, print minimal output
        """
        if quiet:
            # Minimal output mode
            print(f"RMS Error: {result.rms_error:.4f} pixels")
            print(f"fx={result.fx:.2f} fy={result.fy:.2f} cx={result.cx:.2f} cy={result.cy:.2f}")
            print(f"K1={result.k1:.6f} K2={result.k2:.6f} P1={result.p1:.6f} P2={result.p2:.6f}")
            return

        # Full output mode
        print("\n" + "=" * 60)
        print("Camera Calibration Results")
        print("=" * 60)

        print("\nCamera Matrix (K):")
        print(f"  fx: {result.fx:10.2f} pixels  (focal length X)")
        print(f"  fy: {result.fy:10.2f} pixels  (focal length Y)")
        print(f"  cx: {result.cx:10.2f} pixels  (principal point X)")
        print(f"  cy: {result.cy:10.2f} pixels  (principal point Y)")

        print("\nDistortion Coefficients:")
        print(f"  K1: {result.k1:10.6f}  (radial)")
        print(f"  K2: {result.k2:10.6f}  (radial)")
        print(f"  P1: {result.p1:10.6f}  (tangential)")
        print(f"  P2: {result.p2:10.6f}  (tangential)")
        print(f"  K3: {result.k3:10.6f}  (radial)")
        print(f"  K4: {result.k4:10.6f}  (radial)")

        print(f"\nOverall RMS Reprojection Error: {result.rms_error:.4f} pixels")

        # Quality statistics
        stats = QualityMetrics.get_statistics(result)

        print("\nPer-Image Quality:")
        total_calibrated = stats['calibrated_images']

        if total_calibrated > 0:
            counts = stats['quality_counts']
            print(f"  Excellent (<0.3px):     {counts['excellent']:3d} images ({counts.get('excellent_percent', 0):5.1f}%)")
            print(f"  Good (0.3-0.5px):       {counts['good']:3d} images ({counts.get('good_percent', 0):5.1f}%)")
            print(f"  Acceptable (0.5-1.0px): {counts['acceptable']:3d} images ({counts.get('acceptable_percent', 0):5.1f}%)")
            print(f"  Poor (1.0-2.0px):       {counts['poor']:3d} images ({counts.get('poor_percent', 0):5.1f}%)")
            print(f"  Bad (>2.0px):           {counts['bad']:3d} images ({counts.get('bad_percent', 0):5.1f}%)")
        else:
            print("  No images calibrated")

        if stats['excluded_images'] > 0:
            print(f"\n  Excluded from calibration: {stats['excluded_images']} images")

        print()

    @staticmethod
    def print_detection_summary(detections: List):
        """Print summary of board detections.

        Args:
            detections: List of ImageDetection objects
        """
        total = len(detections)
        detected = sum(1 for d in detections if d.has_detection)

        print(f"\nBoard Detection Summary:")
        print(f"  Total images: {total}")
        print(f"  Boards detected: {detected}")
        print(f"  Failed detections: {total - detected}")

        if detected > 0:
            avg_corners = sum(d.num_corners for d in detections if d.has_detection) / detected
            print(f"  Average corners per image: {avg_corners:.1f}")

    @staticmethod
    def print_error(message: str):
        """Print error message to stderr.

        Args:
            message: Error message
        """
        import sys
        print(f"ERROR: {message}", file=sys.stderr)

    @staticmethod
    def print_warning(message: str):
        """Print warning message to stderr.

        Args:
            message: Warning message
        """
        import sys
        print(f"WARNING: {message}", file=sys.stderr)
