"""Main entry point for ChArUco calibration application."""

import argparse
import json
import sys
from pathlib import Path
import numpy as np

from .core import BoardConfig, Calibrator
from .core.calibration_options import CalibrationOptions
from .utils import ImageLoader
from .cli import CLIFormatter


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='ChArUco Camera Calibration Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # GUI mode (interactive)
  python -m chalibrate.main

  # GUI mode with pre-loaded images
  python -m chalibrate.main --images-dir ./images --squares-x 5 --squares-y 7 --square-length 30 --marker-length 20

  # CLI mode (headless)
  python -m chalibrate.main --images-dir ./images --squares-x 5 --squares-y 7 --square-length 30 --marker-length 20 --no-gui --output calib.json
        """
    )

    # Image input (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        '--images-dir',
        type=str,
        help='Directory containing calibration images'
    )
    input_group.add_argument(
        '--images',
        nargs='+',
        type=str,
        help='Individual image files'
    )

    # Board configuration
    parser.add_argument(
        '--squares-x',
        type=int,
        help='Number of squares in X direction'
    )
    parser.add_argument(
        '--squares-y',
        type=int,
        help='Number of squares in Y direction'
    )
    parser.add_argument(
        '--square-length',
        type=float,
        help='Square side length in mm'
    )
    parser.add_argument(
        '--marker-length',
        type=float,
        help='ArUco marker side length in mm'
    )
    parser.add_argument(
        '--dict',
        type=str,
        default='DICT_6X6_250',
        help='ArUco dictionary name (default: DICT_6X6_250)'
    )

    # Mode selection
    parser.add_argument(
        '--no-gui',
        action='store_true',
        help='Run in CLI mode without GUI'
    )
    parser.add_argument(
        '--auto-detect',
        action='store_true',
        help='Automatically detect board configuration from images'
    )

    # Output options
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (.json or .npz)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimal output (CLI mode only)'
    )

    # Advanced calibration options
    advanced_group = parser.add_argument_group('Advanced Calibration Options')

    # Preset
    advanced_group.add_argument(
        '--preset',
        type=str,
        choices=['default', 'high_accuracy', 'fast', 'fisheye', 'webcam'],
        help='Use a preset configuration (overrides other advanced options)'
    )

    # Subpixel refinement
    advanced_group.add_argument(
        '--no-subpixel',
        action='store_true',
        help='Disable subpixel corner refinement'
    )

    # Quality filtering
    advanced_group.add_argument(
        '--no-quality-filter',
        action='store_true',
        help='Disable automatic image quality filtering'
    )
    advanced_group.add_argument(
        '--blur-threshold',
        type=float,
        help='Blur detection threshold (default: 30.0, lower = more permissive)'
    )
    advanced_group.add_argument(
        '--brightness-min',
        type=float,
        help='Minimum brightness (default: 30.0)'
    )
    advanced_group.add_argument(
        '--brightness-max',
        type=float,
        help='Maximum brightness (default: 225.0)'
    )

    # Calibration flags
    advanced_group.add_argument(
        '--rational-model',
        action='store_true',
        help='Use rational distortion model (K4-K6)'
    )
    advanced_group.add_argument(
        '--thin-prism',
        action='store_true',
        help='Use thin prism distortion model'
    )
    advanced_group.add_argument(
        '--fix-principal-point',
        action='store_true',
        help='Fix principal point at image center'
    )
    advanced_group.add_argument(
        '--fix-aspect-ratio',
        action='store_true',
        help='Fix aspect ratio (fx/fy = 1)'
    )

    # Iterative refinement
    advanced_group.add_argument(
        '--iterative-refinement',
        action='store_true',
        help='Enable iterative outlier removal'
    )
    advanced_group.add_argument(
        '--max-iterations',
        type=int,
        help='Maximum refinement iterations (default: 3)'
    )
    advanced_group.add_argument(
        '--outlier-percentile',
        type=float,
        help='Outlier percentile threshold (default: 95.0)'
    )

    # Preprocessing
    advanced_group.add_argument(
        '--clahe',
        action='store_true',
        help='Enable CLAHE preprocessing'
    )

    return parser.parse_args()


def validate_cli_args(args):
    """Validate arguments for CLI mode.

    Args:
        args: Parsed arguments

    Returns:
        True if valid, False otherwise
    """
    if not args.images_dir and not args.images:
        CLIFormatter.print_error("--images-dir or --images required for CLI mode")
        return False

    # If auto-detect is enabled, board params are optional
    if args.auto_detect:
        return True

    required_board_params = [
        ('--squares-x', args.squares_x),
        ('--squares-y', args.squares_y),
        ('--square-length', args.square_length),
        ('--marker-length', args.marker_length),
    ]

    missing = [name for name, value in required_board_params if value is None]
    if missing:
        CLIFormatter.print_error(
            f"Missing required arguments for CLI mode: {', '.join(missing)}\n"
            f"Use --auto-detect to automatically detect board configuration"
        )
        return False

    return True


def run_cli(args):
    """Run calibration in CLI mode.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Validate arguments
        if not validate_cli_args(args):
            return 1

        # Load images
        if not args.quiet:
            print("Loading images...")

        if args.images_dir:
            images = ImageLoader.load_directory(args.images_dir)
            if not args.quiet:
                print(f"Loaded {len(images)} images from {args.images_dir}")
        else:
            images = ImageLoader.load_files(args.images)
            if not args.quiet:
                print(f"Loaded {len(images)} images")

        # Create board configuration
        if args.auto_detect:
            if not args.quiet:
                print("\nAuto-detecting board configuration...")

            from .core import auto_detect_board
            board_config, message = auto_detect_board(images, max_images=5)

            if not board_config:
                CLIFormatter.print_error(message)
                return 1

            if not args.quiet:
                print(message)
        else:
            board_config = BoardConfig(
                squares_x=args.squares_x,
                squares_y=args.squares_y,
                square_length=args.square_length,
                marker_length=args.marker_length,
                dict_name=args.dict,
            )

        # Create calibration options
        if args.preset:
            calibration_options = CalibrationOptions.from_preset(args.preset)
            if not args.quiet:
                print(f"Using preset: {args.preset}")
        else:
            calibration_options = CalibrationOptions()

            # Apply individual options
            if args.no_subpixel:
                calibration_options.enable_subpixel = False
            if args.no_quality_filter:
                calibration_options.enable_quality_filter = False
            if args.blur_threshold is not None:
                calibration_options.blur_threshold = args.blur_threshold
            if args.brightness_min is not None:
                calibration_options.brightness_min = args.brightness_min
            if args.brightness_max is not None:
                calibration_options.brightness_max = args.brightness_max
            if args.rational_model:
                calibration_options.use_rational_model = True
            if args.thin_prism:
                calibration_options.use_thin_prism = True
            if args.fix_principal_point:
                calibration_options.fix_principal_point = True
            if args.fix_aspect_ratio:
                calibration_options.fix_aspect_ratio = True
            if args.iterative_refinement:
                calibration_options.enable_iterative_refinement = True
            if args.max_iterations is not None:
                calibration_options.max_refinement_iterations = args.max_iterations
            if args.outlier_percentile is not None:
                calibration_options.outlier_percentile = args.outlier_percentile
            if args.clahe:
                calibration_options.enable_clahe = True

        # Print options summary if not quiet
        if not args.quiet:
            CLIFormatter.print_options_summary(calibration_options)

        # Create calibrator
        calibrator = Calibrator(board_config, calibration_options)

        # Detect boards
        if not args.quiet:
            print("\nDetecting ChArUco boards...")

        def progress_callback(current, total, path):
            if not args.quiet:
                CLIFormatter.print_progress(current, total, f"Processing {Path(path).name}")

        detections = calibrator.detect_boards(images, progress_callback)

        # Print detection summary
        if not args.quiet:
            CLIFormatter.print_detection_summary(detections)
            if calibration_options.enable_quality_filter:
                CLIFormatter.print_quality_summary(detections)

        # Perform calibration
        if not args.quiet:
            print("\nComputing camera calibration...")

        image_size = ImageLoader.get_image_size(images)

        # Use iterative refinement if enabled
        if calibration_options.enable_iterative_refinement:
            result = calibrator.calibrate_with_refinement(detections, image_size)
        else:
            result = calibrator.calibrate(detections, image_size)

        # Print results
        CLIFormatter.print_results(result, quiet=args.quiet)

        # Save output if requested
        if args.output:
            output_path = Path(args.output)

            if output_path.suffix == '.json':
                # Save as JSON
                with open(output_path, 'w') as f:
                    json.dump(result.to_dict(), f, indent=2)
                if not args.quiet:
                    print(f"Calibration saved to: {output_path}")

            elif output_path.suffix == '.npz':
                # Save as NumPy
                np.savez(
                    output_path,
                    camera_matrix=result.camera_matrix,
                    dist_coeffs=result.dist_coeffs,
                    rms_error=result.rms_error,
                    image_size=np.array(result.image_size)
                )
                if not args.quiet:
                    print(f"Calibration saved to: {output_path}")

            else:
                CLIFormatter.print_warning(f"Unknown output format: {output_path.suffix}, saving as JSON")
                with open(output_path.with_suffix('.json'), 'w') as f:
                    json.dump(result.to_dict(), f, indent=2)

        return 0

    except Exception as e:
        CLIFormatter.print_error(str(e))
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


def run_gui(args):
    """Run calibration in GUI mode.

    Args:
        args: Parsed command-line arguments (may contain pre-loaded config)

    Returns:
        Exit code (0 for success, 1 for error)
    """
    from PyQt6.QtWidgets import QApplication
    from .gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("ChArUco Calibration")

    # Prepare pre-loaded configuration
    preload_config = None
    preload_images = None

    # Load images if provided
    if args.images_dir or args.images:
        try:
            if args.images_dir:
                preload_images = ImageLoader.load_directory(args.images_dir)
            else:
                preload_images = ImageLoader.load_files(args.images)
        except Exception as e:
            CLIFormatter.print_error(f"Failed to load images: {e}")
            return 1

    # Load board config if parameters provided, or auto-detect
    if args.auto_detect and preload_images:
        try:
            from .core import auto_detect_board
            preload_config, message = auto_detect_board(preload_images, max_images=5)
            if preload_config:
                print(f"Auto-detected board configuration: "
                      f"{preload_config.squares_x}x{preload_config.squares_y}")
            else:
                CLIFormatter.print_warning("Auto-detection failed, will prompt for configuration")
        except Exception as e:
            CLIFormatter.print_warning(f"Auto-detection error: {e}")
    elif all([args.squares_x, args.squares_y, args.square_length, args.marker_length]):
        try:
            preload_config = BoardConfig(
                squares_x=args.squares_x,
                squares_y=args.squares_y,
                square_length=args.square_length,
                marker_length=args.marker_length,
                dict_name=args.dict,
            )
        except Exception as e:
            CLIFormatter.print_error(f"Invalid board configuration: {e}")
            return 1

    # Create and show main window
    window = MainWindow(
        preload_config=preload_config,
        preload_images=preload_images
    )
    window.show()

    return app.exec()


def main():
    """Main entry point."""
    args = parse_args()

    if args.no_gui:
        # CLI mode
        sys.exit(run_cli(args))
    else:
        # GUI mode
        sys.exit(run_gui(args))


if __name__ == '__main__':
    main()
