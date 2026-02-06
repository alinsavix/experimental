#!/usr/bin/env python3
"""Generate a ChArUco calibration board for printing.

This script creates a printable ChArUco board image that can be used
for camera calibration.
"""

import argparse
import cv2


def main():
    parser = argparse.ArgumentParser(
        description='Generate ChArUco calibration board for printing'
    )
    parser.add_argument('--squares-x', type=int, default=5,
                       help='Number of squares in X direction (default: 5)')
    parser.add_argument('--squares-y', type=int, default=7,
                       help='Number of squares in Y direction (default: 7)')
    parser.add_argument('--square-length', type=float, default=30,
                       help='Square side length in mm (default: 30)')
    parser.add_argument('--marker-length', type=float, default=20,
                       help='Marker side length in mm (default: 20)')
    parser.add_argument('--dict', type=str, default='DICT_6X6_250',
                       help='ArUco dictionary (default: DICT_6X6_250)')
    parser.add_argument('--output', type=str, default='charuco_board.png',
                       help='Output filename (default: charuco_board.png)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for printing (default: 300)')
    parser.add_argument('--paper', type=str, default='A4',
                       choices=['A4', 'Letter'],
                       help='Paper size (default: A4)')

    args = parser.parse_args()

    # Paper sizes at specified DPI (width, height)
    paper_sizes = {
        'A4': (int(8.27 * args.dpi), int(11.69 * args.dpi)),  # 210mm x 297mm
        'Letter': (int(8.5 * args.dpi), int(11 * args.dpi)),   # 8.5" x 11"
    }

    width, height = paper_sizes[args.paper]

    # Create board
    dict_attr = getattr(cv2.aruco, args.dict)
    dictionary = cv2.aruco.getPredefinedDictionary(dict_attr)
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y),
        args.square_length,
        args.marker_length,
        dictionary
    )

    # Generate image
    # Note: generateImage returns the image directly, size is in pixels
    img = board.generateImage((width, height))

    # Save
    cv2.imwrite(args.output, img)

    print(f"ChArUco board generated: {args.output}")
    print(f"Configuration:")
    print(f"  Squares: {args.squares_x} x {args.squares_y}")
    print(f"  Square length: {args.square_length} mm")
    print(f"  Marker length: {args.marker_length} mm")
    print(f"  Dictionary: {args.dict}")
    print(f"  Paper size: {args.paper} at {args.dpi} DPI")
    print(f"\nPrint instructions:")
    print(f"  1. Print {args.output} at actual size (100% scale)")
    print(f"  2. Measure the printed squares to verify {args.square_length}mm size")
    print(f"  3. Mount on a flat, rigid surface (cardboard, foam board, etc.)")
    print(f"  4. Use for calibration with the same configuration values")


if __name__ == '__main__':
    main()
