#!/usr/bin/env python3
"""
Deep diagnostic - tries MANY more combinations and shows what's happening
"""

import sys
import cv2
import numpy as np

def test_board_config(gray, marker_corners, marker_ids, dictionary, dict_name,
                     squares_x, squares_y, square_len, marker_len):
    """Test a specific board configuration."""
    try:
        board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y),
            square_len,
            marker_len,
            dictionary
        )

        # Get the board's expected corners and IDs
        board_obj_points = board.getChessboardCorners()

        result = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, board
        )

        num_corners = result[0] if result[0] else 0

        if num_corners and num_corners > 0:
            return {
                'success': True,
                'num_corners': num_corners,
                'corners': result[1],
                'ids': result[2],
                'board': board,
                'config': f"{dict_name} {squares_x}x{squares_y} sq={square_len} mk={marker_len}"
            }
    except Exception as e:
        pass

    return None


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python deep_diagnose.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("="*80)
    print("DEEP DIAGNOSTIC - Testing exhaustive combinations")
    print("="*80)
    print()

    # Test all dictionary types
    dict_types = [
        ('DICT_4X4_50', cv2.aruco.DICT_4X4_50),
        ('DICT_4X4_100', cv2.aruco.DICT_4X4_100),
        ('DICT_4X4_250', cv2.aruco.DICT_4X4_250),
        ('DICT_5X5_50', cv2.aruco.DICT_5X5_50),
        ('DICT_5X5_100', cv2.aruco.DICT_5X5_100),
        ('DICT_5X5_250', cv2.aruco.DICT_5X5_250),
        ('DICT_6X6_50', cv2.aruco.DICT_6X6_50),
        ('DICT_6X6_100', cv2.aruco.DICT_6X6_100),
        ('DICT_6X6_250', cv2.aruco.DICT_6X6_250),
        ('DICT_7X7_50', cv2.aruco.DICT_7X7_50),
        ('DICT_7X7_100', cv2.aruco.DICT_7X7_100),
        ('DICT_7X7_250', cv2.aruco.DICT_7X7_250),
    ]

    # All board dimensions to try (based on common factorizations)
    board_sizes = [
        (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (3, 11),
        (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10),
        (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10),
        (6, 6), (6, 7), (6, 8), (6, 9), (6, 10),
        (7, 7), (7, 8), (7, 9), (7, 10),
        (8, 8), (8, 9), (8, 10),
        (9, 9), (9, 10),
        (10, 10),
    ]

    # Physical sizes to try (in mm)
    # Format: (square_length, marker_length, description)
    physical_sizes = [
        # Common ratios
        (30.0, 20.0, "3:2 ratio"),
        (40.0, 30.0, "4:3 ratio"),
        (50.0, 40.0, "5:4 ratio"),
        (60.0, 45.0, "4:3 ratio"),
        (25.0, 20.0, "5:4 ratio"),
        (30.0, 24.0, "5:4 ratio"),
        (35.0, 28.0, "5:4 ratio"),

        # Different ratios
        (30.0, 15.0, "2:1 ratio"),
        (40.0, 20.0, "2:1 ratio"),
        (30.0, 18.0, "5:3 ratio"),
        (40.0, 24.0, "5:3 ratio"),
        (30.0, 22.5, "4:3 ratio"),
        (45.0, 30.0, "3:2 ratio"),

        # More variations
        (20.0, 15.0, "4:3 ratio"),
        (25.0, 18.75, "4:3 ratio"),
        (35.0, 26.25, "4:3 ratio"),
        (50.0, 37.5, "4:3 ratio"),

        # Edge cases
        (100.0, 75.0, "4:3 ratio"),
        (10.0, 7.5, "4:3 ratio"),
        (30.0, 25.0, "6:5 ratio"),
        (40.0, 35.0, "8:7 ratio"),
    ]

    successful_configs = []
    total_tests = 0

    for dict_name, dict_id in dict_types:
        dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

        marker_corners, marker_ids, _ = detector.detectMarkers(gray)

        if marker_ids is None or len(marker_ids) == 0:
            continue

        print(f"\n{dict_name}: Detected {len(marker_ids)} markers")
        print(f"  IDs: {sorted(marker_ids.flatten())}")

        # Try all combinations for this dictionary
        for squares_x, squares_y in board_sizes:
            for square_len, marker_len, ratio_desc in physical_sizes:
                total_tests += 1

                result = test_board_config(
                    gray, marker_corners, marker_ids, dictionary, dict_name,
                    squares_x, squares_y, square_len, marker_len
                )

                if result:
                    successful_configs.append(result)
                    print(f"  ✓ FOUND: {squares_x}x{squares_y}, {ratio_desc}, {result['num_corners']} corners")

    print("\n" + "="*80)
    print(f"RESULTS: Tested {total_tests} combinations")
    print("="*80)

    if successful_configs:
        print(f"\n✓ SUCCESS! Found {len(successful_configs)} working configurations:\n")

        # Sort by number of corners (descending)
        successful_configs.sort(key=lambda x: x['num_corners'], reverse=True)

        for i, cfg in enumerate(successful_configs[:10], 1):  # Show top 10
            print(f"{i}. {cfg['config']}")
            print(f"   → {cfg['num_corners']} corners detected")
            print()

        # Use the best one for visualization
        best = successful_configs[0]

        print("\n" + "="*80)
        print("BEST CONFIGURATION:")
        print("="*80)
        print(best['config'])
        print()

        # Create visualization
        vis_img = img.copy()

        # Draw markers
        cv2.aruco.drawDetectedMarkers(vis_img, marker_corners, marker_ids)

        # Draw ChArUco corners
        for i, corner in enumerate(best['corners']):
            corner_id = best['ids'][i][0]
            x, y = int(corner[0][0]), int(corner[0][1])
            cv2.circle(vis_img, (x, y), 12, (0, 255, 0), -1)
            cv2.putText(vis_img, str(corner_id), (x+15, y+15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Save
        cv2.imwrite('deep_diagnosis_SUCCESS.jpg', vis_img)

        # Also save a thumbnail
        scale = 1200 / max(vis_img.shape[:2])
        thumb = cv2.resize(vis_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite('deep_diagnosis_SUCCESS_thumb.jpg', thumb)

        print("✓ Saved visualizations:")
        print("  - deep_diagnosis_SUCCESS.jpg")
        print("  - deep_diagnosis_SUCCESS_thumb.jpg")

        # Extract config for command line
        parts = best['config'].split()
        dict_name = parts[0]
        size = parts[1].split('x')
        squares_x, squares_y = int(size[0]), int(size[1].split()[0])

        # Find sq= and mk= values
        for part in parts:
            if part.startswith('sq='):
                square_len = float(part[3:])
            elif part.startswith('mk='):
                marker_len = float(part[3:])

        print("\n" + "="*80)
        print("USE THIS COMMAND:")
        print("="*80)
        print(f"uv run python calibrate.py \\")
        print(f"  --images-dir ./board_images \\")
        print(f"  --squares-x {squares_x} \\")
        print(f"  --squares-y {squares_y} \\")
        print(f"  --square-length {square_len} \\")
        print(f"  --marker-length {marker_len} \\")
        print(f"  --dict {dict_name}")

    else:
        print("\n❌ NO WORKING CONFIGURATIONS FOUND")
        print("\nThis is very unusual. Possible explanations:")
        print("1. The board uses a custom/proprietary format")
        print("2. The board was created with very old OpenCV version")
        print("3. The board has non-standard marker placement")
        print("4. There's a bug in the detection code")
        print("\nPlease provide:")
        print("- How was your board generated?")
        print("- What code/tool created it?")
        print("- Board generation parameters?")
