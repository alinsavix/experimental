# ChArUco Camera Calibration Application

A professional Python application for camera/lens calibration using ChArUco (Chessboard + ArUco markers) boards. Features both a graphical user interface (GUI) and command-line interface (CLI) for flexible usage.

## Features

- **Complete Camera Calibration**: Calculate camera intrinsics (fx, fy, cx, cy) and distortion coefficients (K1-K4 radial, P1-P2 tangential)
- **Automatic Board Detection**: Auto-detect board configuration (dictionary, dimensions) from images
- **Interactive GUI**:
  - Scrollable image grid with quality indicators
  - Color-coded borders showing calibration quality
  - Full-size image viewer with detection overlay
  - Undistortion preview with before/after comparison
  - Image exclusion for iterative quality improvement
  - Auto-detect button in configuration dialog
- **CLI Mode**: Headless operation for automation and scripting with `--auto-detect` support
- **Multiple Export Formats**: JSON (human-readable) and NumPy .npz (efficient)
- **Quality Metrics**: Per-image reprojection errors and overall RMS error
- **Visualization**: View detected ChArUco corners and ArUco markers
- **Mixed Orientations**: Handles portrait/landscape images from same camera

## Installation

### Prerequisites

- Python 3.8 or higher
- `uv` (recommended) or `pip` for package management

### Setup with `uv` (Recommended)

`uv` is a fast Python package manager that's 10-100x faster than pip.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone or navigate to the project directory
cd calibrate

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows

# Install dependencies
uv pip install -r requirements.txt
```

### Setup with pip

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### GUI Mode (Default)

#### Interactive Launch

Start the application without arguments for fully interactive mode:

```bash
# Using uv (no need to activate venv)
uv run calibrate.py

# Or after activating venv
python calibrate.py
```

**Workflow:**
1. Load images from directory
2. Click "🔍 Auto-Detect from Images" (or configure manually)
3. Verify detected configuration
4. Click "Run Calibration"
5. View results with color-coded quality indicators
6. Optionally exclude poor images and re-calibrate
7. Export results to JSON or NumPy format

#### Pre-loaded Launch with Auto-Detection

Start with images pre-loaded and auto-detect configuration:

```bash
# Auto-detect board configuration from images
uv run calibrate.py \
  --images-dir ./calibration_images \
  --auto-detect
```

#### Pre-loaded Launch with Manual Configuration

Or specify configuration explicitly:

```bash
uv run calibrate.py \
  --images-dir ./calibration_images \
  --squares-x 5 \
  --squares-y 7 \
  --square-length 30 \
  --marker-length 20
```

The GUI will open with images already loaded and board configured.

### CLI Mode (Headless)

Run calibration from command line without GUI:

```bash
# With auto-detection
uv run calibrate.py \
  --images-dir ./calibration_images \
  --auto-detect \
  --no-gui \
  --output calibration.json

# Or with manual configuration
uv run calibrate.py \
  --images-dir ./calibration_images \
  --squares-x 5 \
  --squares-y 7 \
  --square-length 30 \
  --marker-length 20 \
  --no-gui \
  --output calibration.json
```

#### CLI Options

**Required for CLI mode:**
- `--images-dir DIR` or `--images FILE1 FILE2 ...` - Input images

**Board configuration (choose one):**
- `--auto-detect` - Automatically detect board from images
- OR manually specify:
  - `--squares-x N` - Number of squares in X direction
  - `--squares-y N` - Number of squares in Y direction
  - `--square-length MM` - Square side length in mm
  - `--marker-length MM` - Marker side length in mm

**Optional:**
- `--no-gui` - Run in CLI mode (required for headless)
- `--output FILE` - Save to file (.json or .npz)
- `--quiet` - Minimal output
- `--dict NAME` - ArUco dictionary (default: DICT_6X6_250)

#### CLI Examples

**Basic calibration with JSON output:**
```bash
uv run calibrate.py \
  --images-dir ./images \
  --squares-x 5 --squares-y 7 \
  --square-length 30 --marker-length 20 \
  --no-gui \
  --output calib.json
```

**Using specific image files:**
```bash
uv run calibrate.py \
  --images img1.jpg img2.jpg img3.jpg \
  --squares-x 5 --squares-y 7 \
  --square-length 30 --marker-length 20 \
  --no-gui
```

**NumPy output format:**
```bash
uv run calibrate.py \
  --images-dir ./images \
  --squares-x 5 --squares-y 7 \
  --square-length 30 --marker-length 20 \
  --no-gui \
  --output calibration.npz
```

**Quiet mode (minimal output):**
```bash
uv run calibrate.py \
  --images-dir ./images \
  --squares-x 5 --squares-y 7 \
  --square-length 30 --marker-length 20 \
  --no-gui \
  --quiet
```

## ChArUco Board Configuration

The application requires a ChArUco calibration board configuration:

- **Squares X/Y**: Number of chessboard squares (e.g., 5×7)
- **Square Length**: Physical size of chessboard squares in mm (e.g., 30mm)
- **Marker Length**: Physical size of ArUco markers in mm (e.g., 20mm)
- **ArUco Dictionary**: Marker set to use (default: DICT_6X6_250)

**Note:** Marker length must be less than square length.

### Auto-Detection

Use the `--auto-detect` flag or "🔍 Auto-Detect from Images" button to automatically determine board configuration from your images. See [AUTO_DETECTION.md](AUTO_DETECTION.md) for details.

⚠️ **Important**: Auto-detection estimates physical sizes using defaults (30mm squares, 20mm markers). Always verify these match your actual board by measuring with a ruler.

### Creating a ChArUco Board

Use the included board generator script:

```bash
# Generate default 5x7 board
uv run generate_board.py

# Custom board size
uv run generate_board.py \
  --squares-x 6 \
  --squares-y 8 \
  --square-length 25 \
  --marker-length 18 \
  --output my_board.png

# Letter paper instead of A4
uv run generate_board.py --paper Letter
```

**Printing instructions:**
1. Print the generated image at **actual size** (100% scale, no fit-to-page)
2. Measure the printed squares with a ruler to verify the specified size
3. Mount on a **flat, rigid surface** (foam board, cardboard, acrylic)
4. Use the same configuration values when calibrating

## Output Formats

### JSON Format

Human-readable format suitable for inspection and integration:

```json
{
  "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "distortion_coefficients": [K1, K2, P1, P2, K3, K4],
  "rms_error": 0.42,
  "image_size": [1920, 1080],
  "focal_length": {"fx": 1245.67, "fy": 1248.23},
  "principal_point": {"cx": 960.12, "cy": 540.45},
  "distortion": {
    "K1": -0.2345, "K2": 0.0567,
    "P1": 0.0012, "P2": -0.0034,
    "K3": -0.0123, "K4": 0.0089
  }
}
```

### NumPy Format

Efficient binary format for Python applications:

```python
import numpy as np

# Load calibration
data = np.load('calibration.npz')
camera_matrix = data['camera_matrix']
dist_coeffs = data['dist_coeffs']
rms_error = data['rms_error']

# Use for undistortion
undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
```

## Quality Metrics

Images are classified by reprojection error:

- **Excellent** (<0.3px): Green border - Very accurate detection
- **Good** (0.3-0.5px): Dark green border - Good detection
- **Acceptable** (0.5-1.0px): Yellow border - Acceptable quality
- **Poor** (1.0-2.0px): Orange border - Consider excluding
- **Bad** (>2.0px): Red border - Should exclude or retake

## Tips for Best Results

1. **Take 15-25 images** from different angles and distances
2. **Vary the board orientation** - tilts, rotations, corners of frame
3. **Ensure good lighting** - avoid glare and shadows on the board
4. **Keep images sharp** - avoid motion blur
5. **Fill the frame** - board should be clearly visible but vary in size
6. **Use the exclusion feature** - remove poor quality images and re-calibrate
7. **Target RMS error < 0.5 pixels** for good calibration

## GUI Features

### Image Grid

- Displays all loaded images as thumbnails
- Color-coded borders indicate calibration quality
- Right-click to exclude/include images from calibration
- Click to view full-size with detection overlay

### Detection Viewer

- Shows detected ChArUco corners (green circles with IDs)
- Displays ArUco marker boundaries
- Tabs for Original, Detection, and Undistorted views

### Undistortion Preview

- Side-by-side comparison of original and corrected images
- Browse through all calibration images
- Export all undistorted images to directory

### Quality Panel

- Camera intrinsics with **principal point prominently displayed**
- Complete distortion coefficients (K1-K4, P1-P2)
- RMS reprojection error
- Per-image quality statistics
- Export buttons for JSON and NumPy formats

## Troubleshooting

### "No boards detected"

- Verify board parameters match your physical board
- Ensure good image quality (focus, lighting)
- Check that the entire board is visible in images
- Try different ArUco dictionary if using custom board

### "Insufficient detections"

- Need at least 3 images with detected boards
- Verify images contain visible ChArUco boards
- Check board configuration parameters

### High reprojection error (>1.0px)

- Exclude poor quality images (red/orange borders)
- Ensure camera didn't change focus during capture
- Verify physical board is flat and accurately printed
- Take more images from varied angles

### Different image resolutions error

- All images must be the same resolution
- Ensure no images are rotated or cropped
- Use consistent camera settings for all captures

## Requirements

- Python 3.8+
- opencv-contrib-python >= 4.8.0
- PyQt6 >= 6.6.0
- numpy >= 1.24.0
- Pillow >= 10.0.0

## License

This project is provided as-is for camera calibration purposes.

## Contributing

Issues and improvements are welcome. Please test thoroughly before submitting changes.
