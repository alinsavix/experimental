"""Image loading and preprocessing."""

import os
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np
from PIL import Image


class ImageLoader:
    """Load and preprocess images for calibration."""

    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    @staticmethod
    def load_directory(directory: str) -> List[Tuple[str, np.ndarray]]:
        """Load all images from a directory.

        Args:
            directory: Path to directory containing images

        Returns:
            List of (path, image) tuples

        Raises:
            ValueError: If directory doesn't exist or contains no valid images
        """
        dir_path = Path(directory)

        if not dir_path.exists():
            raise ValueError(f"Directory does not exist: {directory}")

        if not dir_path.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        # Find all image files
        image_files = []
        for ext in ImageLoader.SUPPORTED_FORMATS:
            image_files.extend(dir_path.glob(f"*{ext}"))
            image_files.extend(dir_path.glob(f"*{ext.upper()}"))

        if not image_files:
            raise ValueError(f"No images found in directory: {directory}")

        # Load images
        images = []
        for img_path in sorted(image_files):
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    images.append((str(img_path), img))
            except Exception as e:
                print(f"Warning: Failed to load {img_path}: {e}")

        if not images:
            raise ValueError(f"No valid images could be loaded from: {directory}")

        # Verify all images have same size
        ImageLoader.verify_image_sizes(images)

        return images

    @staticmethod
    def load_files(file_paths: List[str]) -> List[Tuple[str, np.ndarray]]:
        """Load specific image files.

        Args:
            file_paths: List of image file paths

        Returns:
            List of (path, image) tuples

        Raises:
            ValueError: If no valid images could be loaded
        """
        images = []

        for path in file_paths:
            if not os.path.exists(path):
                print(f"Warning: File not found: {path}")
                continue

            try:
                img = cv2.imread(path)
                if img is not None:
                    images.append((path, img))
                else:
                    print(f"Warning: Failed to load {path}")
            except Exception as e:
                print(f"Warning: Error loading {path}: {e}")

        if not images:
            raise ValueError("No valid images could be loaded")

        # Verify all images have same size
        ImageLoader.verify_image_sizes(images)

        return images

    @staticmethod
    def verify_image_sizes(images: List[Tuple[str, np.ndarray]]) -> None:
        """Verify all images have compatible dimensions.

        Allows mixed orientations (portrait/landscape) as long as the
        resolution matches when normalized.

        Args:
            images: List of (path, image) tuples

        Raises:
            ValueError: If images have inconsistent sizes
        """
        if not images:
            return

        # Get reference resolution (sorted to normalize orientation)
        reference_shape = images[0][1].shape[:2]
        reference_resolution = tuple(sorted(reference_shape))

        for path, img in images[1:]:
            current_shape = img.shape[:2]
            current_resolution = tuple(sorted(current_shape))

            if current_resolution != reference_resolution:
                raise ValueError(
                    f"Image resolution mismatch: {path} is {current_shape}, "
                    f"expected resolution {reference_resolution} (any orientation)"
                )

    @staticmethod
    def create_thumbnail(image: np.ndarray, size: int = 200) -> np.ndarray:
        """Create thumbnail of image for GUI display.

        Args:
            image: Input image (BGR format)
            size: Maximum dimension of thumbnail

        Returns:
            Thumbnail image (BGR format)
        """
        h, w = image.shape[:2]

        # Calculate scaling factor
        scale = size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize using high-quality interpolation
        thumbnail = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return thumbnail

    @staticmethod
    def get_image_size(images: List[Tuple[str, np.ndarray]]) -> Tuple[int, int]:
        """Get image size from loaded images.

        Note: With mixed orientations, uses the most common dimensions.

        Args:
            images: List of (path, image) tuples

        Returns:
            (width, height) tuple
        """
        if not images:
            raise ValueError("No images provided")

        # Count frequency of each dimension
        from collections import Counter
        dimensions = Counter((img.shape[1], img.shape[0]) for _, img in images)

        # Use most common dimensions
        most_common_dims, _ = dimensions.most_common(1)[0]
        return most_common_dims
