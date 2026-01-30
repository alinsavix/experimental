# Audio Frequency Analyzer

A Python script that analyzes audio files for high-frequency spikes above 17kHz.

## Features

- **Audio Segmentation**: Splits audio/video files into 1-minute segments (audio-only output)
- **Smart Caching**: Skips re-splitting if segments already exist
- **High-Resolution FFT**: Performs detailed frequency analysis on each segment
- **Visualization**: Generates FFT plots for each segment with full spectrum and high-frequency zoom
- **Spike Detection**: Identifies and measures amplitude of spikes above 17kHz
- **Comprehensive Reporting**: Creates JSON report with all detected spikes

## Requirements

- Python 3.7+
- FFmpeg (must be installed and available in PATH)
- Python packages (install via requirements.txt):
  - numpy
  - scipy
  - matplotlib

## Installation

1. Install FFmpeg:
   - Windows: Download from https://ffmpeg.org/ or use `winget install ffmpeg`
   - Linux: `sudo apt install ffmpeg`
   - Mac: `brew install ffmpeg`

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Basic usage:
```bash
python audio_frequency_analyzer.py <input_file>
```

With custom output directory:
```bash
python audio_frequency_analyzer.py <input_file> <output_dir>
```

With custom spike detection threshold:
```bash
python audio_frequency_analyzer.py <input_file> <output_dir> <threshold_db>
```

### Example

```bash
python audio_frequency_analyzer.py testaudio/testfile.mp4
```

This will:
1. Create an `output_segments` directory
2. Split the file into 1-minute WAV segments
3. Generate FFT plots for each segment
4. Create a `spike_report.json` with detected spikes

## Output

The script creates:
- **WAV segments**: `segment_0000.wav`, `segment_0001.wav`, etc.
- **FFT plots**: `segment_0000_fft.png`, `segment_0001_fft.png`, etc.
- **JSON report**: `spike_report.json` containing all detected spikes

### Sample JSON Report

```json
{
  "input_file": "testaudio/testfile.mp4",
  "spike_threshold_db": -40,
  "total_segments": 2,
  "spikes": [
    {
      "segment": "segment_0002.wav",
      "frequency_hz": 18500,
      "frequency_khz": 18.5,
      "amplitude_db": -25.3
    }
  ]
}
```

## Parameters

- **spike_threshold_db**: Threshold in dB for spike detection (default: -40)
  - Lower values = more sensitive (detect quieter spikes)
  - Higher values = less sensitive (only detect louder spikes)

## Technical Details

- **Sample Rate**: 48kHz (allows detection up to 24kHz)
- **FFT Size**: 262,144 points (2^18) for very high frequency resolution
- **Audio Format**: PCM 16-bit stereo WAV files
- **Analysis Method**: Welch's method for improved frequency resolution
