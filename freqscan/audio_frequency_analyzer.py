#!/usr/bin/env python3
"""
Audio Frequency Analyzer
Splits audio files into 1-minute segments, performs FFT analysis,
and detects high-frequency spikes above 17kHz.
"""

import os
import sys
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for parallel plotting
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import wavfile
from scipy import signal
import json


class AudioFrequencyAnalyzer:
    def __init__(self, input_file, output_dir="output", spike_threshold_db=10, fft_size=65536, verbose=False, use_prominence=True):
        """
        Initialize the audio frequency analyzer.

        Args:
            input_file: Path to the input audio/video file
            output_dir: Base directory for output files
            spike_threshold_db: Minimum dB difference above surrounding frequencies to consider a spike
            fft_size: FFT size for frequency analysis (default: 65536)
            verbose: Show detailed per-segment output (default: False)
            use_prominence: Use prominence-based detection (True) or absolute amplitude (False) (default: True)
        """
        self.input_file = Path(input_file)
        self.spike_threshold_db = spike_threshold_db
        self.fft_size = fft_size
        self.segment_duration = 60  # seconds
        self.spikes_detected = []
        self.verbose = verbose
        self.use_prominence = use_prominence

        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Create base output directory structure based on input filename
        base_name = self.input_file.stem  # Get filename without extension
        base_output = Path(output_dir)

        self.segments_dir = base_output / f"{base_name}_segments"
        self.analysis_dir = base_output / f"{base_name}_analysis"

        # Create directories if they don't exist
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

    def get_audio_duration(self):
        """Get the duration of the input audio file using ffprobe."""
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(self.input_file)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except subprocess.CalledProcessError as e:
            print(f"Error getting audio duration: {e}")
            return None

    def split_audio(self):
        """Split the input file into 1-minute audio segments using a single ffmpeg call."""
        duration = self.get_audio_duration()
        if duration is None:
            print("Could not determine audio duration. Exiting.")
            return []

        print(f"Total duration: {duration:.2f} seconds")
        num_segments = int(np.ceil(duration / self.segment_duration))
        print(f"Number of segments to create: {num_segments}")

        # Check if all segments already exist
        base_name = self.input_file.stem
        segment_files = []
        all_exist = True
        for i in range(num_segments):
            output_file = self.segments_dir / f"{base_name}_segment_{i:04d}.wav"
            segment_files.append(output_file)
            if not output_file.exists():
                all_exist = False

        if all_exist:
            print("All segments already exist, skipping extraction.")
            return segment_files

        print(f"Splitting into {num_segments} segments with a single ffmpeg call...")

        # Use ffmpeg's segment muxer to split in one command
        # Output pattern with %04d for zero-padded numbering
        output_pattern = str(self.segments_dir / f"{base_name}_segment_%04d.wav")

        cmd = [
            'ffmpeg',
            '-i', str(self.input_file),
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', '48000',  # Sample rate 48kHz (good for high freq analysis)
            '-ac', '2',  # Stereo
            '-f', 'segment',  # Use segment muxer
            '-segment_time', str(self.segment_duration),  # Split every 60 seconds
            '-reset_timestamps', '1',  # Reset timestamps for each segment
            '-y',  # Overwrite output files if they exist
            output_pattern
        ]

        try:
            print("Running ffmpeg...")
            result = subprocess.run(cmd, capture_output=True, check=True, text=True)
            print(f"Successfully created {num_segments} segments")
        except subprocess.CalledProcessError as e:
            print(f"Error creating segments: {e}")
            if e.stderr:
                print(e.stderr)
            return []

        # Verify all expected segments were created
        segment_files = []
        base_name = self.input_file.stem
        for i in range(num_segments):
            output_file = self.segments_dir / f"{base_name}_segment_{i:04d}.wav"
            if output_file.exists():
                segment_files.append(output_file)
            else:
                print(f"Warning: Expected segment {output_file.name} was not created")

        return segment_files

    def perform_fft_analysis(self, audio_file):
        """
        Perform high-resolution FFT on an audio file.

        Returns:
            tuple: (frequencies, magnitude_db_left, magnitude_db_right, sample_rate, is_stereo)
        """
        # Read the audio file
        sample_rate, data = wavfile.read(str(audio_file))

        # Check if stereo or mono
        is_stereo = len(data.shape) > 1 and data.shape[1] == 2

        # Normalize the data
        data = data.astype(float)

        # Use configurable FFT size for frequency resolution
        nfft = self.fft_size

        if is_stereo:
            # Process left and right channels separately
            left_channel = data[:, 0]
            right_channel = data[:, 1]

            # Compute FFT for left channel
            frequencies, psd_left = signal.welch(
                left_channel,
                fs=sample_rate,
                nperseg=min(len(left_channel), nfft),
                noverlap=None,
                nfft=nfft,
                scaling='spectrum'
            )

            # Compute FFT for right channel
            _, psd_right = signal.welch(
                right_channel,
                fs=sample_rate,
                nperseg=min(len(right_channel), nfft),
                noverlap=None,
                nfft=nfft,
                scaling='spectrum'
            )

            # Convert to dB
            magnitude_db_left = 10 * np.log10(psd_left + 1e-12)
            magnitude_db_right = 10 * np.log10(psd_right + 1e-12)
        else:
            # Mono processing
            frequencies, psd = signal.welch(
                data,
                fs=sample_rate,
                nperseg=min(len(data), nfft),
                noverlap=None,
                nfft=nfft,
                scaling='spectrum'
            )

            # Convert to dB
            magnitude_db_left = 10 * np.log10(psd + 1e-12)
            magnitude_db_right = None

        return frequencies, magnitude_db_left, magnitude_db_right, sample_rate, is_stereo

    def plot_fft(self, frequencies, magnitude_db_left, magnitude_db_right, audio_file,
                 is_stereo, sample_rate, highlight_spikes=None):
        """
        Create and save a plot of the FFT results.

        Args:
            frequencies: Array of frequencies
            magnitude_db_left: Array of magnitudes in dB for left channel (or mono)
            magnitude_db_right: Array of magnitudes in dB for right channel (or None for mono)
            audio_file: Path to the audio file being analyzed
            is_stereo: Whether the audio is stereo
            sample_rate: Sample rate of the audio in Hz
            highlight_spikes: Optional dict with 'left' and 'right' keys containing (freq, amp) tuples
        """
        high_freq_mask = frequencies >= 15000

        if is_stereo:
            # Create 2x2 grid for stereo: left and right channels side by side
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 10))

            # Left channel - Full spectrum
            ax1.plot(frequencies / 1000, magnitude_db_left, linewidth=0.5, color='blue')
            ax1.set_xlabel('Frequency (kHz)')
            ax1.set_ylabel('Magnitude (dB)')
            ax1.set_title(f'Left Channel - Full Spectrum')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim([0, frequencies[-1] / 1000])

            # Right channel - Full spectrum
            ax2.plot(frequencies / 1000, magnitude_db_right, linewidth=0.5, color='red')
            ax2.set_xlabel('Frequency (kHz)')
            ax2.set_ylabel('Magnitude (dB)')
            ax2.set_title(f'Right Channel - Full Spectrum')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim([0, frequencies[-1] / 1000])

            # Left channel - High frequency zoom (15kHz+)
            ax3.plot(frequencies[high_freq_mask] / 1000, magnitude_db_left[high_freq_mask],
                    linewidth=1, color='blue')
            ax3.set_xlabel('Frequency (kHz)')
            ax3.set_ylabel('Magnitude (dB)')
            ax3.set_title('Left Channel - High Frequency (15kHz+)')
            ax3.grid(True, alpha=0.3)
            ax3.axvline(x=17, color='gray', linestyle='--', alpha=0.5, label='17kHz threshold')

            if highlight_spikes and 'left' in highlight_spikes and highlight_spikes['left']:
                spike_freq, spike_amp = highlight_spikes['left']
                ax3.plot(spike_freq / 1000, spike_amp, 'bo', markersize=10,
                        label=f'Spike: {spike_freq/1000:.2f}kHz, {spike_amp:.2f}dB')
            ax3.legend()

            # Right channel - High frequency zoom (15kHz+)
            ax4.plot(frequencies[high_freq_mask] / 1000, magnitude_db_right[high_freq_mask],
                    linewidth=1, color='red')
            ax4.set_xlabel('Frequency (kHz)')
            ax4.set_ylabel('Magnitude (dB)')
            ax4.set_title('Right Channel - High Frequency (15kHz+)')
            ax4.grid(True, alpha=0.3)
            ax4.axvline(x=17, color='gray', linestyle='--', alpha=0.5, label='17kHz threshold')

            if highlight_spikes and 'right' in highlight_spikes and highlight_spikes['right']:
                spike_freq, spike_amp = highlight_spikes['right']
                ax4.plot(spike_freq / 1000, spike_amp, 'ro', markersize=10,
                        label=f'Spike: {spike_freq/1000:.2f}kHz, {spike_amp:.2f}dB')
            ax4.legend()

            # Add overall title with FFT size and frequency resolution
            freq_resolution = sample_rate / self.fft_size
            fig.suptitle(f'FFT Analysis: {audio_file.name} | FFT Size: {self.fft_size:,} | Freq Resolution: {freq_resolution:.2f} Hz', 
                        fontsize=12, y=0.995)

        else:
            # Mono: use 2x1 layout as before
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

            # Full spectrum plot
            ax1.plot(frequencies / 1000, magnitude_db_left, linewidth=0.5)
            ax1.set_xlabel('Frequency (kHz)')
            ax1.set_ylabel('Magnitude (dB)')
            freq_resolution = sample_rate / self.fft_size
            ax1.set_title(f'FFT Analysis: {audio_file.name} | FFT Size: {self.fft_size:,} | Freq Resolution: {freq_resolution:.2f} Hz')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim([0, frequencies[-1] / 1000])

            # Zoomed plot focusing on high frequencies (15kHz+)
            ax2.plot(frequencies[high_freq_mask] / 1000, magnitude_db_left[high_freq_mask],
                    linewidth=1)
            ax2.set_xlabel('Frequency (kHz)')
            ax2.set_ylabel('Magnitude (dB)')
            ax2.set_title('High Frequency Region (15kHz+)')
            ax2.grid(True, alpha=0.3)
            ax2.axvline(x=17, color='gray', linestyle='--', alpha=0.5, label='17kHz threshold')

            if highlight_spikes and 'left' in highlight_spikes and highlight_spikes['left']:
                spike_freq, spike_amp = highlight_spikes['left']
                ax2.plot(spike_freq / 1000, spike_amp, 'ro', markersize=10,
                        label=f'Spike: {spike_freq/1000:.2f}kHz, {spike_amp:.2f}dB')
            ax2.legend()

        plt.tight_layout()

        # Save the plot
        plot_file = self.analysis_dir / f"{audio_file.stem}_fft.png"
        plt.savefig(str(plot_file), dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved FFT plot: {plot_file.name}")

    def detect_spike_above_17khz(self, frequencies, magnitude_db):
        """
        Detect if there's a noticeable spike above 17kHz.
        Finds the spike with the largest prominence (difference from surrounding baseline).

        Returns:
            tuple: (spike_detected, peak_frequency, peak_amplitude) or (False, None, None)
        """
        # Focus on frequencies above 17kHz
        mask = frequencies >= 17000
        if not np.any(mask):
            return False, None, None

        high_freqs = frequencies[mask]
        high_mags = magnitude_db[mask]

        # Find all local maxima using scipy's find_peaks
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(high_mags, prominence=1)  # Find peaks with at least 1dB prominence
        
        if len(peaks) == 0:
            # No peaks found, check if there's any significant spike at all
            return False, None, None
        
        # For each peak, calculate its prominence relative to nearby baseline
        window_size = max(10, len(high_mags) // 20)  # Use ~5% of data or min 10 points
        max_prominence = -np.inf
        best_peak_idx = None
        
        for peak_idx in peaks:
            # Define window around this peak
            start_idx = max(0, peak_idx - window_size)
            end_idx = min(len(high_mags), peak_idx + window_size + 1)
            
            # Get baseline from adjacent regions (before and after the peak window)
            left_region = high_mags[max(0, start_idx - window_size):start_idx]
            right_region = high_mags[end_idx:min(len(high_mags), end_idx + window_size)]
            
            # Combine adjacent regions for baseline
            adjacent_values = np.concatenate([left_region, right_region]) if len(left_region) > 0 and len(right_region) > 0 else (
                left_region if len(left_region) > 0 else right_region
            )
            
            if len(adjacent_values) > 0:
                local_baseline = np.median(adjacent_values)
            else:
                # Fallback to excluding just the peak itself
                baseline_mask = np.ones(len(high_mags), dtype=bool)
                baseline_mask[peak_idx] = False
                local_baseline = np.median(high_mags[baseline_mask])
            
            # Calculate prominence for this peak
            prominence = high_mags[peak_idx] - local_baseline
            
            if prominence > max_prominence:
                max_prominence = prominence
                best_peak_idx = peak_idx
        
        # Check if the best spike meets the threshold
        if best_peak_idx is not None and max_prominence >= self.spike_threshold_db:
            peak_freq = high_freqs[best_peak_idx]
            peak_amp = high_mags[best_peak_idx]
            return True, peak_freq, peak_amp

        return False, None, None
    
    def detect_spike_above_17khz_absolute(self, frequencies, magnitude_db):
        """
        Detect spikes above 17kHz using absolute amplitude (old behavior).
        Returns (has_spike, peak_frequency, peak_amplitude)
        """
        # Filter for frequencies above 17kHz
        mask = frequencies >= 17000
        if not np.any(mask):
            return False, None, None
        
        high_freqs = frequencies[mask]
        high_mags = magnitude_db[mask]
        
        # Find peak with maximum absolute amplitude
        peak_idx = np.argmax(high_mags)
        peak_freq = high_freqs[peak_idx]
        peak_amp = high_mags[peak_idx]
        
        # Calculate baseline from all other frequencies
        baseline_mask = np.ones(len(high_mags), dtype=bool)
        baseline_mask[peak_idx] = False
        baseline = np.median(high_mags[baseline_mask])
        
        # Check if spike is significant
        if peak_amp - baseline >= self.spike_threshold_db:
            return True, peak_freq, peak_amp
        
        return False, None, None

    @staticmethod
    def process_segment_worker(segment_file, segment_idx, segment_duration, fft_size, 
                               spike_threshold_db, analysis_dir, use_prominence=True):
        """
        Worker function to process a single segment in parallel.
        Returns spike detection results for the segment.
        """
        # Create a temporary analyzer instance just for this worker
        # We need to recreate objects since they can't be pickled across processes
        from pathlib import Path
        import numpy as np
        from scipy.io import wavfile
        from scipy import signal
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        segment_file = Path(segment_file)
        analysis_dir = Path(analysis_dir)
        
        # Read the audio file
        sample_rate, data = wavfile.read(str(segment_file))
        
        # Check if stereo or mono
        is_stereo = len(data.shape) > 1 and data.shape[1] == 2
        data = data.astype(float)
        
        # Perform FFT analysis
        if is_stereo:
            left_channel = data[:, 0]
            right_channel = data[:, 1]
            
            frequencies, psd_left = signal.welch(
                left_channel, fs=sample_rate,
                nperseg=min(len(left_channel), fft_size),
                noverlap=None, nfft=fft_size, scaling='spectrum'
            )
            _, psd_right = signal.welch(
                right_channel, fs=sample_rate,
                nperseg=min(len(right_channel), fft_size),
                noverlap=None, nfft=fft_size, scaling='spectrum'
            )
            mag_left = 10 * np.log10(psd_left + 1e-12)
            mag_right = 10 * np.log10(psd_right + 1e-12)
        else:
            frequencies, psd = signal.welch(
                data, fs=sample_rate,
                nperseg=min(len(data), fft_size),
                noverlap=None, nfft=fft_size, scaling='spectrum'
            )
            mag_left = 10 * np.log10(psd + 1e-12)
            mag_right = None
        
        # Detect spikes using prominence or absolute amplitude
        def detect_spike(frequencies, magnitude_db, spike_threshold_db, use_prominence):
            mask = frequencies >= 17000
            if not np.any(mask):
                return False, None, None
            
            high_freqs = frequencies[mask]
            high_mags = magnitude_db[mask]
            
            if use_prominence:
                # New behavior: Find spike with maximum prominence
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(high_mags, prominence=1)
                
                if len(peaks) == 0:
                    return False, None, None
                
                # For each peak, calculate prominence relative to nearby baseline
                window_size = max(10, len(high_mags) // 20)
                max_prominence = -np.inf
                best_peak_idx = None
                
                for peak_idx in peaks:
                    start_idx = max(0, peak_idx - window_size)
                    end_idx = min(len(high_mags), peak_idx + window_size + 1)
                    
                    left_region = high_mags[max(0, start_idx - window_size):start_idx]
                    right_region = high_mags[end_idx:min(len(high_mags), end_idx + window_size)]
                    
                    adjacent_values = np.concatenate([left_region, right_region]) if len(left_region) > 0 and len(right_region) > 0 else (
                        left_region if len(left_region) > 0 else right_region
                    )
                    
                    if len(adjacent_values) > 0:
                        local_baseline = np.median(adjacent_values)
                    else:
                        baseline_mask = np.ones(len(high_mags), dtype=bool)
                        baseline_mask[peak_idx] = False
                        local_baseline = np.median(high_mags[baseline_mask])
                    
                    prominence = high_mags[peak_idx] - local_baseline
                    
                    if prominence > max_prominence:
                        max_prominence = prominence
                        best_peak_idx = peak_idx
                
                if best_peak_idx is not None and max_prominence >= spike_threshold_db:
                    peak_freq = high_freqs[best_peak_idx]
                    peak_amp = high_mags[best_peak_idx]
                    return True, peak_freq, peak_amp
            else:
                # Old behavior: Find spike with maximum absolute amplitude
                peak_idx = np.argmax(high_mags)
                peak_freq = high_freqs[peak_idx]
                peak_amp = high_mags[peak_idx]
                
                # Calculate baseline from all other frequencies
                baseline_mask = np.ones(len(high_mags), dtype=bool)
                baseline_mask[peak_idx] = False
                baseline = np.median(high_mags[baseline_mask])
                
                # Check if spike is significant
                if peak_amp - baseline >= spike_threshold_db:
                    return True, peak_freq, peak_amp
            
            return False, None, None
        
        # Detect spikes
        spike_detected_left, peak_freq_left, peak_amp_left = detect_spike(
            frequencies, mag_left, spike_threshold_db, use_prominence
        )
        
        spikes_found = []
        highlight_spikes = {}
        segment_time = segment_idx * segment_duration
        
        if spike_detected_left:
            channel_label = 'left' if is_stereo else 'mono'
            spikes_found.append({
                'segment': segment_file.name,
                'segment_index': segment_idx,
                'time_seconds': segment_time,
                'channel': channel_label,
                'frequency_hz': float(peak_freq_left),
                'frequency_khz': float(peak_freq_left / 1000),
                'amplitude_db': float(peak_amp_left)
            })
            highlight_spikes['left'] = (peak_freq_left, peak_amp_left)
        
        if is_stereo:
            spike_detected_right, peak_freq_right, peak_amp_right = detect_spike(
                frequencies, mag_right, spike_threshold_db, use_prominence
            )
            if spike_detected_right:
                spikes_found.append({
                    'segment': segment_file.name,
                    'segment_index': segment_idx,
                    'time_seconds': segment_time,
                    'channel': 'right',
                    'frequency_hz': float(peak_freq_right),
                    'frequency_khz': float(peak_freq_right / 1000),
                    'amplitude_db': float(peak_amp_right)
                })
                highlight_spikes['right'] = (peak_freq_right, peak_amp_right)
        
        # Generate plot
        high_freq_mask = frequencies >= 15000
        
        if is_stereo:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 10))
            
            ax1.plot(frequencies / 1000, mag_left, linewidth=0.5, color='blue')
            ax1.set_xlabel('Frequency (kHz)')
            ax1.set_ylabel('Magnitude (dB)')
            ax1.set_title('Left Channel - Full Spectrum')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim([0, frequencies[-1] / 1000])
            
            ax2.plot(frequencies / 1000, mag_right, linewidth=0.5, color='red')
            ax2.set_xlabel('Frequency (kHz)')
            ax2.set_ylabel('Magnitude (dB)')
            ax2.set_title('Right Channel - Full Spectrum')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim([0, frequencies[-1] / 1000])
            
            ax3.plot(frequencies[high_freq_mask] / 1000, mag_left[high_freq_mask],
                    linewidth=1, color='blue')
            ax3.set_xlabel('Frequency (kHz)')
            ax3.set_ylabel('Magnitude (dB)')
            ax3.set_title('Left Channel - High Frequency (15kHz+)')
            ax3.grid(True, alpha=0.3)
            ax3.axvline(x=17, color='gray', linestyle='--', alpha=0.5, label='17kHz threshold')
            if 'left' in highlight_spikes:
                spike_freq, spike_amp = highlight_spikes['left']
                ax3.plot(spike_freq / 1000, spike_amp, 'bo', markersize=10,
                        label=f'Spike: {spike_freq/1000:.2f}kHz, {spike_amp:.2f}dB')
            ax3.legend()
            
            ax4.plot(frequencies[high_freq_mask] / 1000, mag_right[high_freq_mask],
                    linewidth=1, color='red')
            ax4.set_xlabel('Frequency (kHz)')
            ax4.set_ylabel('Magnitude (dB)')
            ax4.set_title('Right Channel - High Frequency (15kHz+)')
            ax4.grid(True, alpha=0.3)
            ax4.axvline(x=17, color='gray', linestyle='--', alpha=0.5, label='17kHz threshold')
            if 'right' in highlight_spikes:
                spike_freq, spike_amp = highlight_spikes['right']
                ax4.plot(spike_freq / 1000, spike_amp, 'ro', markersize=10,
                        label=f'Spike: {spike_freq/1000:.2f}kHz, {spike_amp:.2f}dB')
            ax4.legend()
            
            freq_resolution = sample_rate / fft_size
            fig.suptitle(f'FFT Analysis: {segment_file.name} | FFT Size: {fft_size:,} | Freq Resolution: {freq_resolution:.2f} Hz', 
                        fontsize=12, y=0.995)
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            ax1.plot(frequencies / 1000, mag_left, linewidth=0.5)
            ax1.set_xlabel('Frequency (kHz)')
            ax1.set_ylabel('Magnitude (dB)')
            freq_resolution = sample_rate / fft_size
            ax1.set_title(f'FFT Analysis: {segment_file.name} | FFT Size: {fft_size:,} | Freq Resolution: {freq_resolution:.2f} Hz')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim([0, frequencies[-1] / 1000])
            
            ax2.plot(frequencies[high_freq_mask] / 1000, mag_left[high_freq_mask],
                    linewidth=1)
            ax2.set_xlabel('Frequency (kHz)')
            ax2.set_ylabel('Magnitude (dB)')
            ax2.set_title('High Frequency Region (15kHz+)')
            ax2.grid(True, alpha=0.3)
            ax2.axvline(x=17, color='gray', linestyle='--', alpha=0.5, label='17kHz threshold')
            if 'left' in highlight_spikes:
                spike_freq, spike_amp = highlight_spikes['left']
                ax2.plot(spike_freq / 1000, spike_amp, 'ro', markersize=10,
                        label=f'Spike: {spike_freq/1000:.2f}kHz, {spike_amp:.2f}dB')
            ax2.legend()
        
        plt.tight_layout()
        plot_file = analysis_dir / f"{segment_file.stem}_fft.png"
        plt.savefig(str(plot_file), dpi=150, bbox_inches='tight')
        plt.close()
        
        return {
            'segment_idx': segment_idx,
            'segment_name': segment_file.name,
            'sample_rate': sample_rate,
            'is_stereo': is_stereo,
            'max_frequency': frequencies[-1],
            'spikes': spikes_found,
            'plot_file': plot_file.name
        }

    def analyze_segments(self, segment_files):
        """Analyze all audio segments for high-frequency spikes using parallel processing."""
        print("\n" + "="*60)
        print("Analyzing segments for high-frequency spikes...")
        print(f"Using parallel processing with {os.cpu_count()} CPU cores")
        print("="*60 + "\n")

        total_segments = len(segment_files)
        completed = 0
        
        # Process segments in parallel
        results = []
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            # Submit all tasks
            future_to_segment = {
                executor.submit(
                    self.process_segment_worker,
                    str(segment_file),
                    idx,
                    self.segment_duration,
                    self.fft_size,
                    self.spike_threshold_db,
                    str(self.analysis_dir),
                    self.use_prominence
                ): (idx, segment_file) for idx, segment_file in enumerate(segment_files)
            }
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_segment):
                idx, segment_file = future_to_segment[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    if self.verbose:
                        # Print detailed progress
                        print(f"Processed: {result['segment_name']}")
                        print(f"  Sample rate: {result['sample_rate']} Hz")
                        print(f"  Channels: {'Stereo' if result['is_stereo'] else 'Mono'}")
                        print(f"  Max frequency analyzed: {result['max_frequency']/1000:.2f} kHz")
                        
                        if result['spikes']:
                            for spike in result['spikes']:
                                channel_label = spike['channel'].capitalize()
                                print(f"  ✓ SPIKE DETECTED ({channel_label}): "
                                      f"{spike['frequency_khz']:.2f} kHz at {spike['amplitude_db']:.2f} dB")
                        else:
                            if result['is_stereo']:
                                print(f"  No significant spikes above 17kHz in either channel")
                            else:
                                print(f"  No significant spike above 17kHz")
                        
                        print(f"  Saved FFT plot: {result['plot_file']}")
                        print()
                    else:
                        # Simple progress counter
                        print(f"\rCompleted: {completed}/{total_segments}", end='', flush=True)
                    
                except Exception as e:
                    completed += 1
                    if self.verbose:
                        print(f"Error processing {segment_file.name}: {e}")
                        import traceback
                        traceback.print_exc()
                    else:
                        print(f"\rCompleted: {completed}/{total_segments} (1 error)", end='', flush=True)
        
        if not self.verbose:
            print()  # New line after progress counter
        
        # Sort results by segment index and collect spikes
        results.sort(key=lambda x: x['segment_idx'])
        for result in results:
            self.spikes_detected.extend(result['spikes'])

    def generate_report(self):
        """Generate a summary report of detected spikes."""
        if self.verbose:
            print("\n" + "="*60)
            print("SPIKE DETECTION REPORT")
            print("="*60 + "\n")

            if not self.spikes_detected:
                print("No noticeable spikes detected above 17kHz in any segment.")
            else:
                print(f"Found {len(self.spikes_detected)} spike(s) across all segments:\n")
                for i, spike in enumerate(self.spikes_detected, 1):
                    print(f"{i}. {spike['segment']} ({spike['channel'].upper()})")
                    print(f"   Frequency: {spike['frequency_khz']:.2f} kHz ({spike['frequency_hz']:.0f} Hz)")
                    print(f"   Amplitude: {spike['amplitude_db']:.2f} dB")
                    print()
        else:
            # Brief summary in non-verbose mode
            print(f"\nFound {len(self.spikes_detected)} spike(s) above 17kHz")

        # Save report to JSON
        report_file = self.analysis_dir / f"{self.input_file.stem}_spike_report.json"
        with open(report_file, 'w') as f:
            json.dump({
                'input_file': str(self.input_file),
                'spike_threshold_db': self.spike_threshold_db,
                'fft_size': self.fft_size,
                'total_spikes': len(self.spikes_detected),
                'spikes': self.spikes_detected
            }, f, indent=2)

        print(f"Report saved to: {report_file}")

    def generate_summary_plots(self):
        """Generate summary plots showing spike frequency and amplitude over time."""
        if not self.spikes_detected:
            print("No spikes to plot in summary.")
            return

        print("\nGenerating summary plots...")

        # Organize data by channel
        channels = {}
        for spike in self.spikes_detected:
            channel = spike['channel']
            if channel not in channels:
                channels[channel] = {'times': [], 'frequencies': [], 'amplitudes': []}
            channels[channel]['times'].append(spike['time_seconds'])
            channels[channel]['frequencies'].append(spike['frequency_khz'])
            channels[channel]['amplitudes'].append(spike['amplitude_db'])

        # Determine if we have stereo (left and right) or just mono
        has_left = 'left' in channels
        has_right = 'right' in channels
        has_mono = 'mono' in channels
        is_stereo = has_left or has_right

        if is_stereo:
            # Create 2x2 grid: left and right channels side by side
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 10))

            # Left channel - Frequency over time
            if has_left:
                ax1.scatter(channels['left']['times'], channels['left']['frequencies'],
                           c='blue', s=20, alpha=0.7)
                ax1.set_xlabel('Time (seconds)', fontsize=11)
                ax1.set_ylabel('Frequency (kHz)', fontsize=11)
                ax1.set_title('Left Channel - Spike Frequency Over Time', fontsize=12, fontweight='bold')
                ax1.grid(True, alpha=0.3)
                ax1.axhline(y=17, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            else:
                ax1.text(0.5, 0.5, 'No left channel spikes', ha='center', va='center',
                        transform=ax1.transAxes, fontsize=12)
                ax1.set_title('Left Channel - Spike Frequency Over Time', fontsize=12, fontweight='bold')

            # Right channel - Frequency over time
            if has_right:
                ax2.scatter(channels['right']['times'], channels['right']['frequencies'],
                           c='red', s=20, alpha=0.7)
                ax2.set_xlabel('Time (seconds)', fontsize=11)
                ax2.set_ylabel('Frequency (kHz)', fontsize=11)
                ax2.set_title('Right Channel - Spike Frequency Over Time', fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                ax2.axhline(y=17, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            else:
                ax2.text(0.5, 0.5, 'No right channel spikes', ha='center', va='center',
                        transform=ax2.transAxes, fontsize=12)
                ax2.set_title('Right Channel - Spike Frequency Over Time', fontsize=12, fontweight='bold')

            # Left channel - Amplitude over time
            if has_left:
                ax3.scatter(channels['left']['times'], channels['left']['amplitudes'],
                           c='blue', s=20, alpha=0.7)
                ax3.set_xlabel('Time (seconds)', fontsize=11)
                ax3.set_ylabel('Amplitude (dB)', fontsize=11)
                ax3.set_title('Left Channel - Spike Amplitude Over Time', fontsize=12, fontweight='bold')
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'No left channel spikes', ha='center', va='center',
                        transform=ax3.transAxes, fontsize=12)
                ax3.set_title('Left Channel - Spike Amplitude Over Time', fontsize=12, fontweight='bold')

            # Right channel - Amplitude over time
            if has_right:
                ax4.scatter(channels['right']['times'], channels['right']['amplitudes'],
                           c='red', s=20, alpha=0.7)
                ax4.set_xlabel('Time (seconds)', fontsize=11)
                ax4.set_ylabel('Amplitude (dB)', fontsize=11)
                ax4.set_title('Right Channel - Spike Amplitude Over Time', fontsize=12, fontweight='bold')
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'No right channel spikes', ha='center', va='center',
                        transform=ax4.transAxes, fontsize=12)
                ax4.set_title('Right Channel - Spike Amplitude Over Time', fontsize=12, fontweight='bold')

            fig.suptitle('Spike Analysis Summary', fontsize=14, fontweight='bold', y=0.995)

        else:
            # Mono: use 2x1 layout
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

            # Frequency over time
            ax1.scatter(channels['mono']['times'], channels['mono']['frequencies'],
                       c='green', s=20, alpha=0.7)
            ax1.set_xlabel('Time (seconds)', fontsize=12)
            ax1.set_ylabel('Frequency (kHz)', fontsize=12)
            ax1.set_title('Spike Frequency Over Time', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=17, color='gray', linestyle='--', alpha=0.5, linewidth=1)

            # Amplitude over time
            ax2.scatter(channels['mono']['times'], channels['mono']['amplitudes'],
                       c='green', s=20, alpha=0.7)
            ax2.set_xlabel('Time (seconds)', fontsize=12)
            ax2.set_ylabel('Amplitude (dB)', fontsize=12)
            ax2.set_title('Spike Amplitude Over Time', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the summary plot
        summary_file = self.analysis_dir / f"{self.input_file.stem}_spike_summary.png"
        plt.savefig(str(summary_file), dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Summary plot saved to: {summary_file}")

    def run(self):
        """Run the complete analysis pipeline."""
        print("="*60)
        print("AUDIO FREQUENCY ANALYZER")
        print("="*60)
        print(f"Input file: {self.input_file}")
        print(f"Segments directory: {self.segments_dir}")
        print(f"Analysis directory: {self.analysis_dir}")
        print(f"Spike threshold: {self.spike_threshold_db} dB above surrounding frequencies")
        print("="*60 + "\n")

        # Step 1: Split audio into segments
        print("Step 1: Splitting audio into segments...")
        segment_files = self.split_audio()

        if not segment_files:
            print("No segments were created. Exiting.")
            return

        print(f"\nTotal segments: {len(segment_files)}\n")

        # Step 2: Analyze each segment
        self.analyze_segments(segment_files)

        # Step 3: Generate summary plots
        self.generate_summary_plots()

        # Step 4: Generate report
        self.generate_report()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze audio files for high-frequency spikes above 17kHz.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s testaudio/testfile.mp4
  %(prog)s testaudio/testfile.mp4 -o my_output
  %(prog)s testaudio/testfile.mp4 -t 12 -f 131072
  %(prog)s testaudio/testfile.mp4 --output-dir results --spike-threshold 15 --fft-size 262144
        ''')

    parser.add_argument(
        'input_file',
        help='Path to the input audio/video file')

    parser.add_argument(
        '-o', '--output-dir',
        default='output',
        help='Base directory for output files (default: output)')

    parser.add_argument(
        '-t', '--spike-threshold',
        type=float,
        default=10,
        metavar='DB',
        help='Minimum dB above surrounding frequencies to consider a spike (default: 10)')

    parser.add_argument(
        '-f', '--fft-size',
        type=int,
        default=65536,
        metavar='SIZE',
        help='FFT size for frequency analysis (default: 65536)')
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed per-segment output')
    
    parser.add_argument(
        '--absolute-amplitude',
        action='store_true',
        help='Use absolute amplitude detection instead of prominence-based detection')

    args = parser.parse_args()

    try:
        analyzer = AudioFrequencyAnalyzer(
            args.input_file,
            args.output_dir,
            args.spike_threshold,
            args.fft_size,
            args.verbose,
            use_prominence=not args.absolute_amplitude)
        analyzer.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
