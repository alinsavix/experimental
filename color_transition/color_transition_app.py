#!/usr/bin/env -S uv run --script
import tkinter as tk
from tkinter import ttk
import colorsys
import random
import time
import threading
import math
try:
    from colorspacious import cspace_convert
    HAS_COLORSPACIOUS = True
except ImportError:
    HAS_COLORSPACIOUS = False


class ColorTransitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Color Transition Visualizer")
        # Let window size itself based on content
        # self.root.geometry("900x1100")  # Removed fixed size

        self.transition_running = False
        self.transition_thread = None
        self.cycling_tests = False
        self.cycle_thread = None

        # Define problematic test cases for RGB interpolation
        self.test_cases = {
            "Red → Green (muddy brown)": ("#ff0000", "#00ff00"),
            "Blue → Yellow (gray/desaturated)": ("#0000ff", "#ffff00"),
            "Red → Cyan (desaturated)": ("#ff0000", "#00ffff"),
            "Magenta → Green (gray)": ("#ff00ff", "#00ff00"),
            "Orange → Blue (dark/muddy)": ("#ff8000", "#0080ff"),
            "Bright Red → Dark Green (brightness dip)": ("#ff0000", "#006400"),
            "Purple → Yellow (muddy/gray)": ("#8000ff", "#ffff00"),
            "Lime → Magenta (desaturated)": ("#80ff00", "#ff00ff"),
            "Cyan → Orange (gray mid-tones)": ("#00ffff", "#ff8800"),
            "Pink → Teal (brightness shift)": ("#ff69b4", "#008080")
        }

        # Main container
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Algorithm selection
        ttk.Label(main_frame, text="Transition Algorithm:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.algorithm_var = tk.StringVar()
        self.algorithm_combo = ttk.Combobox(main_frame, textvariable=self.algorithm_var, state="readonly", width=30)
        algorithms = [
            'Linear RGB',
            'HSV (Hue-Saturation-Value)',
            'HSL (Hue-Saturation-Lightness)',
            'Cubic Bezier RGB',
            'Exponential RGB'
        ]
        if HAS_COLORSPACIOUS:
            algorithms.extend([
                'LAB (CIE L*a*b*)',
                'LCH (Lightness-Chroma-Hue)'
            ])
        # OKLab and OKLCh use manual conversion, always available
        algorithms.extend([
            'OKLab (Perceptual)',
            'OKLCh (Perceptual Cylindrical)'
        ])
        self.algorithm_combo['values'] = tuple(algorithms)
        self.algorithm_combo.current(0)
        self.algorithm_combo.grid(row=0, column=1, columnspan=2, sticky=tk.W, pady=5)

        # From color
        ttk.Label(main_frame, text="From Color (hex):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.from_color_var = tk.StringVar(value="#ff0000")
        self.from_color_entry = ttk.Entry(main_frame, textvariable=self.from_color_var, width=15)
        self.from_color_entry.grid(row=1, column=1, sticky=tk.W, pady=5, padx=(0, 5))

        # From color preview
        self.from_color_preview = tk.Canvas(main_frame, width=30, height=30, bg="#ff0000", highlightthickness=1)
        self.from_color_preview.grid(row=1, column=2, sticky=tk.W, pady=5)

        # To color
        ttk.Label(main_frame, text="To Color (hex):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.to_color_var = tk.StringVar(value="#00ff00")
        self.to_color_entry = ttk.Entry(main_frame, textvariable=self.to_color_var, width=15)
        self.to_color_entry.grid(row=2, column=1, sticky=tk.W, pady=5, padx=(0, 5))

        # To color preview
        self.to_color_preview = tk.Canvas(main_frame, width=30, height=30, bg="#00ff00", highlightthickness=1)
        self.to_color_preview.grid(row=2, column=2, sticky=tk.W, pady=5)

        # Test cases selection
        ttk.Label(main_frame, text="Test Cases:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.test_case_var = tk.StringVar()
        self.test_case_combo = ttk.Combobox(main_frame, textvariable=self.test_case_var, state="readonly", width=30)
        self.test_case_combo['values'] = tuple(self.test_cases.keys())
        self.test_case_combo.bind('<<ComboboxSelected>>', self.load_test_case)
        self.test_case_combo.grid(row=3, column=1, columnspan=2, sticky=tk.W, pady=5)

        # Cycle test cases button
        self.cycle_btn = ttk.Button(main_frame, text="Cycle All Test Cases", command=self.cycle_test_cases)
        self.cycle_btn.grid(row=4, column=0, columnspan=3, pady=5)

        # Randomize button
        self.randomize_btn = ttk.Button(main_frame, text="Randomize Colors", command=self.randomize_colors)
        self.randomize_btn.grid(row=5, column=0, columnspan=3, pady=5)

        # Transition duration
        ttk.Label(main_frame, text="Duration (seconds):").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.duration_var = tk.StringVar(value="3")
        self.duration_entry = ttk.Entry(main_frame, textvariable=self.duration_var, width=15)
        self.duration_entry.grid(row=6, column=1, sticky=tk.W, pady=5)

        # Go button
        self.go_btn = ttk.Button(main_frame, text="Go", command=self.start_transition)
        self.go_btn.grid(row=7, column=0, columnspan=3, pady=10)

        # Transition display - multiple algorithm patches
        self.previews_label = ttk.Label(main_frame, text="Transition Previews - Color Distance (ΔE): --", font=('Arial', 10))
        self.previews_label.grid(row=8, column=0, columnspan=3, sticky=tk.W, pady=(10, 5))

        # Create frame for algorithm patches
        patches_frame = ttk.Frame(main_frame)
        patches_frame.grid(row=9, column=0, columnspan=3, pady=5)

        # Get all algorithm names
        algorithms = [
            'Linear RGB',
            'HSV (Hue-Saturation-Value)',
            'HSL (Hue-Saturation-Lightness)',
            'Cubic Bezier RGB',
            'Exponential RGB'
        ]
        if HAS_COLORSPACIOUS:
            algorithms.extend([
                'LAB (CIE L*a*b*)',
                'LCH (Lightness-Chroma-Hue)'
            ])
        algorithms.extend([
            'OKLab (Perceptual)',
            'OKLCh (Perceptual Cylindrical)'
        ])

        # Create patches in a grid (3 columns)
        self.algorithm_patches = {}
        row = 0
        col = 0
        patch_width = 170
        patch_height = 80

        for i, algo in enumerate(algorithms):
            # Create container for each patch
            patch_container = ttk.Frame(patches_frame)
            patch_container.grid(row=row, column=col, padx=5, pady=5)

            # Algorithm name label
            label = ttk.Label(patch_container, text=algo, font=('Arial', 8))
            label.pack()

            # Color patch canvas
            canvas = tk.Canvas(patch_container, width=patch_width, height=patch_height,
                             bg="white", highlightthickness=1)
            canvas.pack()

            self.algorithm_patches[algo] = canvas

            # Move to next position
            col += 1
            if col >= 3:
                col = 0
                row += 1

        # Bind color entry changes to update previews and distance
        self.from_color_var.trace('w', lambda *args: self.update_color_preview(self.from_color_var, self.from_color_preview))
        self.to_color_var.trace('w', lambda *args: self.update_color_preview(self.to_color_var, self.to_color_preview))

        self.from_color_var.trace('w', lambda *args: self.update_color_distance())
        self.to_color_var.trace('w', lambda *args: self.update_color_distance())

        # Calculate initial distance
        self.update_color_distance()

    def update_color_preview(self, color_var, preview_canvas):
        """Update the color preview when hex input changes"""
        try:
            color = color_var.get()
            if color.startswith('#') and len(color) == 7:
                preview_canvas.config(bg=color)
        except:
            pass

    def calculate_delta_e_oklab(self, rgb1, rgb2):
        """Calculate Delta E (color difference) using OKLab space"""
        # Convert both colors to OKLab
        linear1 = self.rgb_to_linear(*rgb1)
        linear2 = self.rgb_to_linear(*rgb2)

        oklab1 = self.linear_rgb_to_oklab(*linear1)
        oklab2 = self.linear_rgb_to_oklab(*linear2)

        # Calculate Euclidean distance in OKLab space
        delta_L = oklab1[0] - oklab2[0]
        delta_a = oklab1[1] - oklab2[1]
        delta_b = oklab1[2] - oklab2[2]

        delta_e = math.sqrt(delta_L**2 + delta_a**2 + delta_b**2)

        # Scale to match traditional Delta E ranges (OKLab L is 0-1, traditional is 0-100)
        # Multiply by 100 to get values comparable to CIE Delta E
        delta_e = delta_e * 100

        return delta_e

    def update_color_distance(self):
        """Update the color distance display"""
        try:
            from_hex = self.from_color_var.get()
            to_hex = self.to_color_var.get()

            if not (from_hex.startswith('#') and len(from_hex) == 7):
                self.previews_label.config(text="Transition Previews - Color Distance (ΔE): --")
                return
            if not (to_hex.startswith('#') and len(to_hex) == 7):
                self.previews_label.config(text="Transition Previews - Color Distance (ΔE): --")
                return

            from_rgb = self.hex_to_rgb(from_hex)
            to_rgb = self.hex_to_rgb(to_hex)

            # Calculate Delta E using OKLab (perceptually uniform)
            delta_e = self.calculate_delta_e_oklab(from_rgb, to_rgb)

            # Update label with description
            if delta_e < 1.0:
                desc = "(imperceptible)"
            elif delta_e < 2.3:
                desc = "(just noticeable)"
            elif delta_e < 5.0:
                desc = "(noticeable)"
            elif delta_e < 10.0:
                desc = "(significant)"
            else:
                desc = "(very large)"

            self.previews_label.config(text=f"Transition Previews - Color Distance (ΔE): {delta_e:.2f} {desc}")
        except Exception as e:
            self.previews_label.config(text="Transition Previews - Color Distance (ΔE): --")
            print(f"Error calculating distance: {e}")

    def hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple (0-255)"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def rgb_to_hex(self, r, g, b):
        """Convert RGB tuple to hex color"""
        return '#{:02x}{:02x}{:02x}'.format(int(r), int(g), int(b))

    def rgb_to_linear(self, r, g, b):
        """Convert sRGB to linear RGB"""
        def srgb_to_linear(c):
            c = c / 255.0
            if c <= 0.04045:
                return c / 12.92
            else:
                return ((c + 0.055) / 1.055) ** 2.4
        return (srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b))

    def linear_to_rgb(self, lr, lg, lb):
        """Convert linear RGB to sRGB"""
        def linear_to_srgb(c):
            if c <= 0.0031308:
                return c * 12.92 * 255.0
            else:
                return (1.055 * (c ** (1.0/2.4)) - 0.055) * 255.0
        return (linear_to_srgb(lr), linear_to_srgb(lg), linear_to_srgb(lb))

    def linear_rgb_to_oklab(self, lr, lg, lb):
        """Convert linear RGB to OKLab"""
        l = 0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb
        m = 0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb
        s = 0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb

        l_ = l ** (1/3) if l >= 0 else -((-l) ** (1/3))
        m_ = m ** (1/3) if m >= 0 else -((-m) ** (1/3))
        s_ = s ** (1/3) if s >= 0 else -((-s) ** (1/3))

        L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
        a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
        b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

        return (L, a, b)

    def oklab_to_linear_rgb(self, L, a, b):
        """Convert OKLab to linear RGB"""
        l_ = L + 0.3963377774 * a + 0.2158037573 * b
        m_ = L - 0.1055613458 * a - 0.0638541728 * b
        s_ = L - 0.0894841775 * a - 1.2914855480 * b

        l = l_ ** 3
        m = m_ ** 3
        s = s_ ** 3

        lr = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
        lg = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
        lb = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

        return (lr, lg, lb)

    def oklab_to_oklch(self, L, a, b):
        """Convert OKLab to OKLCh"""
        import math
        C = math.sqrt(a * a + b * b)
        h = math.atan2(b, a) * 180 / math.pi
        if h < 0:
            h += 360
        return (L, C, h)

    def oklch_to_oklab(self, L, C, h):
        """Convert OKLCh to OKLab"""
        import math
        h_rad = h * math.pi / 180
        a = C * math.cos(h_rad)
        b = C * math.sin(h_rad)
        return (L, a, b)

    def random_color(self):
        """Generate a random hex color"""
        return '#{:06x}'.format(random.randint(0, 0xFFFFFF))

    def randomize_colors(self):
        """Randomize both colors and start transition"""
        self.from_color_var.set(self.random_color())
        self.to_color_var.set(self.random_color())
        self.start_transition()

    def load_test_case(self, event=None):
        """Load selected test case colors"""
        test_name = self.test_case_var.get()
        if test_name in self.test_cases:
            from_color, to_color = self.test_cases[test_name]
            self.from_color_var.set(from_color)
            self.to_color_var.set(to_color)
            self.start_transition()

    def cycle_test_cases(self):
        """Cycle through all test cases with 2 second pause between each"""
        if self.cycling_tests:
            # Stop cycling
            self.cycling_tests = False
            self.cycle_btn.config(text="Cycle All Test Cases")
            return

        # Start cycling
        self.cycling_tests = True
        self.cycle_btn.config(text="Stop Cycling")
        self.go_btn.config(state='disabled')
        self.randomize_btn.config(state='disabled')

        # Run cycle in separate thread
        self.cycle_thread = threading.Thread(
            target=self.run_cycle,
            daemon=True
        )
        self.cycle_thread.start()

    def run_cycle(self):
        """Run through all test cases once"""
        test_names = list(self.test_cases.keys())

        # Run through test cases once
        for test_name in test_names:
            if not self.cycling_tests:
                break

            # Load test case
            from_color, to_color = self.test_cases[test_name]
            self.root.after(0, self.from_color_var.set, from_color)
            self.root.after(0, self.to_color_var.set, to_color)
            self.root.after(0, self.test_case_combo.set, test_name)

            # Wait a moment for colors to update
            time.sleep(0.1)

            # Get colors and duration
            try:
                from_rgb = self.hex_to_rgb(from_color)
                to_rgb = self.hex_to_rgb(to_color)
                duration = float(self.duration_var.get())
            except:
                continue

            # Run transition
            start_time = time.time()
            while time.time() - start_time < duration:
                if not self.cycling_tests:
                    break

                elapsed = time.time() - start_time
                t = elapsed / duration

                # Update all patches
                for algo_name, canvas in self.algorithm_patches.items():
                    transition_func = self.get_transition_function(algo_name)
                    current_rgb = transition_func(from_rgb, to_rgb, t)
                    current_hex = self.rgb_to_hex(*current_rgb)
                    self.root.after(0, self.update_patch, canvas, current_hex)

                time.sleep(1/60)

            # Pause between test cases (but not after the last one)
            if self.cycling_tests and test_name != test_names[-1]:
                time.sleep(2)

        # Stop cycling and re-enable buttons
        self.cycling_tests = False
        self.root.after(0, self.cycle_btn.config, {'text': 'Cycle All Test Cases'})
        self.root.after(0, self.go_btn.config, {'state': 'normal'})
        self.root.after(0, self.randomize_btn.config, {'state': 'normal'})

    def linear_rgb_transition(self, from_rgb, to_rgb, t):
        """Linear interpolation in RGB space"""
        r = from_rgb[0] + (to_rgb[0] - from_rgb[0]) * t
        g = from_rgb[1] + (to_rgb[1] - from_rgb[1]) * t
        b = from_rgb[2] + (to_rgb[2] - from_rgb[2]) * t
        return (r, g, b)

    def hsv_transition(self, from_rgb, to_rgb, t):
        """Transition through HSV color space"""
        from_hsv = colorsys.rgb_to_hsv(from_rgb[0]/255, from_rgb[1]/255, from_rgb[2]/255)
        to_hsv = colorsys.rgb_to_hsv(to_rgb[0]/255, to_rgb[1]/255, to_rgb[2]/255)

        # Interpolate HSV values
        h = from_hsv[0] + (to_hsv[0] - from_hsv[0]) * t
        s = from_hsv[1] + (to_hsv[1] - from_hsv[1]) * t
        v = from_hsv[2] + (to_hsv[2] - from_hsv[2]) * t

        rgb = colorsys.hsv_to_rgb(h, s, v)
        return (rgb[0] * 255, rgb[1] * 255, rgb[2] * 255)

    def hsl_transition(self, from_rgb, to_rgb, t):
        """Transition through HSL color space"""
        from_hls = colorsys.rgb_to_hls(from_rgb[0]/255, from_rgb[1]/255, from_rgb[2]/255)
        to_hls = colorsys.rgb_to_hls(to_rgb[0]/255, to_rgb[1]/255, to_rgb[2]/255)

        # Interpolate HLS values
        h = from_hls[0] + (to_hls[0] - from_hls[0]) * t
        l = from_hls[1] + (to_hls[1] - from_hls[1]) * t
        s = from_hls[2] + (to_hls[2] - from_hls[2]) * t

        rgb = colorsys.hls_to_rgb(h, l, s)
        return (rgb[0] * 255, rgb[1] * 255, rgb[2] * 255)

    def cubic_bezier_rgb_transition(self, from_rgb, to_rgb, t):
        """Cubic Bezier interpolation in RGB space"""
        # Cubic easing function
        t_cubic = t * t * (3 - 2 * t)

        r = from_rgb[0] + (to_rgb[0] - from_rgb[0]) * t_cubic
        g = from_rgb[1] + (to_rgb[1] - from_rgb[1]) * t_cubic
        b = from_rgb[2] + (to_rgb[2] - from_rgb[2]) * t_cubic
        return (r, g, b)

    def exponential_rgb_transition(self, from_rgb, to_rgb, t):
        """Exponential interpolation in RGB space"""
        # Exponential easing
        if t == 0:
            t_exp = 0
        elif t == 1:
            t_exp = 1
        else:
            t_exp = 2 ** (10 * (t - 1))

        r = from_rgb[0] + (to_rgb[0] - from_rgb[0]) * t_exp
        g = from_rgb[1] + (to_rgb[1] - from_rgb[1]) * t_exp
        b = from_rgb[2] + (to_rgb[2] - from_rgb[2]) * t_exp
        return (r, g, b)

    def lab_transition(self, from_rgb, to_rgb, t):
        """Transition through CIE LAB color space (perceptually uniform)"""
        if not HAS_COLORSPACIOUS:
            return self.linear_rgb_transition(from_rgb, to_rgb, t)

        # Convert RGB (0-255) to sRGB255 for colorspacious
        from_lab = cspace_convert([from_rgb[0], from_rgb[1], from_rgb[2]], "sRGB255", "CIELab")
        to_lab = cspace_convert([to_rgb[0], to_rgb[1], to_rgb[2]], "sRGB255", "CIELab")

        # Interpolate in LAB space
        L = from_lab[0] + (to_lab[0] - from_lab[0]) * t
        a = from_lab[1] + (to_lab[1] - from_lab[1]) * t
        b = from_lab[2] + (to_lab[2] - from_lab[2]) * t

        # Convert back to RGB
        rgb = cspace_convert([L, a, b], "CIELab", "sRGB255")
        # Clamp values to valid range
        return (max(0, min(255, rgb[0])), max(0, min(255, rgb[1])), max(0, min(255, rgb[2])))

    def lch_transition(self, from_rgb, to_rgb, t):
        """Transition through LCH color space (cylindrical LAB)"""
        if not HAS_COLORSPACIOUS:
            return self.linear_rgb_transition(from_rgb, to_rgb, t)

        # Convert RGB (0-255) to CIELCh
        from_lch = cspace_convert([from_rgb[0], from_rgb[1], from_rgb[2]], "sRGB255", "CIELCh")
        to_lch = cspace_convert([to_rgb[0], to_rgb[1], to_rgb[2]], "sRGB255", "CIELCh")

        # Interpolate in LCH space
        L = from_lch[0] + (to_lch[0] - from_lch[0]) * t
        C = from_lch[1] + (to_lch[1] - from_lch[1]) * t
        h = from_lch[2] + (to_lch[2] - from_lch[2]) * t

        # Convert back to RGB
        rgb = cspace_convert([L, C, h], "CIELCh", "sRGB255")
        # Clamp values to valid range
        return (max(0, min(255, rgb[0])), max(0, min(255, rgb[1])), max(0, min(255, rgb[2])))

    def oklab_transition(self, from_rgb, to_rgb, t):
        """Transition through OKLab color space (improved perceptual uniformity)"""
        # Convert RGB to linear RGB then to OKLab
        from_linear = self.rgb_to_linear(from_rgb[0], from_rgb[1], from_rgb[2])
        to_linear = self.rgb_to_linear(to_rgb[0], to_rgb[1], to_rgb[2])

        from_oklab = self.linear_rgb_to_oklab(*from_linear)
        to_oklab = self.linear_rgb_to_oklab(*to_linear)

        # Interpolate in OKLab space
        L = from_oklab[0] + (to_oklab[0] - from_oklab[0]) * t
        a = from_oklab[1] + (to_oklab[1] - from_oklab[1]) * t
        b = from_oklab[2] + (to_oklab[2] - from_oklab[2]) * t

        # Convert back to RGB
        linear_rgb = self.oklab_to_linear_rgb(L, a, b)
        rgb = self.linear_to_rgb(*linear_rgb)

        # Clamp values to valid range
        return (max(0, min(255, rgb[0])), max(0, min(255, rgb[1])), max(0, min(255, rgb[2])))

    def oklch_transition(self, from_rgb, to_rgb, t):
        """Transition through OKLCh color space (cylindrical OKLab)"""
        # Convert RGB to linear RGB then to OKLab then to OKLCh
        from_linear = self.rgb_to_linear(from_rgb[0], from_rgb[1], from_rgb[2])
        to_linear = self.rgb_to_linear(to_rgb[0], to_rgb[1], to_rgb[2])

        from_oklab = self.linear_rgb_to_oklab(*from_linear)
        to_oklab = self.linear_rgb_to_oklab(*to_linear)

        from_oklch = self.oklab_to_oklch(*from_oklab)
        to_oklch = self.oklab_to_oklch(*to_oklab)

        # Interpolate in OKLCh space
        L = from_oklch[0] + (to_oklch[0] - from_oklch[0]) * t
        C = from_oklch[1] + (to_oklch[1] - from_oklch[1]) * t
        h = from_oklch[2] + (to_oklch[2] - from_oklch[2]) * t

        # Convert back to RGB
        oklab = self.oklch_to_oklab(L, C, h)
        linear_rgb = self.oklab_to_linear_rgb(*oklab)
        rgb = self.linear_to_rgb(*linear_rgb)

        # Clamp values to valid range
        return (max(0, min(255, rgb[0])), max(0, min(255, rgb[1])), max(0, min(255, rgb[2])))

    def get_transition_function(self, algorithm):
        """Get the transition function based on selected algorithm"""
        if algorithm == 'Linear RGB':
            return self.linear_rgb_transition
        elif algorithm == 'HSV (Hue-Saturation-Value)':
            return self.hsv_transition
        elif algorithm == 'HSL (Hue-Saturation-Lightness)':
            return self.hsl_transition
        elif algorithm == 'Cubic Bezier RGB':
            return self.cubic_bezier_rgb_transition
        elif algorithm == 'Exponential RGB':
            return self.exponential_rgb_transition
        elif algorithm == 'LAB (CIE L*a*b*)':
            return self.lab_transition
        elif algorithm == 'LCH (Lightness-Chroma-Hue)':
            return self.lch_transition
        elif algorithm == 'OKLab (Perceptual)':
            return self.oklab_transition
        elif algorithm == 'OKLCh (Perceptual Cylindrical)':
            return self.oklch_transition
        else:
            return self.linear_rgb_transition

    def start_transition(self):
        """Start the color transition in a separate thread"""
        if self.transition_running:
            return

        # Validate inputs
        try:
            from_color = self.from_color_var.get()
            to_color = self.to_color_var.get()
            duration = float(self.duration_var.get())

            if not (from_color.startswith('#') and len(from_color) == 7):
                raise ValueError("Invalid from color")
            if not (to_color.startswith('#') and len(to_color) == 7):
                raise ValueError("Invalid to color")
            if duration <= 0:
                raise ValueError("Duration must be positive")

            from_rgb = self.hex_to_rgb(from_color)
            to_rgb = self.hex_to_rgb(to_color)

        except Exception as e:
            print(f"Error: {e}")
            return

        self.transition_running = True
        self.go_btn.config(state='disabled')
        self.randomize_btn.config(state='disabled')

        # Run transition in separate thread
        self.transition_thread = threading.Thread(
            target=self.run_transition,
            args=(from_rgb, to_rgb, duration),
            daemon=True
        )
        self.transition_thread.start()

    def run_transition(self, from_rgb, to_rgb, duration):
        """Run the actual transition animation for all algorithms"""
        # Debug output
        print(f"Running transitions from {self.rgb_to_hex(*from_rgb)} -> {self.rgb_to_hex(*to_rgb)}")

        steps = 60  # Number of color steps to display
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break

            t = elapsed / duration

            # Update all algorithm patches simultaneously
            for algo_name, canvas in self.algorithm_patches.items():
                transition_func = self.get_transition_function(algo_name)
                current_rgb = transition_func(from_rgb, to_rgb, t)
                current_hex = self.rgb_to_hex(*current_rgb)
                self.root.after(0, self.update_patch, canvas, current_hex)

            # Sleep for smooth animation (aim for ~60 FPS)
            time.sleep(1/60)

        # Set final colors for all patches
        for algo_name, canvas in self.algorithm_patches.items():
            transition_func = self.get_transition_function(algo_name)
            final_rgb = transition_func(from_rgb, to_rgb, 1.0)
            final_hex = self.rgb_to_hex(*final_rgb)
            self.root.after(0, self.update_patch, canvas, final_hex)

        # Re-enable buttons
        self.root.after(0, self.enable_buttons)

    def update_patch(self, canvas, color):
        """Update a specific algorithm patch with the current color"""
        canvas.config(bg=color)

    def enable_buttons(self):
        """Re-enable buttons after transition completes"""
        self.transition_running = False
        self.go_btn.config(state='normal')
        self.randomize_btn.config(state='normal')


def main():
    root = tk.Tk()
    app = ColorTransitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
