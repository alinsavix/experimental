#!/usr/bin/env -S uv run --script
import tkinter as tk
from tkinter import ttk
import json
import math
import os
import pathlib
import random
import time
import threading
import urllib.request


# --- Hue configuration (loaded from .env next to this script) ---

def _load_env():
    env_path = pathlib.Path(__file__).parent / '.env'
    env = {}
    try:
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, _, val = line.partition('=')
                env[key.strip()] = val.strip()
    except FileNotFoundError:
        pass
    return env

_env = _load_env()
HUE_IP    = _env.get('HUE_IP',      '127.0.0.1')
HUE_API_KEY = _env.get('HUE_API_KEY', '')
HUE_GROUP = _env.get('HUE_GROUP',   '0')


# --- Color math ---

def srgb_to_linear(c):
    c = c / 255.0
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def linear_to_srgb(c):
    c = max(0.0, min(1.0, c))
    if c <= 0.0031308:
        return c * 12.92 * 255.0
    return (1.055 * (c ** (1.0 / 2.4)) - 0.055) * 255.0


def linear_rgb_to_oklab(lr, lg, lb):
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


def oklab_to_linear_rgb(L, a, b):
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


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(
        int(max(0, min(255, r))),
        int(max(0, min(255, g))),
        int(max(0, min(255, b))),
    )


def rgb_to_oklch(rgb):
    lr = srgb_to_linear(rgb[0])
    lg = srgb_to_linear(rgb[1])
    lb = srgb_to_linear(rgb[2])
    L, a, b_ok = linear_rgb_to_oklab(lr, lg, lb)
    C = math.sqrt(a * a + b_ok * b_ok)
    h = math.atan2(b_ok, a) * 180 / math.pi
    if h < 0:
        h += 360
    return (L, C, h)


def oklch_to_rgb(L, C, h):
    h_rad = h * math.pi / 180
    a = C * math.cos(h_rad)
    b = C * math.sin(h_rad)
    lr, lg, lb = oklab_to_linear_rgb(L, a, b)
    return (linear_to_srgb(lr), linear_to_srgb(lg), linear_to_srgb(lb))


def delta_e_oklab(rgb1, rgb2):
    """Perceptual color distance in OKLab space, scaled to ~0-100."""
    lr1, lg1, lb1 = srgb_to_linear(rgb1[0]), srgb_to_linear(rgb1[1]), srgb_to_linear(rgb1[2])
    lr2, lg2, lb2 = srgb_to_linear(rgb2[0]), srgb_to_linear(rgb2[1]), srgb_to_linear(rgb2[2])
    L1, a1, b1 = linear_rgb_to_oklab(lr1, lg1, lb1)
    L2, a2, b2 = linear_rgb_to_oklab(lr2, lg2, lb2)
    return math.sqrt((L1 - L2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2) * 100


def rgb_to_hue_xy_bri(r, g, b):
    """Convert RGB (0-255) to Philips Hue CIE xy + brightness (1-254).

    Uses the wide-gamut D65 matrix recommended by the Hue API docs.
    Y (luminance) is used directly for brightness.
    """
    lr = srgb_to_linear(r)
    lg = srgb_to_linear(g)
    lb = srgb_to_linear(b)

    # Wide gamut D65 matrix (Philips Hue developer docs)
    X = lr * 0.664511 + lg * 0.154324 + lb * 0.162028
    Y = lr * 0.283881 + lg * 0.668433 + lb * 0.047685
    Z = lr * 0.000088 + lg * 0.072310 + lb * 0.986039

    total = X + Y + Z
    if total == 0:
        return (0.3127, 0.3290, 1)  # D65 white point at minimum brightness

    x = X / total
    y = Y / total
    bri = max(1, min(254, int(Y * 254)))
    return (x, y, bri)


def _shortest_hue(fh, th, t):
    dh = th - fh
    if dh > 180:
        dh -= 360
    elif dh < -180:
        dh += 360
    return fh + dh * t


def lerp_oklch(from_rgb, to_rgb, t):
    fl, fc, fh = rgb_to_oklch(from_rgb)
    tl, tc, th = rgb_to_oklch(to_rgb)
    L = fl + (tl - fl) * t
    C = fc + (tc - fc) * t
    h = _shortest_hue(fh, th, t)
    return oklch_to_rgb(L, C, h)


def random_color():
    return f'#{random.randint(0, 0xFFFFFF):06x}'


# --- App ---

class ColorFlowApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Color Flow")
        self.root.geometry("640x520")
        self.root.minsize(500, 400)

        self.running = True
        self.paused = False
        self.current_de = 0.0
        self._last_hue_send = 0.0

        self._build_ui()

        self.thread = threading.Thread(
            target=self._run_loop, args=(random_color(),), daemon=True
        )
        self.thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        # --- Top info panel ---
        info = ttk.Frame(self.root, padding="10")
        info.pack(fill=tk.X)

        # FROM column
        from_col = ttk.Frame(info)
        from_col.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(from_col, text="FROM", font=('Arial', 9, 'bold')).pack()
        self.from_swatch = tk.Canvas(
            from_col, width=55, height=55, highlightthickness=1, highlightbackground='#888'
        )
        self.from_swatch.pack(pady=3)
        self.from_hex_lbl = ttk.Label(from_col, text="#------", font=('Courier', 10))
        self.from_hex_lbl.pack()
        self.from_lch_lbl = ttk.Label(
            from_col, text="L --  C --  H --°", font=('Arial', 8), foreground='#666'
        )
        self.from_lch_lbl.pack()

        ttk.Separator(info, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=12, pady=4)

        # Middle column: ΔE, duration, speed, hue sync
        mid = ttk.Frame(info)
        mid.pack(side=tk.LEFT, expand=True, fill=tk.X)

        ttk.Label(mid, text="ΔE (OKLab)", font=('Arial', 9)).pack()
        self.de_lbl = ttk.Label(mid, text="--", font=('Arial', 22, 'bold'))
        self.de_lbl.pack()

        ttk.Label(mid, text="Est. duration", font=('Arial', 9)).pack(pady=(6, 0))
        self.dur_lbl = ttk.Label(mid, text="-- s", font=('Arial', 13))
        self.dur_lbl.pack()

        speed_header = ttk.Frame(mid)
        speed_header.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(speed_header, text="Speed", font=('Arial', 9)).pack(side=tk.LEFT)
        self.speed_val_lbl = ttk.Label(speed_header, text="30 ΔE/s", font=('Arial', 9))
        self.speed_val_lbl.pack(side=tk.RIGHT)

        self.speed_var = tk.DoubleVar(value=30.0)
        ttk.Scale(
            mid, from_=1, to=150, variable=self.speed_var,
            orient=tk.HORIZONTAL, command=self._on_speed_change
        ).pack(fill=tk.X)

        # Pause button
        self.pause_btn = ttk.Button(mid, text="Pause", command=self._toggle_pause)
        self.pause_btn.pack(pady=(6, 0))

        # Hue sync row
        hue_row = ttk.Frame(mid)
        hue_row.pack(fill=tk.X, pady=(8, 0))
        self.hue_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            hue_row, text="Sync to Hue (group 7)", variable=self.hue_enabled
        ).pack(side=tk.LEFT)
        self.hue_status_lbl = ttk.Label(hue_row, text="", font=('Arial', 9))
        self.hue_status_lbl.pack(side=tk.LEFT, padx=(6, 0))

        hue_rate_row = ttk.Frame(mid)
        hue_rate_row.pack(fill=tk.X)
        ttk.Label(hue_rate_row, text="Hue rate", font=('Arial', 9)).pack(side=tk.LEFT)
        self.hue_rate_val_lbl = ttk.Label(hue_rate_row, text="1 upd/s", font=('Arial', 9))
        self.hue_rate_val_lbl.pack(side=tk.RIGHT)
        self.hue_rate_var = tk.DoubleVar(value=1.0)
        ttk.Scale(
            mid, from_=0.1, to=10.0, variable=self.hue_rate_var,
            orient=tk.HORIZONTAL, command=self._on_hue_rate_change
        ).pack(fill=tk.X)

        ttk.Separator(info, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=12, pady=4)

        # TO column
        to_col = ttk.Frame(info)
        to_col.pack(side=tk.LEFT, padx=(5, 0))
        ttk.Label(to_col, text="TO", font=('Arial', 9, 'bold')).pack()
        self.to_swatch = tk.Canvas(
            to_col, width=55, height=55, highlightthickness=1, highlightbackground='#888'
        )
        self.to_swatch.pack(pady=3)
        self.to_hex_lbl = ttk.Label(to_col, text="#------", font=('Courier', 10))
        self.to_hex_lbl.pack()
        self.to_lch_lbl = ttk.Label(
            to_col, text="L --  C --  H --°", font=('Arial', 8), foreground='#666'
        )
        self.to_lch_lbl.pack()

        ttk.Separator(self.root, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5)

        # Main color display
        self.canvas = tk.Canvas(self.root, bg='#808080', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def _on_hue_rate_change(self, *_):
        rate = self.hue_rate_var.get()
        self.hue_rate_val_lbl.config(text=f"{rate:.1f} upd/s")

    def _toggle_pause(self):
        self.paused = not self.paused
        self.pause_btn.config(text="Resume" if self.paused else "Pause")

    def _on_speed_change(self, *_):
        speed = self.speed_var.get()
        self.speed_val_lbl.config(text=f"{speed:.0f} ΔE/s")
        if self.current_de > 0:
            self.dur_lbl.config(text=f"{self.current_de / speed:.2f}s")

    def _set_transition_info(self, from_hex, to_hex, de):
        fl, fc, fh = rgb_to_oklch(hex_to_rgb(from_hex))
        tl, tc, th = rgb_to_oklch(hex_to_rgb(to_hex))
        speed = self.speed_var.get()

        self.from_swatch.config(bg=from_hex)
        self.from_hex_lbl.config(text=from_hex)
        self.from_lch_lbl.config(text=f"L {fl:.2f}  C {fc:.3f}  H {fh:.0f}°")

        self.to_swatch.config(bg=to_hex)
        self.to_hex_lbl.config(text=to_hex)
        self.to_lch_lbl.config(text=f"L {tl:.2f}  C {tc:.3f}  H {th:.0f}°")

        self.de_lbl.config(text=f"{de:.1f}")
        self.dur_lbl.config(text=f"{de / speed:.2f}s")
        self.speed_val_lbl.config(text=f"{speed:.0f} ΔE/s")

    # --- Hue ---

    def _maybe_send_hue(self, rgb):
        """Rate-limited fire-and-forget: sends to Hue at most every HUE_SEND_INTERVAL seconds."""
        if not self.hue_enabled.get():
            return
        now = time.time()
        if now - self._last_hue_send < 1.0 / max(self.hue_rate_var.get(), 0.01):
            return
        self._last_hue_send = now
        threading.Thread(target=self._send_hue, args=(rgb,), daemon=True).start()

    def _send_hue(self, rgb):
        try:
            x, y, bri = rgb_to_hue_xy_bri(*rgb)
            url = f"http://{HUE_IP}/api/{HUE_API_KEY}/groups/{HUE_GROUP}/action"
            rate = self.hue_rate_var.get()
            transition_time = max(0, round(10.0 / max(rate, 0.01)))  # 100ms units
            payload = json.dumps({
                "on": True,
                "xy": [round(x, 4), round(y, 4)],
                "bri": bri,
                "transitiontime": transition_time,
            }).encode('utf-8')
            req = urllib.request.Request(url, data=payload, method='PUT')
            req.add_header('Content-Type', 'application/json')
            with urllib.request.urlopen(req, timeout=2) as resp:
                result = json.loads(resp.read())
            if any('error' in entry for entry in result):
                errors = [e['error']['description'] for e in result if 'error' in e]
                self.root.after(0, self._set_hue_status, False, errors[0])
            else:
                self.root.after(0, self._set_hue_status, True, None)
        except Exception as e:
            self.root.after(0, self._set_hue_status, False, str(e))

    def _set_hue_status(self, ok, msg):
        if ok:
            self.hue_status_lbl.config(text="● OK", foreground='green')
        else:
            short = (msg or "error")[:40]
            self.hue_status_lbl.config(text=f"● {short}", foreground='red')

    # --- Main loop ---

    def _run_loop(self, start_hex):
        current_hex = start_hex

        while self.running:
            next_hex = random_color()
            from_rgb = hex_to_rgb(current_hex)
            to_rgb = hex_to_rgb(next_hex)

            de = delta_e_oklab(from_rgb, to_rgb)
            self.current_de = de
            self.root.after(0, self._set_transition_info, current_hex, next_hex, de)

            # Advance t proportionally to speed/ΔE so the perceived rate is constant
            # regardless of how far apart the colors are.
            t = 0.0
            last = time.time()
            while self.running and t < 1.0:
                if self.paused:
                    last = time.time()  # reset so paused time doesn't count as elapsed
                    time.sleep(1 / 60)
                    continue

                now = time.time()
                dt = now - last
                last = now

                speed = self.speed_var.get()
                t = min(1.0, t + speed * dt / max(de, 0.01))

                rgb = lerp_oklch(from_rgb, to_rgb, t)

                self.root.after(0, self.canvas.config, {'bg': rgb_to_hex(*rgb)})
                self._maybe_send_hue(rgb)

                time.sleep(1 / 60)

            current_hex = next_hex

    def _on_close(self):
        self.running = False
        self.root.destroy()


def main():
    root = tk.Tk()
    ColorFlowApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
