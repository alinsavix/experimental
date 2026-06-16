# mixerlint

A lint checker for Behringer XR18 (and X32-family) mixers. Connects over OSC,
reads the current mixer state, and reports things that look wrong — channels
panned off-center, stereo pairs with mismatched faders or wrong pan positions,
sends that should be silent but aren't, and so on.

**Read-only.** The tool never sends any command that changes a mixer parameter.

---

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) — handles the virtualenv and
  dependencies automatically
- The mixer reachable on the network (default: `192.168.0.28:10024`)

Dependencies (`rich`, `pyyaml`) are declared inline and installed automatically
by `uv run`.

---

## Quick start

```
uv run mixerlint.py
```

That's it. `uv` installs the dependencies on first run, connects to the mixer
address in `mixerlint.yaml`, runs all enabled checks, and prints any issues
found. Exit code is `0` if clean, `1` if any errors or warnings were found
(useful for scripting).

---

## Command-line options

| Flag | Description |
|---|---|
| `--config FILE` | Config file to use (default: `mixerlint.yaml`) |
| `--host IP` | Mixer IP address, overrides the config file |
| `--port N` | Mixer OSC port, overrides the config file |
| `--summary` | Print a channel/bus state table before the lint results |
| `--verbose` | Show a "Checks Run" table listing every check and its result (even checks that pass or are skipped); also adds a "Pan raw" column to the summary with the raw OSC float value (0.0–1.0) |
| `--simulate` | Generate a sample report from synthetic data that triggers at least one of every check — no mixer or config file needed |
| `--timeout SEC` | Per-query OSC timeout in seconds (default: 0.3) |

### Examples

```sh
# Basic lint with default config
uv run mixerlint.py

# Show channel table too
uv run mixerlint.py --summary

# Show every check that ran (plus raw OSC pan values) for diagnosis
uv run mixerlint.py --verbose

# Preview a sample report covering every warning type (no mixer needed)
uv run mixerlint.py --simulate

# Use a different config (e.g. for a different show)
uv run mixerlint.py --config show-saturday.yaml

# Override host without editing the config
uv run mixerlint.py --host 10.0.0.5
```

---

## OSC protocol notes

The XR18 uses OSC over UDP on port **10024** (the X32 uses 10023). The tool
binds a local UDP socket on port **10025**, sends parameter-query packets to the
mixer, and listens for responses on the same socket. All 16 channels and 6 buses
are queried in a single batch (~275 addresses).

Pan values are OSC floats in the range `0.0` (full left / L100) to `1.0` (full
right / R100), with `0.5` as center. Fader/level values are also `0.0`–`1.0`
but on a non-linear dB curve where `0.75 ≈ 0 dB` unity and `0.0 = -∞`.

---

## Configuration

All configuration lives in `mixerlint.yaml`. The file has two top-level keys:
`mixer` (connection settings) and `checks` (lint rules).

### `mixer` section

```yaml
mixer:
  host: "192.168.0.28"
  port: 10024    # XR18 OSC port (X32 uses 10023)
  timeout: 0.3   # seconds to wait per OSC response
```

### `checks` section

Each check is a named key under `checks`. Omit the key entirely (or set
`enabled: false`) to skip that check. The checks are described below.

---

## Checks reference

### `pan_center`

Warns when a mono (non-linked) channel's pan is not centered. Useful for
catching a channel that got nudged off-center accidentally.

Channels that appear in any `stereo_balance` pair are automatically skipped —
they are expected to be panned hard left or right.

```yaml
checks:
  pan_center:
    enabled: true
    pan_tolerance: 0.03       # OSC units each side of 0.5; 0.03 ≈ L12/R12
    skip_muted_channels: true
    skip_channels: []         # 1-based channel numbers to always skip
```

---

### `stereo_balance`

For explicitly named L/R channel pairs, warns if the two main faders differ by
more than `tolerance_db`. Useful when you have manually paired stereo sources
(computer playback, keyboard, etc.) that are not hardware-linked on the mixer.

Channels listed here are also automatically excluded from `pan_center`.

```yaml
checks:
  stereo_balance:
    enabled: true
    tolerance_db: 3.0
    pairs:
      - left: 5
        right: 6
        name: "Keyboards"
      - left: 7
        right: 8
        name: "Backing track"
```

Each pair entry supports:

| Key | Description |
|---|---|
| `left` | Channel number of the left side (1-based) |
| `right` | Channel number of the right side |
| `name` | Label used in reports |

---

### `stereo_pan`

For each pair defined in `stereo_balance`, verifies that the left channel is
panned fully left and the right channel is fully right. Catches the common
mistake of accidentally nudging a pair member off its hard-panned position.

The `pairs` list is inherited automatically from `stereo_balance` — you do not
need to repeat it.

```yaml
checks:
  stereo_pan:
    enabled: true
    pan_tolerance: 0.03        # how far from the expected position before warning
    expected_left_pan: 0.0     # 0.0 = L100 (full left)
    expected_right_pan: 1.0    # 1.0 = R100 (full right)
    skip_muted_channels: true
```

Per-pair overrides are possible directly in the `stereo_balance` pairs list:

```yaml
    pairs:
      - left: 9
        right: 10
        name: "Room mics"
        expected_left_pan: 0.25   # only L50, not full left
        expected_right_pan: 0.75  # only R50, not full right
        pan_tolerance: 0.05
```

---

### `linked_balance`

For channels that are hardware-linked on the mixer (the XR18's stereo-link
feature), checks that their faders match. On a properly linked pair the mixer
mirrors fader moves automatically, so a significant difference would be unusual.

```yaml
checks:
  linked_balance:
    enabled: true
    tolerance_db: 1.0
    skip_muted_channels: true
```

---

### `bus_levels`

Checks that specific channel→bus send levels are at expected values. Configure
one rule per logical constraint.

```yaml
checks:
  bus_levels:
    enabled: true
    rules:
      - description: "Kick/snare not in reverb bus"
        channels: [1, 2]
        buses: [5]
        expect: "zero"
        max_level_db: -60

      - description: "Lead vocal present in all monitors"
        channels: [3]
        buses: [1, 2, 3, 4]
        expect: "nonzero"
        min_level_db: -20

      - description: "Bass DI not routed to FX bus"
        channels: [4]
        buses: [6]
        expect: "off"
```

Each rule supports:

| Key | Default | Description |
|---|---|---|
| `description` | `""` | Label shown in the report |
| `channels` | required | List of 1-based channel numbers |
| `buses` | required | List of 1-based bus numbers |
| `expect` | `"zero"` | `"zero"`, `"nonzero"`, or `"off"` |
| `max_level_db` | `-60` | Used by `expect: "zero"` — warn if above this |
| `min_level_db` | `-60` | Used by `expect: "nonzero"` — warn if below this |
| `skip_muted_channels` | `false` | Skip rule for muted channels |

**`expect` values:**

- `"zero"` — the bus send level must be at or below `max_level_db` (or fully off)
- `"nonzero"` — the bus send level must be above `min_level_db` (info-level issue if not)
- `"off"` — the bus send switch must be in the OFF position

---

### `fader_range`

Flags channels or buses whose main fader is outside a "normal" operating range.
Catches things like a fader accidentally pushed to +10 dB or left near the
bottom when it should be active.

```yaml
checks:
  fader_range:
    enabled: true
    max_db: 6.0       # warn if fader is above this
    min_db: -40.0     # info if fader is below this (but not -inf)
    skip_muted_channels: true
    skip_channels: []
```

---

### `unnamed_channels`

Flags channels that have no name set. Easy to leave a scratch input unnamed.

```yaml
checks:
  unnamed_channels:
    enabled: true
    only_active_channels: true  # skip channels whose fader is at -inf
```

---

### `muted_fader_up`

Flags channels that are muted but have their fader significantly up. This is not
always wrong (you might intentionally mute a ready channel), but is often a
forgotten mute or a source of confusion on stage.

```yaml
checks:
  muted_fader_up:
    enabled: true
    min_fader_db: -10.0   # only flag if fader is at or above this level
    skip_channels: []
```

---

## Severity levels

| Level | Meaning |
|---|---|
| `warn` | Something is probably wrong and should be investigated |
| `info` | Noteworthy but might be intentional — review before the show |

The exit code is `1` if any `warn` (or `error`) issues are found, `0` if the
run is clean.

---

## Multiple config files

You can keep separate configs for different setups and pass them with `--config`:

```sh
uv run mixerlint.py --config rehearsal.yaml   # relaxed checks
uv run mixerlint.py --config showday.yaml     # strict checks, full bus rules
```

A minimal config that only checks pan and fader range looks like:

```yaml
mixer:
  host: "192.168.0.28"

checks:
  pan_center:
    enabled: true
  fader_range:
    enabled: true
    max_db: 3.0
```

Any check key that is absent from the config is simply not run.
