# bbrender

Streamer.bot BBCode browser-source renderer.

`sb_bbrender.html` connects to Streamer.bot's WebSocket server, listens for
`General.Custom` events, and renders BBCode text into a transparent browser
source. The renderer can also be exercised directly from the browser console
for local testing.

## Quick Start

Install dependencies once:

```powershell
npm install
npm run install:browsers
```

Serve the source version locally:

```powershell
npm run serve
```

Then open:

```text
http://127.0.0.1:4173/sb_bbrender.html
```

For OBS, use either the local URL above during development or a built artifact
from `dist/`. The source page loads `bbrender.js` and the vendored
Streamer.bot client separately. The packed build inlines everything into one
HTML file.

## URL Options

The browser-source URL accepts these query parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `host` | `127.0.0.1` | Streamer.bot WebSocket host. |
| `port` | `8085` | Streamer.bot WebSocket port. |
| `secure` | `false` | Set to `true` to use `wss`; otherwise `ws`. |
| `endpoint` | `/` | WebSocket endpoint path. |
| `password` | empty | Base64-encoded Streamer.bot WebSocket password. |
| `duration` | `4000` | Default message display duration in milliseconds. |
| `mode` | `replace` | Set to `queue` to queue messages instead of replacing. |
| `globalTags` | empty | BBCode opening tags wrapped around every message. |
| `diagnostics` | off | Show parse diagnostics on the page. |
| `debug` | off | Show diagnostics and console debug output. |

Layout URL options:

| Parameter | Alias | Description |
| --- | --- | --- |
| `sourceWidth` | `width` | Message box width. Number values become pixels. |
| `sourceHeight` | `height` | Message box height. Number values become pixels. |
| `padding` |  | Message box padding. Number values become pixels. |
| `autoPadding` |  | Add estimated padding for motion/glow/stroke clipping. |
| `lineHeight` | `lineSpacing` | CSS line-height value. |
| `fontSize` | `baseFontSize` | Message font size. Number values become pixels. |
| `anchor` |  | Message position anchor. |
| `x` |  | Horizontal offset from the anchor. |
| `y` |  | Vertical offset from the anchor. |

Supported anchors:

```text
center
top
left
right
bottom
top-left
top-right
bottom-left
bottom-right
```

Example:

```text
http://127.0.0.1:4173/sb_bbrender.html?duration=6000&sourceWidth=900&sourceHeight=240&fontSize=72&anchor=bottom-right&x=40&y=40
```

## Request Envelope

Incoming WebSocket payloads must be `General.Custom` payload objects with
exactly two top-level keys: `type` and `data`. All parameters and settings
nest under `data`:

```js
{
  type: "bbcode.render",
  data: { ... }
}
```

The renderer ignores payloads whose `type` is not `bbcode.render`. Fields
placed at the top level instead of inside `data` are ignored.

### Text Fields

Message text can be supplied in any of these `data` fields:

```js
{
  type: "bbcode.render",
  data: {
    bbcode: "[b]Hello[/b]"
  }
}
```

Accepted text aliases:

```text
data.bbcode
data.text
data.message
data.content
data.value
data.displayText
```

The first non-empty string in that order is rendered.

### Per-Message Options

Per-message duration, layout, and root transitions can be sent in the same
`data` object as the text:

```js
{
  type: "bbcode.render",
  data: {
    bbcode: "[b][color=gold]Alert[/color][/b]",
    duration: 5000,
    fontSize: 72,
    sourceWidth: 900,
    sourceHeight: 260,
    anchor: "top-left",
    x: 120,
    y: 80,
    transition: {
      in: "zoom",
      out: "fade",
      inTime: 400,
      outTime: 600,
      scale: 0.08
    }
  }
}
```

Duration fields:

| Field | Description |
| --- | --- |
| `duration` | Display duration in milliseconds. |
| `ms` | Alias for `duration`. |

Layout fields sit directly inside `data`:

| Field | Alias | Description |
| --- | --- | --- |
| `sourceWidth` | `width` | Message box width. |
| `sourceHeight` | `height` | Message box height. |
| `padding` |  | Message box padding. |
| `autoPadding` |  | Enables estimated padding when truthy. |
| `lineHeight` | `lineSpacing` | CSS line-height value. |
| `fontSize` | `baseFontSize` | Message font size. |
| `anchor` |  | Message position anchor. |
| `x` |  | Horizontal anchor offset. |
| `y` |  | Vertical anchor offset. |

Nested `layout: { ... }` is not currently read from WebSocket payloads.

### Root Transitions

Root transitions animate the whole message in and/or out over its display
duration. Transition settings can be sent inside `data.transition`, inside
`data.animation`, or as aliases directly inside `data`.

Supported transition presets:

```text
fade
zoom
slide-left
slide-right
slide-up
slide-down
```

Transition fields:

| Field | Aliases | Description |
| --- | --- | --- |
| `in` | `enter`, `entrance`, `transitionIn`, `animationIn` | Entrance preset. |
| `out` | `exit`, `leave`, `transitionOut`, `animationOut` | Exit preset. |
| `inTime` | `enterTime`, `entranceTime` | Entrance duration in milliseconds. |
| `outTime` | `exitTime`, `leaveTime` | Exit duration in milliseconds. |
| `duration` |  | Total message duration override. |
| `delay` |  | Entrance delay in milliseconds. |
| `ease` |  | `linear`, `in`, `out`, or `in-out`. |
| `origin` |  | CSS transform origin, default `50% 50%`. |
| `scale` |  | Shared zoom scale fallback. |
| `inScale` | `enterScale` | Entrance zoom scale. |
| `outScale` | `exitScale` | Exit zoom scale. |

Example with aliases directly inside `data`:

```js
{
  type: "bbcode.render",
  data: {
    bbcode: "Sliding message",
    duration: 5000,
    transitionIn: "slide-left",
    transitionOut: "fade",
    enterTime: 300,
    exitTime: 500
  }
}
```

## Commands

Commands are selected from the first available `data` field:

```text
data.command
data.action
data.requestType
```

Supported commands:

| Command | Fields | Description |
| --- | --- | --- |
| `clear` |  | Clears the current message and queue immediately. |
| `set_global_tags` | `tags`, `globalTags` | Sets BBCode opening tags wrapped around future messages. |
| `set_duration` | `duration`, `ms` | Sets the default display duration. |
| `set_layout` | layout fields | Sets the default source layout. |
| `set_preset` | `name`, `value`, `tags` | Stores an in-memory preset value for later preset support. |

Examples:

```js
bbrender.receivePayload({
  type: "bbcode.render",
  data: {
    command: "set_global_tags",
    tags: "[b][color=gold]"
  }
});

bbrender.receivePayload({
  type: "bbcode.render",
  data: {
    command: "set_layout",
    width: 900,
    height: 240,
    fontSize: 72,
    anchor: "bottom-right",
    x: 40,
    y: 40
  }
});

bbrender.receivePayload({
  type: "bbcode.render",
  data: { command: "clear" }
});
```

## Local Payload Testing

`sb_bbrender.html` exposes a browser-console helper:

```js
window.bbrender.receivePayload({
  type: "bbcode.render",
  data: {
    bbcode: "[fire][shake rate=10 level=2]Test[/shake][/fire]"
  }
});
```

This goes through the same payload handling path as a Streamer.bot
`General.Custom` event.

## Supported BBCode Tags

Current implemented tags:

```text
align
b
bg
big
blink
blur
bounce
box
center
class
code
color / colour
dropcap
electric
fade
fire
flip
font
gfont
glow
gradient
hacker
highlight
hr
i
img
indent
justify
left
leading
letterspacing
nl
newline
noparse
opacity
outline / stroke
pulse
rainbow
random
reset
right
rotate
s
shadow
shake
size
slide
small
sub
sup
tornado
tt
typewriter
u
wave
wrap
zoom
metallic
```

Unknown tags render literally and produce diagnostics.

`[class=name]...[/class]` adds `name` as a CSS class on the generated tag
span. Use this when external overlay CSS should style a BBCode range.

## Current Limitations

- Queue mode is configured by URL only: `?mode=queue`.
- `globalTags` is a source-level setting. It is set by URL or the
  `set_global_tags` command, not as a per-message option.
- WebSocket payload layout options must be top-level fields. Nested
  `layout: { ... }` is currently ignored.
- `set_preset` stores values in memory for later preset support, but preset
  expansion is not implemented yet.
- The docs in `docs/` are upstream reference material and may describe tags
  that this renderer has not implemented yet. `FEATURES.md` tracks coverage.

## Visual Harness

Serve the project and open:

```text
http://127.0.0.1:4173/tests/visual-harness.html
```

Useful modes:

```text
/tests/visual-harness.html
/tests/visual-harness.html?mode=sequence
/tests/visual-harness.html?mode=tests
/tests/visual-harness.html?case=fire-electric
```

The harness uses `tests/visual-cases.js`. Browser automation verifies that the
renderer test mode passes and that key static cases match screenshots.

## Development Commands

Make targets are available for the common workflow:

```powershell
make check
make build
make test
make visual-test
make serve
make run-cdn
make vendor-check
```

Use `PORT=5000 make serve` to change the local server port.
Use `make run-tests` and open `/tests/visual-harness.html` for manual visual
review.

Run tests only:

```powershell
npm run test
```

Run automated browser and visual screenshot tests:

```powershell
npm run test:browser
npm run test:visual
```

Generate or refresh screenshot baselines after intentional visual changes:

```powershell
npm run test:visual:update
```

Build artifacts only:

```powershell
npm run build
```

Run tests and build all artifact sets:

```powershell
npm run check
```

The same check can be run without npm:

```powershell
node scripts/check.mjs
```

Check whether the vendored Streamer.bot client is current:

```powershell
npm run vendor:check
```

Update the vendored Streamer.bot client to the latest npm version:

```powershell
npm run vendor:update
```

## Build Output

`npm run build` creates:

- `dist/packed/sb_bbrender.html` - packed Streamer.bot adapter output with bbrender CSS, local renderer/app JS, and the vendored Streamer.bot client inlined into one HTML file.
- `dist/separate/sb_bbrender.html` - separate-file Streamer.bot adapter output.
- `dist/separate/bbrender.css` - separate overlay CSS.
- `dist/separate/bbrender.js` - bundled local renderer/app JS plus the vendored Streamer.bot client.
- `dist/cdn/sb_bbrender.html` - CDN entrypoint that references content-hashed assets.
- `dist/cdn/assets/bbrender.<hash>.css` - immutable CDN CSS asset.
- `dist/cdn/assets/bbrender.<hash>.js` - immutable CDN JS asset with the vendored Streamer.bot client included.
- `dist/cdn/manifest.json` - version, entrypoint, asset names, SHA-256 hashes, and SRI values.

For CDN deployment, upload the full `dist/cdn` directory. Cache
`sb_bbrender.html` and `manifest.json` with short-lived or revalidated headers,
and cache files under `assets/` with long-lived immutable headers.

## Vendored Streamer.bot Client

The Streamer.bot browser client is vendored at `vendor/streamerbot-client.js`,
with version and hash metadata in `vendor/streamerbot-client.json`. The source
`sb_bbrender.html` loads this local vendor file for direct browser use. Build
outputs inline it into `dist/packed/sb_bbrender.html` and bundle it into the
separate/CDN JavaScript files, so generated artifacts do not depend on jsDelivr
or npm at runtime.

`npm run vendor:check` compares the vendored version against the latest npm
dist-tag and exits non-zero when an update is available. `npm run
vendor:update` downloads the latest pinned file and refreshes the metadata; run
`npm run check` after updating before publishing artifacts.
