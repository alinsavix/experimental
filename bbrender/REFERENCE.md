# BBCode Reference

This file documents the BBCode tags implemented by this renderer. It is based
primarily on the contents of `src/tag-registry.js` and `src/tag-styles.js`.

## Syntax

Tags use bracket syntax:

```bbcode
[tag]text[/tag]
[tag=value]text[/tag]
[tag key=value other="quoted value"]text[/tag]
```

`[tag=value]` stores the value in the `value` argument. Boolean-style arguments
without `=` are parsed as `true`.

Unknown tags render literally and produce diagnostics on the browser console.

## Values

Color arguments accept valid browser CSS color keywords and `#RGB` or
`#RRGGBB` hex colors. Color names are normalized by lowercasing and removing
spaces, hyphens, and underscores.

Size arguments usually accept numbers, which become pixels. Image and source
layout size helpers also accept CSS units such as `px`, `em`, `rem`, `%`,
`vw`, `vh`, `vmin`, and `vmax` where noted.

Numeric arguments are clamped to the ranges listed below. Invalid numeric
values fall back to the default.

## Formatting Tags

| Tag | Arguments | Description |
| --- | --- | --- |
| `[b]...[/b]` | none | Bold text. |
| `[i]...[/i]` | none | Italic text. |
| `[u]...[/u]` | none | Underline text. |
| `[s]...[/s]` | none | Strikethrough text. |
| `[font=Georgia]...[/font]` | `value`: font family | Sets `font-family` after sanitizing the family name. |
| `[gfont=Tangerine]...[/gfont]` | `value`/`family`/`name`: Google Font family; `load`: default `true`; `weight`: optional weight list | Applies the font family and optionally injects a Google Fonts stylesheet. |
| `[size=48]...[/size]` | `value`: `1..500`, default `48` | Sets font size in pixels. |
| `[big]...[/big]` | none | Sets font size to `1.5em`. |
| `[small]...[/small]` | none | Sets font size to `0.67em`. |
| `[sub]...[/sub]` | none | Subscript text at `0.75em`. |
| `[sup]...[/sup]` | none | Superscript text at `0.75em`. |
| `[code]...[/code]` | none | Monospace font. |
| `[tt]...[/tt]` | none | Alias for [code] |
| `[dropcap]...[/dropcap]` | `value`/`lines`: `2..8`, default `3` | Makes the first non-whitespace grapheme a floating drop cap. |

## Color and Paint Tags

| Tag | Arguments | Description |
| --- | --- | --- |
| `[color=red]...[/color]` | `value`: CSS color | Sets text color. |
| `[colour=red]...[/colour]` | `value`: CSS color | Alias for `[color]`. |
| `[bg=yellow]...[/bg]` | `value`: CSS color | Sets background color. |
| `[highlight=yellow]...[/highlight]` | `value`: CSS color | Alias-style background highlight. |
| `[opacity=0.5]...[/opacity]` | `value`: `0..1`, default `1` | Sets opacity. |
| `[gradient from=red to=blue]...[/gradient]` | `from`: color, default `white`; `to`: color, default `black`; `dir`/`direction`: `horizontal` or `vertical`; `stops`: comma-separated color stops | Clips a CSS linear gradient to text. `stops` entries may be colors or `color percent` pairs such as `red 0%`. |

## Styling Tags

| Tag | Arguments | Description |
| --- | --- | --- |
| `[outline color=black size=2]...[/outline]` | `color`: default `black`; `size`/`width`: `1..20`, default `2` | Applies `-webkit-text-stroke`. |
| `[stroke color=black width=2]...[/stroke]` | `color`: default `black`; `size`/`width`: `1..20`, default `2` | Same text-stroke implementation as `[outline]`. |
| `[shadow color=gray x=2 y=2]...[/shadow]` | `color`: default `gray`; `x`: `-50..50`, default `2`; `y`: `-50..50`, default `2` | Applies a hard text shadow. |
| `[glow color=white size=8]...[/glow]` | `color`: default `white`; `size`: `1..30`, default `8` | Applies blurred text shadows. |
| `[leading=1.2]...[/leading]` | `value`: unitless `0.2..5` or CSS `px`, `em`, `rem`, `%` line-height | Sets CSS `line-height`, controlling the space between lines. |
| `[letterspacing=10]...[/letterspacing]` | `value`: `-50..100`, default `0` | Sets CSS `letter-spacing` in pixels. |
| `[class=alert]...[/class]` | `value`/`name`: class names | Adds sanitized CSS class names to the generated wrapper span. Multiple whitespace-separated class names are allowed. |
| `[blur=3]...[/blur]` | `value`: `1..20`, default `3` | Applies CSS blur. |
| `[box border=2 color=white padding=10 radius=0]...[/box]` | `border`: `0..100`, default `2`; `color`: default `white`; `padding`: `0..500`, default `10`; `radius`: `0..500`, default `0` | Draws an inline box around content. |
| `[indent=40]...[/indent]` | `value`: `0..1000`, default `40` | Displays as a block with left margin in pixels. |

## Alignment and Wrapping Tags

| Tag | Arguments | Description |
| --- | --- | --- |
| `[align=center]...[/align]` | `value`: `left`, `center`, `right`, or `justify` | Displays as a block and sets text alignment. |
| `[center]...[/center]` | none | Centers block content. |
| `[left]...[/left]` | none | Left-aligns block content. |
| `[right]...[/right]` | none | Right-aligns block content. |
| `[justify]...[/justify]` | none | Justifies block content. |
| `[wrap=word]...[/wrap]` | `value`/`mode`/`type`: `word`, `char`, `character`, or `anywhere` | Displays as a block. Character modes use `overflow-wrap: anywhere` and `word-break: break-word`; word mode uses normal wrapping. |

## Structural and Literal Tags

| Tag | Arguments | Description |
| --- | --- | --- |
| `[newline]` | none | Inserts a line break. |
| `[nl]` | none | Alias for `[newline]`. |
| `[hr]` | none | Inserts a horizontal rule. |
| `[reset]` | none | Closes all currently open parser tags for following content. |
| `[noparse]...[/noparse]` | none | Renders contained BBCode literally. |

## Image and Dynamic Tags

| Tag | Arguments | Description |
| --- | --- | --- |
| `[img src="..."]` | `src`/`value`/`url`/`path`: image URL/path; `width`/`w`: size; `height`/`h`: size; `alt`: text; `fit`: `contain`, `cover`, `fill`, `none`, or `scale-down` | Inserts an inline image. Sources may be `http`, `https`, `file`, `data:image/...`, absolute Windows paths, or relative paths beginning with `.` or `/`. |
| `[random words="Hello,Hi,Hey" speed=2]` | `words`/`value`: comma-separated words; `speed`: `0.1..60`, default `2` | Displays a randomly selected word and updates it with unseeded `Math.random()`. Quoted comma-separated values and backslash escapes are supported by the parser. |

## Animation Tags

Animation tags wrap each grapheme so effects can be applied per character.
Spaces are preserved with non-breaking space wrappers.

| Tag | Arguments | Description |
| --- | --- | --- |
| `[wave amp=50 freq=5]...[/wave]` | `amp`: `0..300`, default `50`; `freq`: `0.05..60`, default `5` | Per-character vertical wave. |
| `[bounce amp=20 freq=3]...[/bounce]` | `amp`: `0..300`, default `20`; `freq`: `0.05..60`, default `3` | Per-character bounce. |
| `[shake level=5 rate=20]...[/shake]` | `level`: `0..80`, default `5`; `rate`: `0.05..60`, default `20` | Per-character stepped jitter. |
| `[pulse freq=1 intensity=0.18]...[/pulse]` | `freq`: `0.05..60`, default `1`; `intensity`: `0..5`, default `0.18` | Per-character scale pulse. Peak scale is `1 + intensity`. |
| `[tornado radius=10 freq=1]...[/tornado]` | `radius`: `0..200`, default `10`; `freq`: `0.05..60`, default `1` | Per-character orbit around the baseline. |
| `[rainbow speed=1]...[/rainbow]` | `speed`: `0.1..20`, default `1` | Per-character color cycle. Duration is `4 / speed` seconds, minimum `0.2s`. |
| `[rotate speed=45]...[/rotate]` | `speed`: `1..1440`, default `45` | Per-character spin. `speed` is degrees per second. |
| `[blink freq=2]...[/blink]` | `freq`: `0.05..60`, default `2` | Synchronized blink. |
| `[flip axis=x speed=1]...[/flip]` | `axis`: `x` or `y`, default `x`; `speed`: `0.05..60`, default `1` | Synchronized flip on the selected axis. |
| `[metallic speed=2]...[/metallic]` | `speed`: `0.05..60`, default `2` | Moving clipped sheen using a CSS gradient. |

## Character Effect Tags

Character effect tags also wrap each grapheme. Some effects are static CSS;
others use runtime timers after rendering.

| Tag | Arguments | Description |
| --- | --- | --- |
| `[typewriter speed=8 cursor=1]...[/typewriter]` | `speed`: `0.1..120`, default `8`; `cursor`: truthy to enable cursor | Reveals characters in order. Cursor advances across the run when enabled. |
| `[hacker speed=4 loop=2 seed=hacker glyphs=...]...[/hacker]` | `speed`: `0.1..120`, default `4`; `loop`: `0..20`, default `2`; `seed`: string; `glyphs`: character set, default `ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#$%&?@` | Scrambles non-space characters through glyphs, then resolves to final text. Uses seeded deterministic randomness. |
| `[fade start=0 length=12]...[/fade]` | `start`: `0..10000`, default `0`; `length`: `1..10000`, default `12` | Static opacity ramp by character index. |
| `[fire intensity=0.5]...[/fire]` | `intensity`: `0..1`, default `0.5` | Warm color and text-shadow flicker. |
| `[electric freq=10 intensity=5]...[/electric]` | `freq`: `0.05..60`, default `10`; `intensity`: `0..80`, default `5` | Stepped jitter with bright text-shadow/glow offsets. |

## Entrance Effect Tags

These are implemented as animations on the whole tag span rather than
per-character wrappers.

| Tag | Arguments | Description |
| --- | --- | --- |
| `[slide dir=left speed=80 distance=120]...[/slide]` | `dir`/`direction`/`value`: `left`, `right`, `up`, or `down`, default `left`; `speed`: `1..2000`, default `80`; `distance`: `1..4000`, default `120` | Slides content in from the configured direction. Duration is `distance / speed`. |
| `[zoom from=0 to=1 speed=2]...[/zoom]` | `from`: `0..10`, default `0`; `to`: `0..10`, default `1`; `speed`: `0.05..60`, default `2` | Scales content from `from` to `to`. Duration is `1 / speed`, minimum `0.05s`. |
