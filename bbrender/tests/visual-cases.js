(function (global) {
  'use strict';

  global.BBCodeTestCases = [
    {
      id: 'plain',
      category: 'Core',
      name: 'Plain Text',
      bbcode: 'Plain websocket text renders safely.',
      expected: 'Plain text should display unchanged.'
    },
    {
      id: 'unknown',
      category: 'Diagnostics',
      name: 'Unknown Tag',
      bbcode: 'Before [unknown value=1]literal[/unknown] after',
      expected: 'The opening unknown tag and unmatched close should render literally and log diagnostics.'
    },
    {
      id: 'noparse',
      category: 'Special',
      name: 'No Parse',
      bbcode: '[noparse][b]this stays literal[/b][/noparse]',
      expected: 'The [b] tags should be visible as text.'
    },
    {
      id: 'reset',
      category: 'Special',
      name: 'Reset',
      bbcode: '[b]Bold section [reset]normal section',
      expected: 'The reset should close open tags for all following text.'
    },
    {
      id: 'reset',
      category: 'Special',
      name: 'Reset',
      bbcode: '[b][i]Styled[reset] plain',
      expected: 'The [reset] marker should reset the open tag stack.'
    },
    {
      id: 'newline',
      category: 'Special',
      name: 'Newline',
      bbcode: 'Line one[newline]Line two[nl]Line three',
      expected: 'The [newline] and [nl] tags should create line breaks.'
    },
    {
      id: 'leading',
      category: 'Styling',
      name: 'Leading',
      bbcode: '[leading=1.8]Wide leading[newline]Second line[newline]Third line[/leading] [leading=0.8]Tight leading[newline]Fourth line[/leading]',
      layout: { width: 680, lineHeight: 1.1 },
      expected: 'The leading tag should change line spacing within each tagged span.'
    },
    {
      id: 'malformed',
      category: 'Diagnostics',
      name: 'Malformed Nesting',
      bbcode: '[b]bold [i]italic[/b] trailing[/i]',
      expected: 'The parser should recover and report an unmatched closing tag.'
    },
    {
      id: 'global-tags',
      category: 'Core',
      name: 'Global Tags',
      bbcode: 'Wrapped by global tags',
      globalTags: '[b][color=gold]',
      expected: 'The renderer should auto-close global tags around the text.'
    },
    {
      id: 'formatting-static',
      category: 'Formatting',
      name: 'Static Formatting',
      bbcode: 'Normal [b]Bold[/b] [i]Italic[/i] [u]Underline[/u] [s]Strike[/s]',
      expected: 'Normal text should be regular weight; the bold tag should visibly increase weight.'
    },
    {
      id: 'colors-static',
      category: 'Colors',
      name: 'Colors and Highlight',
      bbcode: '[color=dodger-blue]Blue[/color] [bg=yellow][color=black]Highlight[/color][/bg] [opacity=0.45]Faded[/opacity]',
      expected: 'Named colors should normalize hyphenated names, and background/opacity should apply.'
    },
    {
      id: 'gradient-static',
      category: 'Colors',
      name: 'Gradient',
      bbcode: '[gradient stops="red 0%,gold 50%,blue 100%"]Gradient text[/gradient]',
      expected: 'Gradient text should use CSS background clipping.'
    },
    {
      id: 'styling-static',
      category: 'Styling',
      name: 'Outline Shadow Glow',
      bbcode: '[outline color=black size=2][shadow color=gray x=3 y=3][glow color=gold size=8]Readable glow[/glow][/shadow][/outline]',
      expected: 'Outline, shadow, and glow should stack through nested spans.'
    },
    {
      id: 'stroke-static',
      category: 'Styling',
      name: 'Stroke Tag',
      bbcode: '[stroke color=black width=3]Stroked text[/stroke] [outline color=black size=2]Outlined text[/outline]',
      expected: 'Stroke and outline should both draw an outline around the text using CSS text stroke.'
    },
    {
      id: 'box-hr',
      category: 'Styling',
      name: 'Box and Rule',
      bbcode: '[box border=2 color=gold radius=8 padding=12]Boxed text[/box][newline][hr]',
      expected: 'Box should render inline-block styling, then a line break and rule.'
    },
    {
      id: 'wave-bounce',
      category: 'Animation',
      name: 'Wave and Bounce',
      bbcode: '[wave amp=24 freq=2][bounce amp=12 freq=3]Nested animation keeps spaces[/bounce][/wave]',
      expected: 'Characters should be individually wrapped and visibly animated with nested transforms. Spaces between words should remain visible.'
    },
    {
      id: 'rainbow',
      category: 'Animation',
      name: 'Rainbow',
      bbcode: '[rainbow speed=2]Color cycling rainbow spaces[/rainbow]',
      expected: 'Characters should cycle through rainbow colors with staggered offsets and preserve word spaces.'
    },
    {
      id: 'rotate-metallic',
      category: 'Animation',
      name: 'Rotate Metallic',
      bbcode: '[rotate speed=90]Spin[/rotate] [metallic speed=2][color=gold]Gold sheen[/color][/metallic]',
      expected: 'Rotate should spin each character, and metallic should show a moving clipped sheen.'
    },
    {
      id: 'blink',
      category: 'Animation',
      name: 'Blink',
      bbcode: '[blink freq=1]Whole tag blink spaces[/blink]',
      expected: 'The entire tag contents should blink together, not ripple character by character, and spaces should remain visible.'
    },
    {
      id: 'pulse',
      category: 'Animation',
      name: 'Pulse',
      bbcode: '[pulse freq=1 intensity=0.35]Pulse intensity test[/pulse]',
      expected: 'Characters should scale up and down using the configured pulse intensity.'
    },
    {
      id: 'shake-pulse-flip',
      category: 'Animation',
      name: 'Shake Pulse Flip',
      bbcode: '[shake rate=10 level=3][pulse freq=1 intensity=0.35][flip axis=y speed=0.5]Energy[/flip][/pulse][/shake]',
      expected: 'Nested wrappers should let transform animations compose visually.'
    },
    {
      id: 'flip-x-y',
      category: 'Animation',
      name: 'Flip X and Y',
      bbcode: '[flip axis=x speed=0.6]Flip X spaces[/flip]  [flip axis=y speed=0.6]Flip Y spaces[/flip]',
      expected: 'Both flip variants should animate their whole tag contents together and preserve spaces.'
    },
    {
      id: 'tornado',
      category: 'Animation',
      name: 'Tornado',
      bbcode: '[tornado radius=18 freq=0.75]Orbiting letters keep spaces[/tornado]',
      expected: 'Characters should orbit around their baseline positions and preserve word spaces.'
    },
    {
      id: 'typewriter',
      category: 'Effects',
      name: 'Typewriter',
      bbcode: '[typewriter speed=8 cursor=1 loop=1]Typed reveal keeps spaces[/typewriter]',
      expected: 'Characters should reveal in order with a blinking cursor at the end of the run.'
    },
    {
      id: 'hacker',
      category: 'Effects',
      name: 'Hacker',
      bbcode: '[hacker speed=8 loop=2 seed=demo]Decoded text keeps spaces[/hacker]',
      expected: 'Characters should start scrambled, then resolve deterministically to the final text while preserving spaces.'
    },
    {
      id: 'fade',
      category: 'Effects',
      name: 'Fade Ramp',
      bbcode: '[fade start=3 length=12]First letters solid, later letters fade[/fade]',
      expected: 'The opacity should ramp down per character after the configured start index.'
    },
    {
      id: 'fire-electric',
      category: 'Effects',
      name: 'Fire Electric',
      bbcode: '[fire intensity=0.7]Burning[/fire] [electric freq=10 intensity=5][color=#89b4fa]Electric[/color][/electric]',
      expected: 'Fire should flicker through warm colors, while electric jitters with bright sparks.'
    },
    {
      id: 'slide-directions',
      category: 'Effects',
      name: 'Slide Directions',
      bbcode: '[slide dir=left speed=80]Left[/slide] [slide dir=right speed=80]Right[/slide][newline][slide dir=up speed=80]Up[/slide] [slide dir=down speed=80]Down[/slide]',
      expected: 'Each span should enter from its configured direction.'
    },
    {
      id: 'zoom',
      category: 'Effects',
      name: 'Zoom',
      bbcode: '[zoom from=0 to=1 speed=2]Zoom in[/zoom]',
      expected: 'The span should scale in from the configured start size.'
    },
    {
      id: 'root-transition',
      category: 'Effects',
      name: 'Root Transition',
      bbcode: 'Root transition uses payload-style options',
      duration: 5000,
      transition: { in: 'zoom', out: 'fade', inTime: 450, outTime: 700 },
      expected: 'The whole rendered message should zoom in, hold, then fade out near the end of its duration.'
    },
    {
      id: 'dropcap',
      category: 'Layout',
      name: 'Dropcap',
      bbcode: '[dropcap]Signal starts with a large first letter and then wraps into normal body text over multiple lines.[/dropcap]',
      layout: { width: 520, lineHeight: 1.18 },
      expected: 'The first visible grapheme should float as a large dropcap and the remaining text should wrap beside it.'
    },
    {
      id: 'dropcap-lines',
      category: 'Layout',
      name: 'Dropcap Lines',
      bbcode: '[dropcap lines=4]Four-line dropcap sizing should be visibly larger than the default dropcap case.[/dropcap]',
      layout: { width: 520, lineHeight: 1.18 },
      expected: 'The dropcap should use the configured line count to control its size.'
    },
    {
      id: 'wrap-word',
      category: 'Layout',
      name: 'Word Wrap',
      bbcode: '[wrap=word]Words should wrap only at normal boundaries inside this deliberately narrow source width.[/wrap]',
      layout: { width: 360, height: 160, padding: 12, lineHeight: 1.2 },
      expected: 'The source should be constrained and wrap at word boundaries.'
    },
    {
      id: 'wrap-char',
      category: 'Layout',
      name: 'Character Wrap',
      bbcode: '[wrap=char]SupercalifragilisticexpialidociousKeepsGoingWithoutSpaces[/wrap]',
      layout: { width: 300, height: 160, padding: 12, lineHeight: 1.2 },
      expected: 'The long unbroken word should wrap within the constrained source width.'
    },
    {
      id: 'auto-padding',
      category: 'Layout',
      name: 'Auto Padding Stress',
      bbcode: '[glow color=gold size=18][wave amp=36 freq=2]Glow and wave should have extra breathing room[/wave][/glow]',
      layout: { width: 720, height: 220, autoPadding: true, lineHeight: 1.1 },
      expected: 'Auto padding should add enough source padding to reduce clipping from glow and vertical animation.'
    },
    {
      id: 'google-font',
      category: 'Assets',
      name: 'Google Font',
      bbcode: '[gfont=Tangerine]Google font text[/gfont]',
      expected: 'The renderer should apply the requested font-family and inject a Google Fonts stylesheet link when the browser can load it.'
    },
    {
      id: 'random-words',
      category: 'Dynamic',
      name: 'Random Words',
      bbcode: 'Random greeting: [random words="Hello,Hi,Hey,Yo" speed=2]',
      expected: 'The dynamic span should display one of the configured words and update at random.'
    },
    {
      id: 'inline-image',
      category: 'Assets',
      name: 'Inline Image',
      bbcode: 'Inline icon [img src="data:image/svg+xml,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 32 32%27%3E%3Crect width=%2732%27 height=%2732%27 rx=%276%27 fill=%27gold%27/%3E%3Ccircle cx=%2716%27 cy=%2716%27 r=%278%27 fill=%27black%27/%3E%3C/svg%3E" width=48 height=48] sits on the baseline.',
      expected: 'The image should render as an inline atomic element with the requested size.'
    },
    {
      id: 'payload-command',
      category: 'WebSocket',
      name: 'Mock Payload',
      payload: { type: 'bbcode.render', data: { bbcode: '[b]Payload text[/b]' } },
      expected: 'The mock websocket path should render the bbcode field without a real websocket message.'
    }
  ];
})(window);
