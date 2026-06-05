import { clampNumber, cssSafeIdent, readLineHeightOption, readSizeOption } from './utils.js';

export const TAG_STYLE_HANDLERS = {
  b(el) {
    el.style.fontWeight = '700';
  },
  i(el) {
    el.style.fontStyle = 'italic';
  },
  u(el) {
    el.style.textDecorationLine = appendTextDecoration(el.style.textDecorationLine, 'underline');
  },
  s(el) {
    el.style.textDecorationLine = appendTextDecoration(el.style.textDecorationLine, 'line-through');
  },
  font(el, attrs) {
    el.style.fontFamily = sanitizeFontFamily(attrs.value);
  },
  gfont: applyGoogleFont,
  class: applyCssClass,
  size(el, attrs) {
    el.style.fontSize = clampNumber(attrs.value, 1, 500, 48) + 'px';
  },
  big(el) {
    el.style.fontSize = '1.5em';
  },
  small(el) {
    el.style.fontSize = '0.67em';
  },
  sub(el) {
    el.style.verticalAlign = 'sub';
    el.style.fontSize = '0.75em';
  },
  sup(el) {
    el.style.verticalAlign = 'super';
    el.style.fontSize = '0.75em';
  },
  code: applyMonospace,
  tt: applyMonospace,
  color(el, attrs) {
    applyColor(el, 'color', attrs.value);
  },
  colour(el, attrs) {
    applyColor(el, 'color', attrs.value);
  },
  opacity(el, attrs) {
    el.style.opacity = String(clampNumber(attrs.value, 0, 1, 1));
  },
  bg(el, attrs) {
    applyColor(el, 'backgroundColor', attrs.value);
  },
  highlight(el, attrs) {
    applyColor(el, 'backgroundColor', attrs.value);
  },
  gradient: applyGradient,
  outline: applyOutline,
  stroke: applyOutline,
  shadow(el, attrs) {
    applyShadow(el, attrs, false);
  },
  glow(el, attrs) {
    applyShadow(el, attrs, true);
  },
  letterspacing(el, attrs) {
    el.style.letterSpacing = clampNumber(attrs.value, -50, 100, 0) + 'px';
  },
  align(el, attrs) {
    applyAlign(el, attrs.value);
  },
  center(el) {
    applyAlign(el, 'center');
  },
  left(el) {
    applyAlign(el, 'left');
  },
  right(el) {
    applyAlign(el, 'right');
  },
  justify(el) {
    applyAlign(el, 'justify');
  },
  leading(el, attrs) {
    const lineHeight = readLineHeightOption(attrs.value);
    if (lineHeight) el.style.lineHeight = lineHeight;
  },
  indent(el, attrs) {
    el.style.display = 'block';
    el.style.marginLeft = clampNumber(attrs.value, 0, 1000, 40) + 'px';
  },
  box: applyBox,
  blur(el, attrs) {
    el.style.filter = 'blur(' + clampNumber(attrs.value, 1, 20, 3) + 'px)';
  },
  dropcap(el, attrs) {
    el.style.display = 'block';
    el.style.textAlign = 'left';
    el.style.setProperty('--bb-dropcap-lines', String(Math.round(clampNumber(attrs.lines || attrs.value, 2, 8, 3))));
  },
  wrap(el, attrs) {
    applyWrap(el, attrs.value || attrs.mode || attrs.type);
  },
  typewriter: applyInlineBlock,
  hacker: applyInlineBlock,
  fire: applyInlineBlock,
  electric: applyInlineBlock,
  fade: applyInlineBlock,
  slide: applySlide,
  zoom: applyZoom
};

function applyMonospace(el) {
  el.style.fontFamily = 'Consolas, Monaco, "Courier New", monospace';
}

function applyInlineBlock(el) {
  el.style.display = 'inline-block';
}

function applyCssClass(el, attrs) {
  const names = String(attrs.value || attrs.name || '')
    .split(/\s+/)
    .map((name) => name.replace(/[^a-z0-9_-]/gi, ''))
    .filter(Boolean);
  if (names.length) el.classList.add(...names);
}

export function applyAnimationStyles(el, context, index) {
  const name = context.name;
  const attrs = context.attrs || {};
  const staggered = !['blink', 'flip'].includes(name);
  const delay = staggered ? -index * 0.055 : 0;

  el.style.display = 'inline-block';
  el.style.willChange = 'transform, opacity, color, filter';
  el.style.transformOrigin = '50% 55%';
  el.style.setProperty('--bb-delay', delay + 's');

  switch (name) {
    case 'wave':
      el.style.setProperty('--bb-wave-amp', clampNumber(attrs.amp, 0, 300, 50) + 'px');
      setAnimation(el, 'bb-wave', cycleDuration(attrs.freq, 5), 'ease-in-out');
      break;
    case 'bounce':
      el.style.setProperty('--bb-bounce-amp', clampNumber(attrs.amp, 0, 300, 20) + 'px');
      setAnimation(el, 'bb-bounce', cycleDuration(attrs.freq, 3), 'cubic-bezier(.3,0,.2,1)');
      break;
    case 'shake':
      el.style.setProperty('--bb-shake-level', clampNumber(attrs.level, 0, 80, 5) + 'px');
      setAnimation(el, 'bb-shake', cycleDuration(attrs.rate, 20), 'steps(2, end)');
      break;
    case 'pulse':
      el.style.setProperty('--bb-pulse-intensity', String(clampNumber(attrs.intensity, 0, 5, 0.18)));
      setAnimation(el, 'bb-pulse', cycleDuration(attrs.freq, 1), 'ease-in-out');
      break;
    case 'tornado':
      el.style.setProperty('--bb-tornado-radius', clampNumber(attrs.radius, 0, 200, 10) + 'px');
      setAnimation(el, 'bb-tornado', cycleDuration(attrs.freq, 1), 'linear');
      break;
    case 'rainbow':
      setAnimation(el, 'bb-rainbow', Math.max(0.2, 4 / clampNumber(attrs.speed, 0.1, 20, 1)), 'linear');
      break;
    case 'rotate':
      setAnimation(el, 'bb-rotate', Math.max(0.05, 360 / clampNumber(attrs.speed, 1, 1440, 45)), 'linear');
      break;
    case 'metallic':
      el.style.backgroundImage = 'linear-gradient(105deg, color-mix(in srgb, currentColor 55%, black) 0%, currentColor 30%, color-mix(in srgb, currentColor 15%, white) 44%, #ffffff 52%, currentColor 64%, color-mix(in srgb, currentColor 45%, black) 100%)';
      el.style.backgroundSize = '240% 100%';
      el.style.webkitBackgroundClip = 'text';
      el.style.backgroundClip = 'text';
      el.style.webkitTextFillColor = 'transparent';
      setAnimation(el, 'bb-metallic', Math.max(0.05, 1 / clampNumber(attrs.speed, 0.05, 60, 2)), 'linear');
      break;
    case 'blink':
      setAnimation(el, 'bb-blink', cycleDuration(attrs.freq, 2), 'steps(1, end)');
      break;
    case 'flip':
      if (String(attrs.axis || 'x').toLowerCase() === 'y') {
        setAnimation(el, 'bb-flip-y', cycleDuration(attrs.speed, 1), 'ease-in-out');
      } else {
        setAnimation(el, 'bb-flip-x', cycleDuration(attrs.speed, 1), 'ease-in-out');
      }
      break;
  }
}

export function applyCharacterEffectStyles(el, context, index, grapheme) {
  const name = context.name;
  const attrs = context.attrs || {};

  el.style.display = 'inline-block';
  el.style.willChange = 'opacity, filter';

  switch (name) {
    case 'typewriter': {
      const speed = clampNumber(attrs.speed, 0.1, 120, 8);
      const delay = index / speed;
      el.style.opacity = '0';
      el.style.setProperty('--bb-delay', delay + 's');
      el.style.animation = 'bb-typewriter 0.001s steps(1, end) ' + delay + 's 1 forwards';
      break;
    }
    case 'hacker': {
      const isSpace = /^\s$/.test(grapheme);
      el.classList.add('bb-hacker-char');
      el.dataset.bbHackerIndex = String(index);
      if (!isSpace) {
        el.style.filter = 'drop-shadow(0 0 0.2em currentColor)';
      }
      break;
    }
    case 'fire': {
      const intensity = clampNumber(attrs.intensity, 0, 1, 0.5);
      el.style.setProperty('--bb-fire-glow', (intensity * 0.25).toFixed(3) + 'em');
      el.style.setProperty('--bb-fire-flare', (intensity * 0.45).toFixed(3) + 'em');
      el.style.setProperty('--bb-fire-hot-glow', (intensity * 0.35).toFixed(3) + 'em');
      el.style.setProperty('--bb-fire-hot-flare', (intensity * 0.7).toFixed(3) + 'em');
      el.style.setProperty('--bb-fire-low-glow', (intensity * 0.22).toFixed(3) + 'em');
      el.style.setProperty('--bb-fire-low-flare', (intensity * 0.55).toFixed(3) + 'em');
      el.style.willChange = 'color, filter, text-shadow';
      el.style.animation = 'bb-fire ' + Math.max(0.12, 0.58 - intensity * 0.32) + 's ease-in-out ' + (-index * 0.037) + 's infinite both';
      break;
    }
    case 'electric': {
      const intensity = clampNumber(attrs.intensity, 0, 80, 5);
      el.style.setProperty('--bb-electric-x1', (intensity * 0.5).toFixed(2) + 'px');
      el.style.setProperty('--bb-electric-y1', (intensity * -0.25).toFixed(2) + 'px');
      el.style.setProperty('--bb-electric-x2', (intensity * -0.35).toFixed(2) + 'px');
      el.style.setProperty('--bb-electric-y2', (intensity * 0.2).toFixed(2) + 'px');
      el.style.setProperty('--bb-electric-x3', (intensity * 0.12).toFixed(2) + 'px');
      el.style.setProperty('--bb-electric-y3', (intensity * 0.35).toFixed(2) + 'px');
      el.style.setProperty('--bb-electric-sx1', (intensity * -0.18).toFixed(2) + 'px');
      el.style.setProperty('--bb-electric-sx2', (intensity * 0.22).toFixed(2) + 'px');
      el.style.setProperty('--bb-electric-sx3', (intensity * 0.2).toFixed(2) + 'px');
      el.style.setProperty('--bb-electric-sx4', (intensity * -0.16).toFixed(2) + 'px');
      el.style.willChange = 'transform, filter, text-shadow';
      el.style.animation = 'bb-electric ' + cycleDuration(attrs.freq, 10) + 's steps(2, end) ' + (-index * 0.019) + 's infinite both';
      break;
    }
    case 'fade': {
      const start = Math.max(0, Math.floor(clampNumber(attrs.start, 0, 10000, 0)));
      const length = Math.max(1, Math.floor(clampNumber(attrs.length, 1, 10000, 12)));
      const progress = Math.max(0, Math.min(1, (index - start) / length));
      el.style.opacity = String(1 - progress);
      break;
    }
  }
}

function setAnimation(el, name, duration, timing) {
  const delay = el.style.getPropertyValue('--bb-delay') || '0s';
  el.style.animation = name + ' ' + duration + 's ' + (timing || 'linear') + ' ' + delay + ' infinite both';
}

function cycleDuration(value, fallback) {
  const freq = clampNumber(value, 0.05, 60, fallback);
  return Math.max(0.05, 1 / freq);
}

function appendTextDecoration(current, next) {
  const parts = new Set(String(current || '').split(/\s+/).filter(Boolean));
  parts.add(next);
  return Array.from(parts).join(' ');
}

function sanitizeFontFamily(value) {
  const font = String(value || '').replace(/[^\w\s'",.-]/g, '').trim();
  return font || 'inherit';
}

function applyGoogleFont(el, attrs) {
  const family = sanitizeGoogleFontName(attrs.value || attrs.family || attrs.name);
  if (!family) return;

  el.style.fontFamily = '"' + family.replace(/"/g, '') + '", sans-serif';
  if (!['0', 'false', 'no', 'off'].includes(String(attrs.load || 'true').toLowerCase())) {
    ensureGoogleFont(el.ownerDocument || document, family, attrs);
  }
}

function sanitizeGoogleFontName(value) {
  return String(value || '').replace(/[^a-z0-9\s_-]/gi, '').replace(/\s+/g, ' ').trim().slice(0, 80);
}

function ensureGoogleFont(doc, family, attrs) {
  const key = family.toLowerCase().replace(/\s+/g, '-');
  const id = 'bb-gfont-' + cssSafeIdent(key);
  if (doc.getElementById(id)) return;

  const weight = String(attrs.weight || attrs.wght || '').replace(/[^0-9;]/g, '');
  const familyParam = family.replace(/\s+/g, '+') + (weight ? ':wght@' + weight : '');
  const link = doc.createElement('link');
  link.id = id;
  link.rel = 'stylesheet';
  link.href = 'https://fonts.googleapis.com/css2?family=' + familyParam + '&display=swap';
  doc.head.appendChild(link);
}

export function normalizeColor(value) {
  const raw = String(value || '').trim();
  if (/^#[0-9a-f]{3}$/i.test(raw) || /^#[0-9a-f]{6}$/i.test(raw)) {
    return raw;
  }

  const keyword = raw.toLowerCase().replace(/[\s_-]/g, '');
  if (isSupportedCssColor(keyword)) return keyword;
  return null;
}

function isSupportedCssColor(value) {
  if (!value) return false;
  if (typeof CSS !== 'undefined' && CSS.supports) {
    return CSS.supports('color', value);
  }

  return BASIC_COLOR_KEYWORDS.has(value);
}

const BASIC_COLOR_KEYWORDS = new Set([
  'transparent', 'currentcolor',
  'black', 'silver', 'gray', 'white', 'maroon', 'red', 'purple', 'fuchsia',
  'green', 'lime', 'olive', 'yellow', 'navy', 'blue', 'teal', 'aqua',
  'orange', 'rebeccapurple'
]);

function applyColor(el, property, value) {
  const color = normalizeColor(value);
  if (color) el.style[property] = color;
}

function applyGradient(el, attrs) {
  const stops = parseGradientStops(attrs);
  if (!stops.length) return;

  const direction = String(attrs.dir || attrs.direction || 'horizontal').toLowerCase() === 'vertical'
    ? 'to bottom'
    : 'to right';

  el.style.backgroundImage = 'linear-gradient(' + direction + ', ' + stops.join(', ') + ')';
  el.style.webkitBackgroundClip = 'text';
  el.style.backgroundClip = 'text';
  el.style.color = 'transparent';
}

function parseGradientStops(attrs) {
  if (attrs.stops) {
    return String(attrs.stops).split(',').map((stop) => {
      const trimmed = stop.trim();
      const match = trimmed.match(/^(.+?)\s+([0-9.]+%)$/);
      if (!match) {
        const colorOnly = normalizeColor(trimmed);
        return colorOnly || null;
      }
      const color = normalizeColor(match[1]);
      return color ? color + ' ' + match[2] : null;
    }).filter(Boolean);
  }

  const from = normalizeColor(attrs.from || 'white');
  const to = normalizeColor(attrs.to || 'black');
  return from && to ? [from, to] : [];
}

function applyOutline(el, attrs) {
  const color = normalizeColor(attrs.color || 'black') || '#000000';
  const size = clampNumber(attrs.size || attrs.width, 1, 20, 2);
  el.style.webkitTextStroke = size + 'px ' + color;
}

function applyShadow(el, attrs, glow) {
  const color = normalizeColor(attrs.color || (glow ? 'white' : 'gray')) || (glow ? '#FFFFFF' : '#BEBEBE');
  if (glow) {
    const size = clampNumber(attrs.size, 1, 30, 8);
    el.style.textShadow = '0 0 ' + size + 'px ' + color + ', 0 0 ' + Math.round(size * 1.6) + 'px ' + color;
    return;
  }

  const x = clampNumber(attrs.x, -50, 50, 2);
  const y = clampNumber(attrs.y, -50, 50, 2);
  el.style.textShadow = x + 'px ' + y + 'px 0 ' + color;
}

function applyAlign(el, value) {
  const align = String(value || '').toLowerCase();
  if (!['left', 'center', 'right', 'justify'].includes(align)) return;
  el.style.display = 'block';
  el.style.textAlign = align;
}

function applyBox(el, attrs) {
  const border = clampNumber(attrs.border, 0, 100, 2);
  const color = normalizeColor(attrs.color || 'white') || '#FFFFFF';
  const padding = clampNumber(attrs.padding, 0, 500, 10);
  const radius = clampNumber(attrs.radius, 0, 500, 0);
  el.style.display = 'inline-block';
  el.style.border = border + 'px solid ' + color;
  el.style.padding = padding + 'px';
  el.style.borderRadius = radius + 'px';
}

function applyWrap(el, value) {
  const mode = String(value || 'word').toLowerCase();
  el.style.display = 'block';
  el.style.whiteSpace = 'normal';

  if (mode === 'char' || mode === 'character' || mode === 'anywhere') {
    el.style.overflowWrap = 'anywhere';
    el.style.wordBreak = 'break-word';
    return;
  }

  el.style.overflowWrap = 'normal';
  el.style.wordBreak = 'normal';
}

function applySlide(el, attrs) {
  const dir = String(attrs.dir || attrs.direction || attrs.value || 'left').toLowerCase();
  const distance = clampNumber(attrs.distance, 1, 4000, 120);
  const speed = clampNumber(attrs.speed, 1, 2000, 80);
  const duration = Math.max(0.05, distance / speed);

  let x = '0';
  let y = '0';
  if (dir === 'right') x = distance + 'px';
  else if (dir === 'up') y = '-' + distance + 'px';
  else if (dir === 'down') y = distance + 'px';
  else x = '-' + distance + 'px';

  el.style.display = 'inline-block';
  el.style.willChange = 'transform, opacity';
  el.style.setProperty('--bb-slide-x', x);
  el.style.setProperty('--bb-slide-y', y);
  el.style.animation = 'bb-slide-in ' + duration + 's cubic-bezier(.16,1,.3,1) 0s 1 both';
}

function applyZoom(el, attrs) {
  const from = clampNumber(attrs.from, 0, 10, 0);
  const to = clampNumber(attrs.to, 0, 10, 1);
  const speed = clampNumber(attrs.speed, 0.05, 60, 2);

  el.style.display = 'inline-block';
  el.style.willChange = 'transform, opacity';
  el.style.transformOrigin = '50% 55%';
  el.style.setProperty('--bb-zoom-from', String(from));
  el.style.setProperty('--bb-zoom-to', String(to));
  el.style.animation = 'bb-zoom ' + Math.max(0.05, 1 / speed) + 's cubic-bezier(.16,1,.3,1) 0s 1 both';
}

export function createImageElement(doc, attrs) {
  const src = normalizeImageSrc(attrs.src || attrs.value || attrs.url || attrs.path);
  if (!src) return null;

  const img = doc.createElement('img');
  img.className = 'bb-img';
  img.dataset.bbTag = 'img';
  img.src = src;
  img.alt = String(attrs.alt || '');
  img.decoding = 'async';
  img.loading = 'eager';

  const width = readSizeOption(attrs.width || attrs.w);
  const height = readSizeOption(attrs.height || attrs.h);
  if (width) {
    img.style.width = width;
    img.style.maxWidth = width;
  }
  if (height) {
    img.style.height = height;
    img.style.maxHeight = height;
  }

  if (attrs.fit) {
    const fit = String(attrs.fit).toLowerCase();
    if (['contain', 'cover', 'fill', 'none', 'scale-down'].includes(fit)) {
      img.style.objectFit = fit;
    }
  }

  return img;
}

export function createRandomElement(doc, attrs) {
  const el = doc.createElement('span');
  el.className = 'bb-random';
  el.dataset.bbTag = 'random';
  Object.keys(attrs || {}).forEach((key) => {
    el.dataset['bbAttr' + key.replace(/(^|-)([a-z0-9])/g, (_, _dash, ch) => ch.toUpperCase())] = String(attrs[key]);
  });
  const words = parseRandomWords(attrs.words || attrs.value);
  el.textContent = chooseRandomWord(words);
  return el;
}

function normalizeImageSrc(value) {
  const raw = String(value || '').trim();
  if (!raw) return '';
  if (/^(https?:|file:|data:image\/)/i.test(raw)) return raw;
  if (/^[a-z]:[\\/]/i.test(raw)) return 'file:///' + raw.replace(/\\/g, '/');
  if (/^[./]/.test(raw)) return raw;
  return '';
}

export function parseRandomWords(value) {
  const source = String(value || '');
  const words = [];
  let current = '';
  let quote = '';

  for (let index = 0; index < source.length; index++) {
    const ch = source[index];
    if (ch === '\\' && index + 1 < source.length) {
      current += source[++index];
      continue;
    }
    if (quote) {
      if (ch === quote) quote = '';
      else current += ch;
      continue;
    }
    if (ch === '"' || ch === "'") {
      quote = ch;
      continue;
    }
    if (ch === ',') {
      pushRandomWord(words, current);
      current = '';
      continue;
    }
    current += ch;
  }

  pushRandomWord(words, current);
  return words.length ? words : [''];
}

export function chooseRandomWord(words) {
  const list = Array.isArray(words) && words.length ? words : [''];
  return list[Math.floor(Math.random() * list.length)] || '';
}

function pushRandomWord(words, value) {
  const word = String(value || '').trim();
  if (word) words.push(word);
}
