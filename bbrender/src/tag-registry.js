import { TAG_STYLE_HANDLERS } from './tag-styles.js';
import { clampNumber } from './utils.js';

function styleTag(name) {
  return {
    kind: 'style',
    apply: TAG_STYLE_HANDLERS[name]
  };
}

function animationTag(name, autoPadding) {
  return {
    kind: 'animation',
    apply: TAG_STYLE_HANDLERS[name],
    autoPadding
  };
}

function effectTag(name) {
  return {
    kind: 'character-effect',
    apply: TAG_STYLE_HANDLERS[name]
  };
}

export const TAGS = {
  align: styleTag('align'),
  b: styleTag('b'),
  big: styleTag('big'),
  blink: animationTag('blink'),
  blur: {
    kind: 'style',
    apply: TAG_STYLE_HANDLERS.blur,
    autoPadding: (attrs) => clampNumber(attrs.value, 1, 20, 3) * 2
  },
  bounce: animationTag('bounce', (attrs) => clampNumber(attrs.amp, 0, 300, 20)),
  box: styleTag('box'),
  bg: styleTag('bg'),
  center: styleTag('center'),
  code: styleTag('code'),
  color: styleTag('color'),
  colour: styleTag('colour'),
  dropcap: styleTag('dropcap'),
  electric: effectTag('electric'),
  fade: effectTag('fade'),
  fire: effectTag('fire'),
  font: styleTag('font'),
  gfont: styleTag('gfont'),
  glow: {
    kind: 'style',
    apply: TAG_STYLE_HANDLERS.glow,
    autoPadding: (attrs) => clampNumber(attrs.size, 1, 30, 8) * 1.8
  },
  gradient: styleTag('gradient'),
  hacker: effectTag('hacker'),
  highlight: styleTag('highlight'),
  hr: { kind: 'self' },
  i: styleTag('i'),
  img: { kind: 'self' },
  indent: styleTag('indent'),
  justify: styleTag('justify'),
  left: styleTag('left'),
  newline: { kind: 'self' },
  noparse: { kind: 'literal' },
  opacity: styleTag('opacity'),
  outline: {
    kind: 'style',
    apply: TAG_STYLE_HANDLERS.outline,
    autoPadding: textStrokePadding
  },
  plain: { kind: 'literal' },
  pulse: animationTag('pulse'),
  rainbow: animationTag('rainbow'),
  random: { kind: 'self' },
  reset: { kind: 'reset' },
  right: styleTag('right'),
  rotate: animationTag('rotate'),
  s: styleTag('s'),
  shadow: {
    kind: 'style',
    apply: TAG_STYLE_HANDLERS.shadow,
    autoPadding(attrs) {
      return Math.max(
        Math.abs(clampNumber(attrs.x, -50, 50, 2)),
        Math.abs(clampNumber(attrs.y, -50, 50, 2))
      );
    }
  },
  shake: animationTag('shake', (attrs) => clampNumber(attrs.level, 0, 80, 5)),
  size: styleTag('size'),
  slide: styleTag('slide'),
  small: styleTag('small'),
  spacing: styleTag('spacing'),
  stroke: {
    kind: 'style',
    apply: TAG_STYLE_HANDLERS.stroke,
    autoPadding: textStrokePadding
  },
  sub: styleTag('sub'),
  sup: styleTag('sup'),
  tornado: animationTag('tornado', (attrs) => clampNumber(attrs.radius, 0, 200, 10)),
  tt: styleTag('tt'),
  typewriter: effectTag('typewriter'),
  u: styleTag('u'),
  wave: animationTag('wave', (attrs) => clampNumber(attrs.amp, 0, 300, 50)),
  wrap: styleTag('wrap'),
  flip: animationTag('flip'),
  metallic: animationTag('metallic'),
  zoom: styleTag('zoom')
};

export const KNOWN_TAGS = new Set(Object.keys(TAGS));
export const RESET_TAGS = new Set(['reset', 'all']);
export const NOPARSE_TAGS = tagsByKind('literal');
export const SELF_TAGS = tagsByKind('self');
export const ANIMATION_TAGS = tagsByKind('animation');
export const CHARACTER_EFFECT_TAGS = tagsByKind('character-effect');

export function applyTagStyles(el, name, attrs = {}) {
  const apply = TAGS[name] && TAGS[name].apply;
  if (apply) apply(el, attrs);
}

export function getTagKind(name) {
  return TAGS[name] && TAGS[name].kind;
}

export function isCharacterWrappedTag(name) {
  return ANIMATION_TAGS.has(name) || CHARACTER_EFFECT_TAGS.has(name);
}

export function getAutoPaddingForTag(name, attrs = {}) {
  const getPadding = TAGS[name] && TAGS[name].autoPadding;
  return getPadding ? getPadding(attrs) : 0;
}

function tagsByKind(kind) {
  return new Set(Object.keys(TAGS).filter((name) => TAGS[name].kind === kind));
}

function textStrokePadding(attrs) {
  return clampNumber(attrs.size || attrs.width, 1, 20, 2);
}
