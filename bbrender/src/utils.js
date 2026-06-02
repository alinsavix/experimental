export function normalizeTagName(name) {
  return String(name || '').trim().toLowerCase();
}

export function clampNumber(value, min, max, fallback) {
  const parsed = parseFloat(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(min, Math.min(max, parsed));
}

export function cssSafeIdent(value) {
  return String(value || '').replace(/[^a-z0-9_-]/gi, '-');
}

export function toDatasetName(value) {
  return String(value || '').replace(/(^|-)([a-z0-9])/g, (_, _dash, ch) => ch.toUpperCase());
}

export function parseBoolean(value) {
  return value === true || value === 1 || ['1', 'true', 'yes', 'on'].includes(String(value || '').toLowerCase());
}

export function readSizeOption(value) {
  if (value === undefined || value === null || value === '') return '';
  const raw = String(value).trim();
  if (/^[0-9.]+(px|vw|vh|vmin|vmax|em|rem|%)$/i.test(raw)) return raw;
  const number = clampNumber(raw, 0, 10000, NaN);
  return Number.isFinite(number) ? number + 'px' : '';
}

export function readLineHeightOption(value) {
  if (value === undefined || value === null || value === '') return '';
  const raw = String(value).trim();
  if (/^[0-9.]+(px|em|rem|%)$/i.test(raw)) return raw;
  const number = clampNumber(raw, 0.2, 5, NaN);
  return Number.isFinite(number) ? String(number) : '';
}

export function splitGraphemes(text) {
  const value = String(text || '');
  if (typeof Intl !== 'undefined' && Intl.Segmenter) {
    const segmenter = new Intl.Segmenter(undefined, { granularity: 'grapheme' });
    return Array.from(segmenter.segment(value), (part) => part.segment);
  }
  return Array.from(value);
}

export function readDatasetAttrs(el) {
  const attrs = {};
  Object.keys(el.dataset || {}).forEach((key) => {
    if (!key.startsWith('bbAttr')) return;
    const attrName = key.slice(6).replace(/[A-Z]/g, (ch) => '-' + ch.toLowerCase()).replace(/^-/, '');
    attrs[attrName] = el.dataset[key];
  });
  return attrs;
}

export function randomGlyph(random, glyphs) {
  const chars = Array.from(glyphs || '?');
  return chars[Math.floor(random() * chars.length)] || '?';
}

export function createSeededRandom(seedText) {
  let seed = 2166136261;
  const source = String(seedText || '');
  for (let i = 0; i < source.length; i++) {
    seed ^= source.charCodeAt(i);
    seed = Math.imul(seed, 16777619);
  }
  return function nextRandom() {
    seed += 0x6D2B79F5;
    let value = seed;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}
