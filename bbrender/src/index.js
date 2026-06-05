import { parse, parseAttributes, parseTagToken } from './parser.js';
import { renderToFragment } from './dom-renderer.js';
import { SELF_TAGS, RESET_TAGS, KNOWN_TAGS } from './tag-definitions.js';
import { applySourceLayout } from './layout.js';
import { clearRuntimeEffects, applyDropCaps, startRuntimeEffects } from './runtime-effects.js';
import { extractBBCode, getCommand, shouldHandlePayload } from './payload.js';
import { normalizeColor, applyAnimationStyles } from './tag-styles.js';
import { applyRootTransition, clearRootTransition, normalizeTransition } from './transitions.js';
import { splitGraphemes } from './utils.js';

export { parse, parseAttributes, parseTagToken } from './parser.js';
export { renderToFragment } from './dom-renderer.js';
export { applySourceLayout } from './layout.js';
export { extractBBCode, getCommand, shouldHandlePayload } from './payload.js';
export { normalizeColor, applyAnimationStyles } from './tag-styles.js';
export { applyRootTransition, clearRootTransition, normalizeTransition } from './transitions.js';
export { splitGraphemes } from './utils.js';

export const knownTags = KNOWN_TAGS;

export function renderToElement(element, bbcode, options) {
  const opts = options || {};
  const diagnostics = [];
  const wrapped = wrapWithGlobalTags(String(bbcode || ''), opts.globalTags || '');
  const parsed = parse(wrapped, { diagnostics });

  clearRuntimeEffects(element);
  clearRootTransition(element);
  element.replaceChildren(renderToFragment(parsed.ast, element.ownerDocument));
  applyDropCaps(element);
  if (opts.layout) {
    applySourceLayout(element, opts.layout);
  }
  applyRootTransition(element, opts.transition, opts.duration);
  startRuntimeEffects(element);

  if (diagnostics.length > 0 && opts.logDiagnostics !== false && typeof console !== 'undefined') {
    console.warn('BBCode diagnostics:', diagnostics.map((item) => item.message));
  }

  if (typeof opts.onDiagnostics === 'function') {
    opts.onDiagnostics(diagnostics);
  }

  return {
    ast: parsed.ast,
    diagnostics,
    source: wrapped
  };
}

export function wrapWithGlobalTags(text, globalTags) {
  const tags = String(globalTags || '').trim();
  if (!tags) return text;

  const opens = [];
  let index = 0;
  while (index < tags.length) {
    const openIndex = tags.indexOf('[', index);
    if (openIndex === -1) break;
    const closeIndex = tags.indexOf(']', openIndex + 1);
    if (closeIndex === -1) break;

    const token = parseTagToken(tags.slice(openIndex + 1, closeIndex));
    if (token && token.type === 'open' && !SELF_TAGS.has(token.name) && !RESET_TAGS.has(token.name)) {
      opens.push(token.name);
    }
    index = closeIndex + 1;
  }

  const closingTags = opens.reverse().map((name) => '[/' + name + ']').join('');
  return tags + text + closingTags;
}

export function createState() {
  return {
    globalTags: '',
    duration: 4000,
    queueMode: 'replace',
    layout: {},
    presets: {}
  };
}
