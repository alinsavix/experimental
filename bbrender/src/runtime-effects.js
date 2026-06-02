import { clampNumber, createSeededRandom, parseBoolean, randomGlyph, readDatasetAttrs, splitGraphemes } from './utils.js';

export function clearRuntimeEffects(element) {
  const timers = element.__bbCodeRuntimeTimers || [];
  timers.forEach((timer) => clearTimeout(timer));
  element.__bbCodeRuntimeTimers = [];
}

export function applyDropCaps(element) {
  const tags = Array.from(element.querySelectorAll('[data-bb-tag="dropcap"]'));
  tags.forEach((tag) => {
    if (tag.querySelector('.bb-dropcap-first')) return;

    const attrs = readDatasetAttrs(tag);
    const lines = Math.round(clampNumber(attrs.lines || attrs.value, 2, 8, 3));
    tag.style.setProperty('--bb-dropcap-lines', String(lines));

    const textNode = findFirstNonWhitespaceTextNode(tag);
    if (!textNode) return;

    const graphemes = splitGraphemes(textNode.nodeValue);
    const firstIndex = graphemes.findIndex((part) => /\S/.test(part));
    if (firstIndex === -1) return;

    const before = graphemes.slice(0, firstIndex).join('');
    const first = graphemes[firstIndex];
    const after = graphemes.slice(firstIndex + 1).join('');
    const span = element.ownerDocument.createElement('span');
    span.className = 'bb-dropcap-first';
    span.textContent = first;

    const fragment = element.ownerDocument.createDocumentFragment();
    if (before) fragment.appendChild(element.ownerDocument.createTextNode(before));
    fragment.appendChild(span);
    if (after) fragment.appendChild(element.ownerDocument.createTextNode(after));
    textNode.parentNode.replaceChild(fragment, textNode);
  });
}

function findFirstNonWhitespaceTextNode(root) {
  const doc = root.ownerDocument || document;
  const view = doc.defaultView || globalThis;
  const filter = {
    acceptNode(node) {
      if (!node.nodeValue || !/\S/.test(node.nodeValue)) return view.NodeFilter.FILTER_REJECT;
      if (node.parentElement && node.parentElement.closest('.bb-dropcap-first')) {
        return view.NodeFilter.FILTER_REJECT;
      }
      return view.NodeFilter.FILTER_ACCEPT;
    }
  };
  const walker = doc.createTreeWalker(root, view.NodeFilter.SHOW_TEXT, filter);
  return walker.nextNode();
}

export function startRuntimeEffects(element) {
  const timers = [];
  const doc = element.ownerDocument || document;
  const view = doc.defaultView || globalThis;
  const typewriterTags = Array.from(element.querySelectorAll('[data-bb-tag="typewriter"]'));
  const hackerTags = Array.from(element.querySelectorAll('[data-bb-tag="hacker"]'));

  typewriterTags.forEach((tag) => {
    const attrs = readDatasetAttrs(tag);
    if (!parseBoolean(attrs.cursor)) return;

    const speed = clampNumber(attrs.speed, 0.1, 120, 8);
    const stepMs = 1000 / speed;
    const chars = Array.from(tag.querySelectorAll('[data-bb-effect="typewriter"]'));
    if (!chars.length) return;

    function setCursor(index) {
      chars.forEach((charEl) => charEl.classList.remove('bb-typewriter-cursor'));
      chars[Math.min(index, chars.length - 1)].classList.add('bb-typewriter-cursor');
    }

    setCursor(0);
    for (let index = 1; index < chars.length; index++) {
      timers.push(view.setTimeout(() => setCursor(index), index * stepMs));
    }
  });

  hackerTags.forEach((tag, tagIndex) => {
    const attrs = readDatasetAttrs(tag);
    const speed = clampNumber(attrs.speed, 0.1, 120, 4);
    const loops = Math.max(0, Math.floor(clampNumber(attrs.loop, 0, 20, 2)));
    const stepMs = 1000 / speed;
    const seed = String(attrs.seed || 'hacker') + '|' + tagIndex + '|' + tag.textContent;
    const random = createSeededRandom(seed);
    const glyphs = String(attrs.glyphs || 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#$%&?@');
    const chars = Array.from(tag.querySelectorAll('[data-bb-effect="hacker"] .bb-char-text'));

    chars.forEach((charEl, charIndex) => {
      const finalValue = charEl.dataset.bbFinal || charEl.textContent || '';
      if (/^\s*$/.test(finalValue)) return;

      charEl.textContent = randomGlyph(random, glyphs);
      for (let cycle = 0; cycle < loops; cycle++) {
        timers.push(view.setTimeout(() => {
          charEl.textContent = randomGlyph(random, glyphs);
        }, (charIndex + cycle + 1) * stepMs));
      }
      timers.push(view.setTimeout(() => {
        charEl.textContent = finalValue;
        const wrapper = charEl.closest('[data-bb-effect="hacker"]');
        if (wrapper) wrapper.style.filter = '';
      }, (charIndex + loops + 1) * stepMs));
    });
  });

  element.__bbCodeRuntimeTimers = timers;
}
