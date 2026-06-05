import { ensureBaseStyles } from './base-styles.js';
import { ANIMATION_TAGS, CHARACTER_EFFECT_TAGS, applyTagStyles, isCharacterWrappedTag } from './tag-registry.js';
import { applyAnimationStyles, applyCharacterEffectStyles, createImageElement, createRandomElement } from './tag-styles.js';
import { cssSafeIdent, splitGraphemes, toDatasetName } from './utils.js';

export function renderToFragment(ast, documentRef = document) {
  ensureBaseStyles(documentRef);
  const fragment = documentRef.createDocumentFragment();
  appendNodes(fragment, ast.children || [], documentRef, [], { index: 0 });
  return fragment;
}

function appendNodes(parent, nodes, doc, animationContexts, renderState) {
  const contexts = animationContexts || [];
  const state = renderState || { index: 0 };

  nodes.forEach((node) => {
    if (node.type === 'text') {
      appendTextNode(parent, doc, node.value, contexts, state);
      return;
    }

    if (node.type === 'newline') {
      parent.appendChild(doc.createElement('br'));
      return;
    }

    if (node.type === 'hr') {
      const el = doc.createElement('hr');
      el.className = 'bb-hr';
      parent.appendChild(el);
      return;
    }

    if (node.type === 'img') {
      const img = createImageElement(doc, node.attrs || {});
      if (img) parent.appendChild(img);
      return;
    }

    if (node.type === 'random') {
      parent.appendChild(createRandomElement(doc, node.attrs || {}));
      return;
    }

    if (node.type === 'tag') {
      appendTagNode(parent, doc, node, contexts, state);
    }
  });
}

function appendTextNode(parent, doc, value, contexts, state) {
  if (contexts.length > 0) {
    splitGraphemes(value).forEach((grapheme) => {
      parent.appendChild(createAnimatedGrapheme(doc, grapheme, contexts, state.index++));
    });
    return;
  }

  parent.appendChild(doc.createTextNode(value));
}

function appendTagNode(parent, doc, node, contexts, state) {
  const el = doc.createElement('span');
  el.className = 'bb-tag bb-tag-' + cssSafeIdent(node.name);
  el.dataset.bbTag = node.name;
  Object.keys(node.attrs || {}).forEach((key) => {
    el.dataset['bbAttr' + toDatasetName(key)] = String(node.attrs[key]);
  });
  applyTagStyles(el, node.name, node.attrs || {});

  const nextContexts = isCharacterWrappedTag(node.name)
    ? contexts.concat([{ name: node.name, attrs: node.attrs || {}, index: 0 }])
    : contexts;

  appendNodes(el, node.children || [], doc, nextContexts, state);
  parent.appendChild(el);
}

function createAnimatedGrapheme(doc, grapheme, contexts) {
  let outer = null;
  let current = null;

  contexts.forEach((context) => {
    const localIndex = context.index++;
    const wrapper = doc.createElement('span');
    wrapper.className = 'bb-char bb-' + contextKind(context.name) + ' bb-' + contextKind(context.name) + '-' + cssSafeIdent(context.name);
    if (ANIMATION_TAGS.has(context.name)) wrapper.dataset.bbAnim = context.name;
    if (CHARACTER_EFFECT_TAGS.has(context.name)) wrapper.dataset.bbEffect = context.name;
    wrapper.dataset.bbCharIndex = String(localIndex);
    wrapper.style.setProperty('--bb-i', String(localIndex));
    applyCharacterContextStyles(wrapper, context, localIndex, grapheme);

    if (!outer) outer = wrapper;
    if (current) current.appendChild(wrapper);
    current = wrapper;
  });

  const textNode = doc.createElement('span');
  textNode.className = 'bb-char-text';

  if (/^\s$/.test(grapheme)) {
    current.classList.add('bb-space');
    textNode.appendChild(doc.createTextNode(grapheme === '\t' ? '\u00A0\u00A0' : '\u00A0'));
  } else {
    textNode.appendChild(doc.createTextNode(grapheme));
  }
  textNode.dataset.bbFinal = grapheme;
  current.appendChild(textNode);
  return outer;
}

function contextKind(name) {
  return ANIMATION_TAGS.has(name) ? 'anim' : 'effect';
}

function applyCharacterContextStyles(el, context, index, grapheme) {
  if (ANIMATION_TAGS.has(context.name)) {
    applyAnimationStyles(el, context, index);
    return;
  }
  applyCharacterEffectStyles(el, context, index, grapheme);
}
