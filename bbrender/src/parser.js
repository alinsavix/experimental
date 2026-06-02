import { KNOWN_TAGS, NOPARSE_TAGS, RESET_TAGS, SELF_TAGS } from './tag-definitions.js';
import { normalizeTagName } from './utils.js';

function createTextNode(value) {
  return { type: 'text', value: String(value || '') };
}

function createTagNode(name, attrs, rawOpen) {
  return {
    type: 'tag',
    name,
    attrs: attrs || {},
    rawOpen: rawOpen || '',
    children: []
  };
}

export function parseTagToken(raw) {
  const source = String(raw || '');
  const trimmed = source.trim();
  if (!trimmed) return null;

  if (trimmed[0] === '/') {
    const closeName = normalizeTagName(trimmed.slice(1).split(/\s+/)[0]);
    return {
      type: 'close',
      name: closeName,
      raw: '[' + source + ']'
    };
  }

  const eqIndex = trimmed.indexOf('=');
  const spaceMatch = /\s/.exec(trimmed);
  const spaceIndex = spaceMatch ? spaceMatch.index : -1;
  let nameEnd = trimmed.length;
  let defaultValue = null;
  let attrText = '';

  if (eqIndex !== -1 && (spaceIndex === -1 || eqIndex < spaceIndex)) {
    nameEnd = eqIndex;
    const afterEquals = trimmed.slice(eqIndex + 1);
    const valueResult = readValue(afterEquals, 0);
    defaultValue = valueResult.value;
    attrText = afterEquals.slice(valueResult.end);
  } else if (spaceIndex !== -1) {
    nameEnd = spaceIndex;
    attrText = trimmed.slice(spaceIndex + 1);
  }

  const name = normalizeTagName(trimmed.slice(0, nameEnd));
  if (!name) return null;

  const attrs = parseAttributes(attrText);
  if (defaultValue !== null) {
    attrs.value = defaultValue;
  }

  return {
    type: 'open',
    name,
    attrs,
    raw: '[' + source + ']'
  };
}

function readValue(source, startIndex) {
  let index = startIndex || 0;
  while (index < source.length && /\s/.test(source[index])) index++;

  if (source[index] === '"' || source[index] === "'") {
    const quote = source[index++];
    let value = '';
    while (index < source.length) {
      const ch = source[index++];
      if (ch === quote) break;
      if (ch === '\\' && index < source.length) {
        value += source[index++];
      } else {
        value += ch;
      }
    }
    return { value, end: index };
  }

  const start = index;
  while (index < source.length && !/\s/.test(source[index])) index++;
  return { value: source.slice(start, index), end: index };
}

export function parseAttributes(source) {
  const attrs = {};
  let index = 0;
  const text = String(source || '');

  while (index < text.length) {
    while (index < text.length && /\s/.test(text[index])) index++;
    if (index >= text.length) break;

    const keyStart = index;
    while (index < text.length && !/[\s=]/.test(text[index])) index++;
    const key = normalizeTagName(text.slice(keyStart, index));
    if (!key) break;

    while (index < text.length && /\s/.test(text[index])) index++;
    if (text[index] !== '=') {
      attrs[key] = true;
      continue;
    }

    index++;
    const valueResult = readValue(text, index);
    attrs[key] = valueResult.value;
    index = valueResult.end;
  }

  return attrs;
}

function findNoParseClose(source, tagName, fromIndex) {
  const lower = source.toLowerCase();
  const needle = '[/' + tagName + ']';
  return lower.indexOf(needle, fromIndex);
}

export function parse(input, options) {
  const source = String(input || '');
  const opts = options || {};
  const diagnostics = [];
  const root = { type: 'root', children: [] };
  const stack = [root];
  let index = 0;

  function currentChildren() {
    return stack[stack.length - 1].children;
  }

  function addText(value) {
    if (value) currentChildren().push(createTextNode(value));
  }

  while (index < source.length) {
    const openIndex = source.indexOf('[', index);
    if (openIndex === -1) {
      addText(source.slice(index));
      break;
    }

    addText(source.slice(index, openIndex));
    const closeIndex = source.indexOf(']', openIndex + 1);
    if (closeIndex === -1) {
      addText(source.slice(openIndex));
      break;
    }

    const rawContent = source.slice(openIndex + 1, closeIndex);
    const token = parseTagToken(rawContent);
    if (!token) {
      addText(source.slice(openIndex, closeIndex + 1));
      index = closeIndex + 1;
      continue;
    }

    if (token.type === 'close') {
      if (RESET_TAGS.has(token.name)) {
        resetStack(stack, diagnostics, token);
      } else {
        const matchIndex = findOpenTagIndex(stack, token.name);
        if (matchIndex === -1) {
          diagnostics.push({
            type: 'unmatched-close',
            message: 'Unmatched closing tag ' + token.raw + '.'
          });
          addText(token.raw);
        } else {
          while (stack.length - 1 >= matchIndex) stack.pop();
        }
      }
      index = closeIndex + 1;
      continue;
    }

    if (!KNOWN_TAGS.has(token.name)) {
      diagnostics.push({
        type: 'unknown-tag',
        message: 'Unknown tag ' + token.raw + ' rendered literally.'
      });
      addText(token.raw);
      index = closeIndex + 1;
      continue;
    }

    if (NOPARSE_TAGS.has(token.name)) {
      const endIndex = findNoParseClose(source, token.name, closeIndex + 1);
      if (endIndex === -1) {
        diagnostics.push({
          type: 'missing-close',
          message: 'Missing closing tag for ' + token.raw + '.'
        });
        addText(source.slice(openIndex));
        index = source.length;
      } else {
        addText(source.slice(closeIndex + 1, endIndex));
        index = endIndex + token.name.length + 3;
      }
      continue;
    }

    if (RESET_TAGS.has(token.name)) {
      resetStack(stack, diagnostics, token);
      index = closeIndex + 1;
      continue;
    }

    if (SELF_TAGS.has(token.name)) {
      currentChildren().push({
        type: token.name,
        attrs: token.attrs,
        raw: token.raw
      });
      index = closeIndex + 1;
      continue;
    }

    const tagNode = createTagNode(token.name, token.attrs, token.raw);
    currentChildren().push(tagNode);
    stack.push(tagNode);
    index = closeIndex + 1;
  }

  if (stack.length > 1) {
    for (let i = stack.length - 1; i > 0; i--) {
      diagnostics.push({
        type: 'missing-close',
        message: 'Missing closing tag for ' + stack[i].rawOpen + '.'
      });
    }
  }

  if (opts.diagnostics) {
    opts.diagnostics.push.apply(opts.diagnostics, diagnostics);
  }

  return { ast: root, diagnostics };
}

function resetStack(stack, diagnostics, token) {
  if (stack.length > 1) {
    diagnostics.push({
      type: 'reset',
      message: token.raw + ' reset ' + (stack.length - 1) + ' open tag(s).'
    });
  }
  stack.length = 1;
}

function findOpenTagIndex(stack, name) {
  for (let i = stack.length - 1; i > 0; i--) {
    if (stack[i].name === name) return i;
  }
  return -1;
}
