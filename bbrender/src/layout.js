import { getAutoPaddingForTag } from './tag-registry.js';
import { parseBoolean, readDatasetAttrs, readLineHeightOption, readSizeOption } from './utils.js';

export function applySourceLayout(element, options) {
  const opts = options || {};
  const width = readSizeOption(opts.width || opts.sourceWidth);
  const height = readSizeOption(opts.height || opts.sourceHeight);
  const padding = readSizeOption(opts.padding);
  const fontSize = readSizeOption(opts.fontSize || opts.baseFontSize);
  const lineHeight = readLineHeightOption(opts.lineHeight || opts.lineSpacing);
  const autoPadding = parseBoolean(opts.autoPadding);

  if (width) {
    element.style.width = width;
    element.style.maxWidth = 'none';
  } else {
    element.style.width = '';
    element.style.maxWidth = '';
  }
  if (height) {
    element.style.height = height;
    element.style.overflow = 'hidden';
  } else {
    element.style.height = '';
    element.style.overflow = '';
  }
  if (padding) {
    element.style.padding = padding;
  } else if (autoPadding) {
    element.style.padding = computeAutoPadding(element) + 'px';
  } else {
    element.style.padding = '';
  }
  if (fontSize) {
    element.style.fontSize = fontSize;
  } else {
    element.style.fontSize = '';
  }
  if (lineHeight) {
    element.style.lineHeight = lineHeight;
  } else {
    element.style.lineHeight = '';
  }
}

function computeAutoPadding(element) {
  let padding = 0;
  Array.from(element.querySelectorAll('[data-bb-tag], [data-bb-anim]')).forEach((el) => {
    const attrs = readDatasetAttrs(el);
    const tagName = el.dataset.bbTag || '';
    const animName = el.dataset.bbAnim || '';
    const motionName = animName || tagName;

    padding = Math.max(
      padding,
      getAutoPaddingForTag(tagName, attrs),
      getAutoPaddingForTag(motionName, attrs)
    );
  });
  return Math.ceil(Math.min(300, padding + 8));
}
