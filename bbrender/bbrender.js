import * as BBRender from './src/index.js';

if (typeof window !== 'undefined') {
  window.BBRender = BBRender;
}

export * from './src/index.js';
