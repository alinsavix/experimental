import assert from 'node:assert/strict';
import * as BBRender from '../src/index.js';

export function runTests() {
  const tests = [
    ['parser creates a b tag', () => {
      const parsed = BBRender.parse('[b]hello[/b]');
      assert.equal(parsed.ast.children[0].name, 'b');
    }],
    ['quoted attributes parse', () => {
      const token = BBRender.parseTagToken('gradient stops="red 0%,gold 50%" speed=2');
      assert.equal(token.attrs.stops, 'red 0%,gold 50%');
      assert.equal(token.attrs.speed, '2');
    }],
    ['noparse keeps literal tags', () => {
      const parsed = BBRender.parse('[noparse][b]x[/b][/noparse]');
      assert.equal(parsed.ast.children[0].value, '[b]x[/b]');
    }],
    ['reset returns to root', () => {
      const parsed = BBRender.parse('[b]x[reset]y');
      assert.equal(parsed.ast.children.length, 2);
    }],
    ['global tags wrap and close in reverse order', () => {
      assert.equal(BBRender.wrapWithGlobalTags('x', '[b][color=red]'), '[b][color=red]x[/color][/b]');
    }],
    ['payload type matching is case-insensitive', () => {
      assert.equal(BBRender.shouldHandlePayload({ type: 'BBCode.Render', bbcode: 'x' }), true);
      assert.equal(BBRender.shouldHandlePayload({ type: 'other-client', bbcode: 'x' }), false);
      assert.equal(BBRender.shouldHandlePayload({ bbcode: 'x' }), false);
    }],
    ['payload extracts supported text fields', () => {
      assert.equal(BBRender.extractBBCode({ type: 'bbcode.render', message: '[b]hello[/b]' }), '[b]hello[/b]');
    }],
    ['basic named colors normalize', () => {
      assert.equal(BBRender.normalizeColor('red'), 'red');
    }],
    ['root transition aliases normalize', () => {
      const transition = BBRender.normalizeTransition({ enter: 'zoom', exit: 'fade', enterTime: 100 }, 1000);
      assert.equal(transition.in, 'zoom');
      assert.equal(transition.out, 'fade');
      assert.equal(transition.inTime, 100);
    }],
    ['state defaults are stable', () => {
      assert.deepEqual(BBRender.createState().layout, {});
      assert.equal(BBRender.createState().queueMode, 'replace');
    }]
  ];

  const failures = [];
  for (const [name, fn] of tests) {
    try {
      fn();
    } catch (error) {
      failures.push({ name, error });
    }
  }

  if (failures.length > 0) {
    failures.forEach((failure) => {
      console.error(`FAIL ${failure.name}`);
      console.error(failure.error);
    });
    throw new Error(`${failures.length} test(s) failed`);
  }

  console.log(`${tests.length} tests passed`);
}

const isMain = process.argv[1] && normalizedScriptPath(process.argv[1]).endsWith('/scripts/test.mjs');

if (isMain) {
  runTests();
}

function normalizedScriptPath(value) {
  return String(value || '').replace(/\\/g, '/').toLowerCase();
}
