import test from 'node:test'
import assert from 'node:assert/strict'
import { buildRequestData } from '../src/domain/requestData.mjs'

test('omits null fields without changing other falsey values', () => {
  assert.deepEqual(
    buildRequestData([
      { name: 'unsetString', type: 'String', value: null },
      { name: 'unsetNumber', type: 'Number', value: null },
      { name: 'unsetBoolean', type: 'Boolean', value: null },
      { name: 'emptyString', type: 'String', value: '' },
      { name: 'zero', type: 'Number', value: 0 },
      { name: 'disabled', type: 'Boolean', value: false },
    ]),
    { emptyString: '', zero: 0, disabled: false },
  )
})

test('omits cleared numeric fields without omitting zero or blank strings', () => {
  assert.deepEqual(
    buildRequestData([
      { name: 'clearedNumber', type: 'Number', value: '' },
      { name: 'whitespaceNumber', type: 'Number', value: '   ' },
      { name: 'zero', type: 'Number', value: 0 },
      { name: 'zeroText', type: 'Number', value: '0' },
      { name: 'blankString', type: 'String', value: '' },
    ]),
    { zero: 0, zeroText: 0, blankString: '' },
  )
})

test('omits unset nested fields and their empty parent object', () => {
  const nestedParam = {
    name: 'keyModifiers',
    type: 'Object',
    value: null,
    params: [
      { name: 'shift', type: 'Boolean', value: null },
      { name: 'control', type: 'Boolean', value: null },
    ],
  }

  assert.deepEqual(buildRequestData([nestedParam]), {})

  nestedParam.params[0].value = false
  assert.deepEqual(buildRequestData([nestedParam]), {
    keyModifiers: { shift: false },
  })
})

test('preserves null explicitly authored inside a JSON value', () => {
  assert.deepEqual(
    buildRequestData([
      {
        name: 'settings',
        type: 'Object',
        value: '{"unset":null,"count":0,"nested":{"unset":null,"enabled":false},"items":[null,{"unset":null,"name":"kept"}]}',
      },
    ]),
    {
      settings: {
        unset: null,
        count: 0,
        nested: { unset: null, enabled: false },
        items: [null, { unset: null, name: 'kept' }],
      },
    },
  )
})

test('preserves an explicit JSON null value', () => {
  assert.deepEqual(
    buildRequestData([{ name: 'value', type: 'Object', value: 'null' }]),
    { value: null },
  )
})

test('supports documented null operations without sending untouched null fields', () => {
  assert.deepEqual(
    buildRequestData([
      { name: 'transitionName', type: 'String', value: 'null', allowsNull: true },
      { name: 'transitionDuration', type: 'Number', value: 'null', allowsNull: true },
      { name: 'otherField', type: 'String', value: null },
    ]),
    { transitionName: null, transitionDuration: null },
  )
})
