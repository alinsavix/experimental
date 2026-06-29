import test from 'node:test'
import assert from 'node:assert/strict'
import { summarizeEvent } from '../src/domain/eventSummary.mjs'

test('summarizes a scene item enable state change', () => {
  assert.equal(
    summarizeEvent('SceneItemEnableStateChanged', {
      sceneName: 'Gameplay',
      sceneUuid: 'ignored-uuid',
      sceneItemId: 12,
      sceneItemEnabled: false,
    }),
    'Scene: Gameplay · Item #12 · Disabled',
  )
})

test('summarizes transition and current scene events', () => {
  assert.equal(
    summarizeEvent('SceneTransitionStarted', { transitionName: 'Fade', transitionUuid: 'uuid' }),
    'Transition: Fade',
  )
  assert.equal(
    summarizeEvent('SceneTransitionEnded', { transitionName: 'Fade', transitionUuid: 'uuid' }),
    'Transition: Fade',
  )
  assert.equal(
    summarizeEvent('CurrentProgramSceneChanged', { sceneName: 'Starting Soon' }),
    'Scene: Starting Soon',
  )
})

test('formats common state changes in plain language', () => {
  assert.equal(
    summarizeEvent('InputMuteStateChanged', { inputName: 'Desktop Audio', inputMuted: true }),
    'Input: Desktop Audio · Muted',
  )
  assert.equal(
    summarizeEvent('SourceFilterEnableStateChanged', {
      sourceName: 'Camera',
      filterName: 'Color Correction',
      filterEnabled: true,
    }),
    'Source: Camera · Filter: Color Correction · Enabled',
  )
})

test('uses a compact primitive-field fallback without UUIDs or nested payloads', () => {
  assert.equal(
    summarizeEvent('FutureEvent', {
      sceneName: 'Main',
      sceneUuid: 'ignored',
      enabled: false,
      eventData: { nested: true },
      count: 0,
    }),
    'Scene: Main · enabled: No · count: 0',
  )
})
