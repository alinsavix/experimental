import { clampNumber } from './utils.js';

const TRANSITION_PRESETS = new Set([
  'fade',
  'zoom',
  'slide-left',
  'slide-right',
  'slide-up',
  'slide-down'
]);

const EASE_VALUES = {
  linear: 'linear',
  in: 'cubic-bezier(.7,0,1,.5)',
  out: 'cubic-bezier(.16,1,.3,1)',
  'in-out': 'cubic-bezier(.65,0,.35,1)'
};

export function applyRootTransition(element, transition, durationMs) {
  clearRootTransition(element);

  const normalized = normalizeTransition(transition, durationMs);
  if (!normalized) return;

  element.dataset.bbTransition = 'root';
  element.style.transition = 'none';
  element.style.transformOrigin = normalized.origin;
  element.style.setProperty('--bb-enter-zoom-scale', String(normalized.inScale));
  element.style.setProperty('--bb-exit-zoom-scale', String(normalized.outScale));
  element.style.animation = normalized.animations.join(', ');
}

export function clearRootTransition(element) {
  delete element.dataset.bbTransition;
  element.style.animation = '';
  element.style.transition = '';
  element.style.transformOrigin = '';
  element.style.removeProperty('--bb-enter-zoom-scale');
  element.style.removeProperty('--bb-exit-zoom-scale');
}

export function normalizeTransition(transition, durationMs) {
  if (!transition || typeof transition !== 'object') return null;

  const totalMs = Math.max(1, Math.floor(clampNumber(transition.duration || durationMs, 1, 3600000, durationMs || 4000)));
  const inName = normalizePreset(transition.in || transition.enter || transition.entrance || transition.from);
  const outName = normalizePreset(transition.out || transition.exit || transition.leave || transition.to);
  const inTime = Math.floor(clampNumber(transition.inTime || transition.enterTime || transition.entranceTime, 1, totalMs, 400));
  const outTime = Math.floor(clampNumber(transition.outTime || transition.exitTime || transition.leaveTime, 1, totalMs, 400));
  const delay = Math.floor(clampNumber(transition.delay, 0, totalMs, 0));
  const ease = EASE_VALUES[String(transition.ease || 'out').toLowerCase()] || EASE_VALUES.out;
  const scale = clampNumber(transition.scale, 0.01, 5, 0.18);
  const inScale = clampNumber(transition.inScale || transition.enterScale, 0.01, 5, scale);
  const outScale = clampNumber(transition.outScale || transition.exitScale, 0.01, 5, scale);
  const animations = [];

  if (inName) {
    animations.push(animationValue('bb-enter-' + inName, inTime, ease, delay));
  }

  if (outName) {
    const exitDelay = Math.max(delay, totalMs - outTime);
    animations.push(animationValue('bb-exit-' + outName, outTime, ease, exitDelay));
  }

  if (!animations.length) return null;

  return {
    animations,
    duration: totalMs,
    in: inName,
    out: outName,
    inTime,
    outTime,
    delay,
    ease,
    inScale,
    outScale,
    origin: String(transition.origin || '50% 50%')
  };
}

function animationValue(name, durationMs, ease, delayMs) {
  return [
    name,
    durationMs + 'ms',
    ease,
    delayMs + 'ms',
    '1',
    'both'
  ].join(' ');
}

function normalizePreset(value) {
  const raw = String(value || '').trim().toLowerCase();
  if (!raw || raw === 'none' || raw === 'off' || raw === 'false') return '';
  const normalized = raw.replace(/_/g, '-');
  return TRANSITION_PRESETS.has(normalized) ? normalized : '';
}
