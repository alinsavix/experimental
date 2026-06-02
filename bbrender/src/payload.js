import { normalizeTagName } from './utils.js';

export function extractBBCode(payload) {
  if (typeof payload === 'string') return payload;
  if (!payload || typeof payload !== 'object') return '';

  const candidates = [
    payload.bbcode,
    payload.text,
    payload.message,
    payload.content,
    payload.value,
    payload.displayText,
    payload.data && payload.data.bbcode,
    payload.data && payload.data.text,
    payload.data && payload.data.message,
    payload.data && payload.data.content,
    payload.data && payload.data.value,
    payload.data && payload.data.displayText
  ];

  const match = candidates.find((value) => typeof value === 'string' && value.trim());
  return match ? match.trim() : '';
}

export function getCommand(payload) {
  if (!payload || typeof payload !== 'object') return '';
  return normalizeTagName(payload.command || payload.action || payload.requestType);
}

export function shouldHandlePayload(payload) {
  if (!payload || typeof payload !== 'object') return false;
  return normalizeTagName(payload.type) === 'bbcode.render';
}
