import { normalizeTagName } from './utils.js';

export function getPayloadData(payload) {
  if (!payload || typeof payload !== 'object') return {};
  const data = payload.data;
  return data && typeof data === 'object' ? data : {};
}

export function extractBBCode(payload) {
  const data = getPayloadData(payload);

  const candidates = [
    data.bbcode,
    data.text,
    data.message,
    data.content,
    data.value,
    data.displayText
  ];

  const match = candidates.find((value) => typeof value === 'string' && value.trim());
  return match ? match.trim() : '';
}

export function getCommand(payload) {
  const data = getPayloadData(payload);
  return normalizeTagName(data.command || data.action || data.requestType);
}

export function shouldHandlePayload(payload) {
  if (!payload || typeof payload !== 'object') return false;
  return normalizeTagName(payload.type) === 'bbcode.render';
}
