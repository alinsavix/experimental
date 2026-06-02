export function ensureBaseStyles(doc) {
  if (doc.getElementById('bbrender-styles')) return;
  const style = doc.createElement('style');
  style.id = 'bbrender-styles';
  style.textContent = `
.bb-char { display: inline-block; transform-origin: 50% 55%; }
.bb-char.bb-space { min-width: 0.28em; }
.bb-char-text { white-space: pre; }
.bb-img {
  display: inline-block;
  max-width: 1.4em;
  max-height: 1.4em;
  object-fit: contain;
  vertical-align: -0.22em;
}
.bb-dropcap-first {
  float: left;
  font-size: calc(var(--bb-dropcap-lines, 3) * 0.86em);
  line-height: 0.76;
  margin: 0.03em 0.06em 0 0;
}
.bb-typewriter-cursor::after {
  content: "";
  display: inline-block;
  width: 0.08em;
  height: 0.95em;
  margin-left: 0.08em;
  vertical-align: -0.08em;
  background: currentColor;
  animation: bb-caret 0.8s steps(1, end) infinite;
}
.bb-hr { width: 100%; border: 0; border-top: 2px solid currentColor; opacity: 0.75; }
@keyframes bb-typewriter {
  to { opacity: 1; }
}
@keyframes bb-caret {
  0%, 49% { opacity: 1; }
  50%, 100% { opacity: 0; }
}
@keyframes bb-slide-in {
  from { opacity: 0; transform: translate(var(--bb-slide-x, 0), var(--bb-slide-y, 0)); }
  to { opacity: 1; transform: translate(0, 0); }
}
@keyframes bb-zoom {
  from { opacity: 0; transform: scale(var(--bb-zoom-from, 0)); }
  to { opacity: 1; transform: scale(var(--bb-zoom-to, 1)); }
}
@keyframes bb-enter-fade {
  from { opacity: 0; }
  to { opacity: 1; }
}
@keyframes bb-exit-fade {
  from { opacity: 1; }
  to { opacity: 0; }
}
@keyframes bb-enter-zoom {
  from { opacity: 0; transform: scale(var(--bb-enter-zoom-scale, 0.18)); }
  to { opacity: 1; transform: scale(1); }
}
@keyframes bb-exit-zoom {
  from { opacity: 1; transform: scale(1); }
  to { opacity: 0; transform: scale(var(--bb-exit-zoom-scale, 0.18)); }
}
@keyframes bb-enter-slide-left {
  from { opacity: 0; transform: translateX(-120px); }
  to { opacity: 1; transform: translateX(0); }
}
@keyframes bb-exit-slide-left {
  from { opacity: 1; transform: translateX(0); }
  to { opacity: 0; transform: translateX(-120px); }
}
@keyframes bb-enter-slide-right {
  from { opacity: 0; transform: translateX(120px); }
  to { opacity: 1; transform: translateX(0); }
}
@keyframes bb-exit-slide-right {
  from { opacity: 1; transform: translateX(0); }
  to { opacity: 0; transform: translateX(120px); }
}
@keyframes bb-enter-slide-up {
  from { opacity: 0; transform: translateY(-80px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes bb-exit-slide-up {
  from { opacity: 1; transform: translateY(0); }
  to { opacity: 0; transform: translateY(-80px); }
}
@keyframes bb-enter-slide-down {
  from { opacity: 0; transform: translateY(80px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes bb-exit-slide-down {
  from { opacity: 1; transform: translateY(0); }
  to { opacity: 0; transform: translateY(80px); }
}
@keyframes bb-wave {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(calc(var(--bb-wave-amp, 50px) * -1)); }
}
@keyframes bb-bounce {
  0%, 100% { transform: translateY(0) scaleY(1); }
  45% { transform: translateY(calc(var(--bb-bounce-amp, 20px) * -1)) scaleY(1.05); }
  60% { transform: translateY(0) scaleY(0.85); }
}
@keyframes bb-shake {
  0% { transform: translate(0, 0); }
  20% { transform: translate(var(--bb-shake-level, 5px), calc(var(--bb-shake-level, 5px) * -1)); }
  40% { transform: translate(calc(var(--bb-shake-level, 5px) * -1), var(--bb-shake-level, 5px)); }
  60% { transform: translate(var(--bb-shake-level, 5px), var(--bb-shake-level, 5px)); }
  80% { transform: translate(calc(var(--bb-shake-level, 5px) * -1), calc(var(--bb-shake-level, 5px) * -1)); }
  100% { transform: translate(0, 0); }
}
@keyframes bb-pulse {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.18); }
}
@keyframes bb-tornado {
  0% { transform: translate(var(--bb-tornado-radius, 10px), 0); }
  25% { transform: translate(0, calc(var(--bb-tornado-radius, 10px) * -1)); }
  50% { transform: translate(calc(var(--bb-tornado-radius, 10px) * -1), 0); }
  75% { transform: translate(0, var(--bb-tornado-radius, 10px)); }
  100% { transform: translate(var(--bb-tornado-radius, 10px), 0); }
}
@keyframes bb-rainbow {
  0% { color: #ff3b30; }
  16% { color: #ff9500; }
  32% { color: #ffcc00; }
  48% { color: #34c759; }
  64% { color: #32ade6; }
  80% { color: #5856d6; }
  100% { color: #ff3b30; }
}
@keyframes bb-blink {
  0%, 49% { opacity: 1; }
  50%, 100% { opacity: 0; }
}
@keyframes bb-flip-x {
  0%, 100% { transform: scaleY(1); }
  50% { transform: scaleY(-1); }
}
@keyframes bb-flip-y {
  0%, 100% { transform: scaleX(1); }
  50% { transform: scaleX(-1); }
}`;
  doc.head.appendChild(style);
}
