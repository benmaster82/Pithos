'use strict';
/* Opening scene: once Pandora is out of the jar, tokens stream out of it, faster and faster,
   while the counter climbs to the measured speed. Decorative only: no network, no inference. */
(() => {
  const intro = document.querySelector('.intro');
  if (!intro) return;
  const layer = intro.querySelector('#intro-tokens');
  const counter = intro.querySelector('[data-count]');
  const target = Number(counter.dataset.count);
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const SVG = 'http://www.w3.org/2000/svg';
  const MOUTH = {x: 210, y: 146};
  const WORDS = ['Local', 'AI', 'on', 'the', 'laptop', 'you', 'already', 'have', '.', 'No', 'big', 'GPU',
                 'needed', ':', 'just', 'Pithos', '.'];
  const START = 1900, RAMP = 2400, CALM = 6500; // ms from load: first token, full speed, slow trickle
  const t0 = performance.now();
  let word = 0, next = 0, visible = true, frame = 0;

  const showFinal = () => { counter.textContent = String(target); };
  if (reduced.matches) { showFinal(); return; }

  const paused = () => document.documentElement.classList.contains('inference-motion-paused');
  const lerp = (a, b, k) => a + (b - a) * Math.min(Math.max(k, 0), 1);

  function emit(age) {
    if (layer.childElementCount > 40) return;
    const k = (age - START) / RAMP;
    const fast = age < CALM;
    const g = document.createElementNS(SVG, 'g');
    const tint = Math.random();
    g.setAttribute('class', 'intro-token' + (tint < .2 ? ' hot' : tint < .35 ? ' warm' : ''));
    const rect = document.createElementNS(SVG, 'rect');
    const text = document.createElementNS(SVG, 'text');
    text.textContent = WORDS[word++ % WORDS.length];
    text.setAttribute('text-anchor', 'middle');
    text.setAttribute('y', '4');
    g.append(rect, text);
    layer.append(g);
    const w = Math.max(22, text.getComputedTextLength() + 16);
    rect.setAttribute('x', String(-w / 2)); rect.setAttribute('y', '-11');
    rect.setAttribute('width', String(w)); rect.setAttribute('height', '22'); rect.setAttribute('rx', '7');
    // out of the right side of the mouth, then off to the right in a spray (clear of Pandora,
    // who stands above the jar), like a stream of generated text
    const angle = lerp(0.1, -0.85, Math.random());
    const reach = lerp(200, 300, Math.random());
    const x0 = MOUTH.x + 60, y0 = MOUTH.y + 8;
    const dx = Math.cos(angle) * reach, dy = Math.sin(angle) * reach;
    const duration = fast ? lerp(1700, 750, k) : 1700;
    g.animate([
      {transform: `translate(${x0}px,${y0}px) scale(.3)`, opacity: 0},
      {transform: `translate(${x0 + dx * .12}px,${y0 + dy * .12 - 10}px) scale(1)`, opacity: 1, offset: .15},
      {transform: `translate(${x0 + dx * .75}px,${y0 + dy * .75}px) scale(1)`, opacity: .9, offset: .75},
      {transform: `translate(${x0 + dx}px,${y0 + dy}px) scale(.95)`, opacity: 0}
    ], {duration, easing: 'cubic-bezier(.25,.6,.35,1)'}).onfinish = () => g.remove();
  }

  function tick(now) {
    frame = 0;
    if (!visible || document.hidden || paused()) return;
    const age = now - t0;
    if (age >= START) {
      const k = (age - START) / 1800;
      counter.textContent = String(Math.round(target * (1 - Math.pow(1 - Math.min(k, 1), 3))));
      if (now >= next) {
        emit(age);
        next = now + (age < CALM ? lerp(260, 55, (age - START) / RAMP) : 650);
      }
    }
    frame = requestAnimationFrame(tick);
  }
  function sync() {
    cancelAnimationFrame(frame); frame = 0;
    if (visible && !document.hidden && !paused()) frame = requestAnimationFrame(tick);
  }
  new IntersectionObserver(entries => { visible = entries[0].isIntersecting; sync(); }).observe(intro);
  document.addEventListener('visibilitychange', sync);
  // the page's "Pause animation" button toggles this class on <html>
  new MutationObserver(sync).observe(document.documentElement, {attributes: true, attributeFilter: ['class']});
  reduced.addEventListener('change', () => {
    if (!reduced.matches) return;
    visible = false; sync(); showFinal(); layer.replaceChildren();
  });
  sync();
})();
